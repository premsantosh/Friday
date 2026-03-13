"""
Coffee Machine Workflow

A dedicated workflow for a Shelly-controlled coffee machine that provides:

  1. Voice control  — direct ("make coffee") and indirect ("I'm in the mood for an
                      espresso", "our guests would like a latte")
  2. Power insights — "how much power is the coffee machine using?", "energy report"
  3. Weekly schedule — auto-on at a configured time that differs by weekday / weekend
  4. Smart idle monitor — proactively tells you when the machine has been idling too
                          long after a brew and offers to turn it off

Configuration (environment variables):
  COFFEE_MACHINE_IP             Required. Device IP, e.g. 192.168.1.100
  COFFEE_MACHINE_NAME           Optional. Human label (default: "coffee machine")
  COFFEE_MACHINE_WARMUP_MINUTES Warm-up time in minutes (default: 20)
  COFFEE_MACHINE_WEEKDAY_ON     HH:MM to auto-turn on weekdays  (default: "07:00")
  COFFEE_MACHINE_WEEKEND_ON     HH:MM to auto-turn on weekends  (default: "09:00")
  COFFEE_MACHINE_AUTO_ON        "true"/"false" — enable scheduled auto-on (default: true)
  COFFEE_MACHINE_IDLE_ALERT_MIN Minutes of idle before speaking an alert (default: 45)
  COFFEE_MACHINE_USERNAME       Optional auth username
  COFFEE_MACHINE_PASSWORD       Optional auth password
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Optional

from .base import Workflow, WorkflowResult, WorkflowStatus, WorkflowTrigger
from .shelly.devices.coffee import BrewState, CoffeeMachine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All coffee drinks / synonyms the assistant should recognise
# ---------------------------------------------------------------------------

COFFEE_DRINKS = frozenset({
    "coffee", "espresso", "latte", "cappuccino", "americano",
    "macchiato", "mocha", "flat white", "cortado", "lungo",
    "ristretto", "affogato", "cold brew", "pour over",
    "drip coffee", "black coffee", "cup of joe", "caffeine",
})

# Words / phrases that express a *desire* for something
DESIRE_PHRASES = frozenset({
    "mood for", "in the mood", "craving", "fancy", "would like",
    "want", "could use", "need", "love to have", "have a",
    "get a", "make me", "prepare", "get some", "have some",
})

# Words that indicate the request is for other people
GUEST_WORDS = frozenset({
    "guest", "guests", "visitor", "visitors", "friend", "friends",
    "family", "everyone", "they", "them", "people", "company",
    "colleague", "colleagues",
})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoffeeSchedule:
    """When the coffee machine should automatically power on each day."""

    weekday_on_time: str = "07:00"   # HH:MM — Monday–Friday
    weekend_on_time: str = "09:00"   # HH:MM — Saturday–Sunday
    auto_on_enabled: bool = True

    def on_time_for_today(self) -> str:
        if date.today().weekday() < 5:
            return self.weekday_on_time
        return self.weekend_on_time

    def on_datetime_for_today(self) -> datetime:
        h, m = map(int, self.on_time_for_today().split(":"))
        return datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)

    def day_label(self) -> str:
        return "weekday" if date.today().weekday() < 5 else "weekend"


@dataclass
class CoffeeMonitorConfig:
    idle_alert_minutes: int = 45
    poll_interval_seconds: int = 60
    schedule_window_minutes: int = 5


# ---------------------------------------------------------------------------
# Background monitor
# ---------------------------------------------------------------------------


class CoffeeMachineMonitor:
    """
    Daemon thread that polls the coffee machine every minute.

    Responsibilities:
      - Auto-turn-on at the scheduled time (if enabled)
      - Detect when a brew cycle finishes and the machine enters idle
      - Speak a proactive alert when the machine has been idling too long
    """

    def __init__(
        self,
        device: CoffeeMachine,
        schedule: CoffeeSchedule,
        config: CoffeeMonitorConfig,
        warmup_minutes: int,
        speak_callback: Callable[[str], None],
    ):
        self.device = device
        self.schedule = schedule
        self.config = config
        self.warmup_minutes = warmup_minutes
        self._speak = speak_callback

        self._prev_brew_state: BrewState = BrewState.OFF
        self._idle_since: Optional[datetime] = None
        self._idle_alert_fired: bool = False
        self._last_auto_on_date: Optional[date] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="coffee-monitor")
        self._thread.start()
        logger.info("Coffee machine monitor started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_loop())
        finally:
            loop.close()

    async def _async_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Coffee monitor tick failed — will retry next interval.")
            for _ in range(self.config.poll_interval_seconds):
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(1)

    async def _tick(self) -> None:
        brew_state, _ = await self.device.get_brew_state()
        now = datetime.now()

        if self.schedule.auto_on_enabled:
            await self._check_schedule(brew_state, now)

        if brew_state != self._prev_brew_state:
            self._on_state_change(brew_state, self._prev_brew_state, now)
            self._prev_brew_state = brew_state

        if brew_state == BrewState.IDLE and not self._idle_alert_fired:
            if self._idle_since is not None:
                idle_minutes = (now - self._idle_since).total_seconds() / 60
                if idle_minutes >= self.config.idle_alert_minutes:
                    await self._fire_idle_alert(idle_minutes)

    def _on_state_change(self, new: BrewState, old: BrewState, now: datetime) -> None:
        if new == BrewState.BREWING:
            self._idle_since = None
            self._idle_alert_fired = False
        elif new == BrewState.IDLE:
            if old in (BrewState.BREWING, BrewState.OFF):
                self._idle_since = now
        elif new == BrewState.OFF:
            self._idle_since = None
            self._idle_alert_fired = False

    async def _check_schedule(self, current_state: BrewState, now: datetime) -> None:
        today = now.date()
        if self._last_auto_on_date == today:
            return
        scheduled = self.schedule.on_datetime_for_today()
        window = timedelta(minutes=self.config.schedule_window_minutes)
        if scheduled <= now < scheduled + window:
            if current_state == BrewState.OFF:
                try:
                    await self.device.turn_on()
                    self._last_auto_on_date = today
                    self._speak(
                        f"Good morning, sir. I have started the {self.device.name} on schedule. "
                        f"It should be ready to brew in approximately {self.warmup_minutes} minutes."
                    )
                except Exception as exc:
                    logger.error("Auto-on failed: %s", exc)
            else:
                self._last_auto_on_date = today

    async def _fire_idle_alert(self, idle_minutes: float) -> None:
        self._idle_alert_fired = True
        self._speak(
            f"Sir, the {self.device.name} has been idling for {int(idle_minutes)} minutes. "
            f"Shall I turn it off? Just say 'turn off the coffee machine'."
        )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class CoffeeMachineWorkflow(Workflow):
    """
    Voice workflow for a Shelly-controlled coffee machine.

    Handles direct commands, indirect coffee-drink expressions, power/energy
    stats, schedule queries and updates, and background smart monitoring.
    """

    def __init__(
        self,
        device: Optional[CoffeeMachine] = None,
        schedule: Optional[CoffeeSchedule] = None,
        monitor_config: Optional[CoffeeMonitorConfig] = None,
        warmup_minutes: int = 20,
    ):
        self.device = device or self._device_from_env()
        self.schedule = schedule or self._schedule_from_env()
        self._monitor_config = monitor_config or self._monitor_config_from_env()
        self.warmup_minutes = warmup_minutes or int(
            os.getenv("COFFEE_MACHINE_WARMUP_MINUTES", "20")
        )
        self._monitor: Optional[CoffeeMachineMonitor] = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _device_from_env() -> Optional[CoffeeMachine]:
        ip = os.getenv("COFFEE_MACHINE_IP", "")
        if not ip:
            return None
        return CoffeeMachine(
            host=ip,
            name=os.getenv("COFFEE_MACHINE_NAME", "coffee machine"),
            username=os.getenv("COFFEE_MACHINE_USERNAME"),
            password=os.getenv("COFFEE_MACHINE_PASSWORD"),
        )

    @staticmethod
    def _schedule_from_env() -> CoffeeSchedule:
        return CoffeeSchedule(
            weekday_on_time=os.getenv("COFFEE_MACHINE_WEEKDAY_ON", "07:00"),
            weekend_on_time=os.getenv("COFFEE_MACHINE_WEEKEND_ON", "09:00"),
            auto_on_enabled=os.getenv("COFFEE_MACHINE_AUTO_ON", "true").lower()
            in ("1", "true", "yes"),
        )

    @staticmethod
    def _monitor_config_from_env() -> CoffeeMonitorConfig:
        return CoffeeMonitorConfig(
            idle_alert_minutes=int(os.getenv("COFFEE_MACHINE_IDLE_ALERT_MIN", "45")),
        )

    # ------------------------------------------------------------------
    # Monitor lifecycle
    # ------------------------------------------------------------------

    def start_monitor(self, speak_callback: Callable[[str], None]) -> None:
        if self.device is None:
            return
        self._monitor = CoffeeMachineMonitor(
            device=self.device,
            schedule=self.schedule,
            config=self._monitor_config,
            warmup_minutes=self.warmup_minutes,
            speak_callback=speak_callback,
        )
        self._monitor.start()

    def stop_monitor(self) -> None:
        if self._monitor:
            self._monitor.stop()
            self._monitor = None

    # ------------------------------------------------------------------
    # Workflow interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "coffee_machine"

    @property
    def description(self) -> str:
        label = self.device.name if self.device else "coffee machine"
        return (
            f"Control and monitor the {label}. "
            f"Warm-up time: ~{self.warmup_minutes} min. "
            f"Schedule: weekdays {self.schedule.weekday_on_time}, "
            f"weekends {self.schedule.weekend_on_time}."
        )

    @property
    def trigger(self) -> WorkflowTrigger:
        drink_list = "|".join(sorted(COFFEE_DRINKS, key=len, reverse=True))
        return WorkflowTrigger(
            keywords=[
                # Drink names — single-word ones that are safe as standalone keywords
                "coffee", "espresso", "latte", "cappuccino", "americano",
                "macchiato", "mocha", "lungo", "cortado", "caffeine",
                # Device / action words
                "brew", "brewing", "coffee machine", "coffee maker",
            ],
            patterns=[
                # Direct control
                r"(make|start|brew|begin|prepare|fire up).{0,15}coffee",
                r"(turn|switch|put).{0,10}(on|off).{0,20}(coffee|machine|espresso)",
                r"(stop|shut|kill|turn off).{0,15}(coffee|machine)",
                # Desire / craving — "I'm in the mood for an espresso"
                rf"(i.?m|i am|i.?d|i would).{{0,25}}({drink_list})",
                rf"(in the mood for|craving|fancy|dying for).{{0,20}}({drink_list})",
                rf"(could use|could really use|could do with).{{0,15}}({drink_list})",
                # Guest / third-party requests — "our guests would like a latte"
                rf"(guest|guests|visitor|visitors|friend|friends|family|they|everyone|someone).{{0,30}}({drink_list})",
                rf"({drink_list}).{{0,20}}(for|to).{{0,15}}(guest|visitor|friend|family|everyone|them)",
                # Any drink name as a standalone request ("Espresso please")
                rf"^({drink_list})\b",
                rf"\b({drink_list})\s+(please|now|anyone|for me|for us)\b",
                # Status / info
                r"(is|are).{0,15}(coffee|machine).{0,15}(on|off|ready|brewing|done|finished|warm)",
                r"(how long|when).{0,20}(coffee|machine|espresso).{0,20}(ready|done|warm)",
                r"(power|energy|watt|consumption|stats|usage).{0,20}(coffee|machine)",
                r"coffee.{0,20}(power|energy|watt|consumption|stats|usage)",
                # Schedule
                r"coffee.{0,15}schedule",
                r"(what time|when).{0,15}coffee.{0,15}(on|start|come on)",
                r"(change|set|update).{0,15}coffee.{0,15}(time|schedule|morning)",
            ],
            examples=[
                "Make coffee",
                "Start brewing",
                "Turn off the coffee machine",
                "I'm in the mood for an espresso",
                "Our guests would like a latte",
                "Could I get a cappuccino?",
                "Is the coffee machine ready?",
                "How much power is the coffee machine using?",
                "What time does my coffee machine come on?",
                "Change my coffee schedule to 8am on weekdays",
            ],
        )

    def get_context_for_llm(self) -> str:
        wd = self.schedule.weekday_on_time
        we = self.schedule.weekend_on_time
        return f"""Workflow: coffee_machine
Description: Controls a Shelly smart plug connected to a coffee machine.

KEY FACTS (always apply these when the topic is coffee):
- Warm-up time: approximately {self.warmup_minutes} minutes from cold to brew-ready.
  Always mention this when starting the machine so the user knows when to expect their drink.
- Auto-start schedule: {wd} on weekdays, {we} on weekends.
- Any mention of a coffee drink (espresso, latte, cappuccino, americano, macchiato,
  mocha, flat white, cortado, lungo, etc.) is an implicit request to start the machine —
  treat it the same as "make coffee".
- When guests or visitors are mentioned alongside a coffee drink, start the machine and
  tailor the response accordingly ("for your guests").
- If the user asks whether coffee is ready and the machine was just turned on, reference
  the {self.warmup_minutes}-minute warm-up. If it has been on longer, check the brew state.

Example triggers for this workflow:
  "Make coffee" / "I'm in the mood for an espresso" / "Our guests would like a latte"
  "Is coffee ready?" / "Turn off the coffee machine" / "Coffee stats"
"""

    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        if self.device is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="The coffee machine is not configured, sir. Please set COFFEE_MACHINE_IP.",
                error="COFFEE_MACHINE_IP not set",
            )

        # Use the workflow's own intent classifier — richer than the generic entity extractor
        classified = self._classify_intent(intent)
        action = classified["action"]
        is_indirect = classified["is_indirect"]
        for_guests = classified["for_guests"]

        try:
            if action == "on":
                return await self._handle_turn_on(is_indirect=is_indirect, for_guests=for_guests)
            elif action == "off":
                return await self._handle_turn_off()
            elif action in ("stats", "power", "energy", "consumption"):
                return await self._handle_stats()
            elif action == "schedule":
                return self._handle_schedule_query()
            elif action == "set_schedule":
                return self._handle_schedule_update(entities)
            else:
                return await self._handle_status()

        except Exception:
            logger.exception("CoffeeMachineWorkflow error")
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=(
                    f"I was unable to reach the {self.device.name}, sir. "
                    f"Please check it is on the network."
                ),
                error="Device unreachable",
            )

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, intent: str) -> Dict[str, Any]:
        """
        Map raw user intent to a structured action dict.

        Returns a dict with keys:
          action      — "on" | "off" | "status" | "stats" | "schedule" | "set_schedule"
          is_indirect — True when the user expressed a coffee desire rather than a
                        direct device command (affects response wording)
          for_guests  — True when the request is on behalf of guests / other people
        """
        t = intent.lower()
        result: Dict[str, Any] = {
            "action": "status",
            "is_indirect": False,
            "for_guests": False,
        }

        # ---- Off commands ------------------------------------------------
        OFF_SIGNALS = ("turn off", "switch off", "shut off", "shut down", "stop", "kill")
        if any(sig in t for sig in OFF_SIGNALS):
            result["action"] = "off"
            return result

        # ---- Stats / power queries ---------------------------------------
        STAT_WORDS = ("power", "watt", "energy", "consumption", "stat", "usage", "electricity", "kwh")
        if any(w in t for w in STAT_WORDS):
            result["action"] = "stats"
            return result

        # ---- Schedule update ---------------------------------------------
        CHANGE_WORDS = ("change", "set", "update", "move", "reschedule")
        SCHED_WORDS = ("schedule", "time", "morning", "o'clock", "oclock")
        if any(c in t for c in CHANGE_WORDS) and any(s in t for s in SCHED_WORDS):
            result["action"] = "set_schedule"
            return result

        # ---- Schedule query ----------------------------------------------
        if "schedule" in t or (
            any(w in t for w in ("what time", "when", "what hour")) and
            any(w in t for w in ("coffee", "machine", "come on", "start", "on"))
        ):
            result["action"] = "schedule"
            return result

        # ---- Status check ------------------------------------------------
        STATUS_SIGNALS = (
            "is it", "is the", "are you", "ready", "done", "finished",
            "status", "how long", "when will", "how warm",
        )
        if any(sig in t for sig in STATUS_SIGNALS):
            result["action"] = "status"
            return result

        # ---- Direct on commands ------------------------------------------
        ON_SIGNALS = ("turn on", "switch on", "start", "begin", "make", "brew", "prepare", "fire up")
        if any(sig in t for sig in ON_SIGNALS):
            result["action"] = "on"
            return result

        # ---- Indirect desire expressions ---------------------------------
        has_drink = any(drink in t for drink in COFFEE_DRINKS)
        has_desire = any(phrase in t for phrase in DESIRE_PHRASES)
        has_guest_word = any(word in t for word in GUEST_WORDS)

        if has_drink:
            result["action"] = "on"
            result["is_indirect"] = True
            # Determine if the request is for guests rather than the user themselves.
            # Use explicit first-person desire phrases; "my friends/guests" still counts
            # as a guest request even though "my" appears.
            import re as _re
            first_person_desire = bool(_re.search(
                r"\b(i (want|need|would|could|am|fancy|crave)|i'm (in the mood|craving)|"
                r"for me\b|myself\b)",
                t,
            ))
            result["for_guests"] = has_guest_word and not first_person_desire
            return result

        # Fallback — treat any coffee-adjacent phrase as a status check
        return result

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _handle_turn_on(self, is_indirect: bool = False, for_guests: bool = False) -> WorkflowResult:
        brew_state, _ = await self.device.get_brew_state()

        if brew_state == BrewState.BREWING:
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=f"The {self.device.name} is already mid-brew, sir — your coffee is on its way.",
                data={"action": "on", "already_brewing": True},
            )
        if brew_state == BrewState.IDLE:
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=(
                    f"The {self.device.name} is already on and warmed up, sir — "
                    f"ready to brew whenever you are."
                ),
                data={"action": "on", "already_on": True},
            )

        await self.device.turn_on()

        if for_guests:
            message = (
                f"Very good, sir. I have started the {self.device.name} for your guests. "
                f"It should be ready to brew in approximately {self.warmup_minutes} minutes."
            )
        elif is_indirect:
            message = (
                f"Right away, sir. The {self.device.name} is now on. "
                f"It will be ready in approximately {self.warmup_minutes} minutes."
            )
        else:
            message = (
                f"The {self.device.name} is now on, sir. "
                f"It should be ready to brew in approximately {self.warmup_minutes} minutes."
            )

        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=message,
            data={"action": "on", "is_indirect": is_indirect, "for_guests": for_guests},
        )

    async def _handle_turn_off(self) -> WorkflowResult:
        brew_state, _ = await self.device.get_brew_state()
        if brew_state == BrewState.OFF:
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=f"The {self.device.name} is already off, sir.",
                data={"action": "off", "already_off": True},
            )
        if brew_state == BrewState.BREWING:
            await self.device.turn_off()
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=(
                    f"Noted, sir — the {self.device.name} was mid-brew, but I have switched "
                    f"it off as requested."
                ),
                data={"action": "off", "interrupted_brew": True},
            )
        await self.device.turn_off()
        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=f"The {self.device.name} has been switched off, sir.",
            data={"action": "off"},
        )

    async def _handle_status(self) -> WorkflowResult:
        brew_state, watts = await self.device.get_brew_state()

        if brew_state == BrewState.BREWING:
            msg = f"The {self.device.name} is currently brewing, sir."
            if watts:
                msg += f" Drawing {watts:.0f} W."
        elif brew_state == BrewState.IDLE:
            msg = f"The {self.device.name} is on and warmed up, sir — ready to brew."
            if watts:
                msg += f" Drawing {watts:.0f} W."
            if self._monitor and self._monitor._idle_since:
                idle_min = int(
                    (datetime.now() - self._monitor._idle_since).total_seconds() / 60
                )
                if idle_min > 0:
                    msg += f" It has been idling for {idle_min} minute{'s' if idle_min != 1 else ''}."
        else:
            msg = f"The {self.device.name} is off, sir."

        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=msg,
            data={"brew_state": brew_state.value, "power_w": watts},
        )

    async def _handle_stats(self) -> WorkflowResult:
        stats = await self.device.get_stats()
        message = self.device.format_stats(stats)
        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=message,
            data={"stats": stats},
        )

    def _handle_schedule_query(self) -> WorkflowResult:
        wd = self.schedule.weekday_on_time
        we = self.schedule.weekend_on_time
        enabled = self.schedule.auto_on_enabled
        today_time = self.schedule.on_time_for_today()
        day_label = self.schedule.day_label()

        if not enabled:
            msg = (
                f"The scheduled auto-on is currently disabled, sir. It would otherwise "
                f"come on at {wd} on weekdays and {we} on weekends."
            )
        else:
            msg = (
                f"The {self.device.name} is scheduled to come on at {wd} on weekdays "
                f"and {we} on weekends, sir. "
                f"Today being a {day_label}, the time is {today_time}."
            )

        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=msg,
            data={"weekday_on": wd, "weekend_on": we, "auto_on_enabled": enabled},
        )

    def _handle_schedule_update(self, entities: Dict[str, Any]) -> WorkflowResult:
        new_time: Optional[str] = entities.get("time")
        day_type: str = entities.get("day_type", "").lower()

        if not new_time:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="I did not catch the time you would like, sir. Could you repeat that?",
                error="No time entity in request",
            )

        try:
            h, m = (int(x) for x in new_time.split(":"))
            assert 0 <= h <= 23 and 0 <= m <= 59
            formatted = f"{h:02d}:{m:02d}"
        except Exception:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=f"I am afraid '{new_time}' does not look like a valid time, sir.",
                error=f"Invalid time: {new_time}",
            )

        WEEKEND_WORDS = {"weekend", "weekends", "saturday", "sunday"}
        WEEKDAY_WORDS = {"weekday", "weekdays", "monday", "tuesday", "wednesday", "thursday", "friday"}

        if day_type in WEEKEND_WORDS:
            self.schedule.weekend_on_time = formatted
            msg = f"Done, sir. The {self.device.name} will come on at {formatted} on weekends."
        elif day_type in WEEKDAY_WORDS:
            self.schedule.weekday_on_time = formatted
            msg = f"Done, sir. The {self.device.name} will come on at {formatted} on weekdays."
        else:
            self.schedule.weekday_on_time = formatted
            self.schedule.weekend_on_time = formatted
            msg = (
                f"Done, sir. The {self.device.name} will now come on at {formatted} every day. "
                f"Should you wish different times for weekdays and weekends, just say the word."
            )

        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=msg,
            data={"weekday_on": self.schedule.weekday_on_time, "weekend_on": self.schedule.weekend_on_time},
        )
