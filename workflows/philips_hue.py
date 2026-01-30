"""
Philips Hue Integration

This module provides integration with Philips Hue Bridge for controlling
smart lights directly via the Hue REST API.

Prerequisites:
- Philips Hue Bridge accessible on the network
- API username (press the bridge button and create one via the API)
- pip install aiohttp

Configuration:
- Set HUE_BRIDGE_IP environment variable (e.g., 10.0.0.242)
- Set HUE_BRIDGE_PORT environment variable (default: 443)
- Set HUE_USERNAME environment variable (your API username)
"""

import os
import ssl
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .base import Workflow, WorkflowResult, WorkflowStatus, WorkflowTrigger

logger = logging.getLogger(__name__)


@dataclass
class PhilipsHueConfig:
    """Configuration for Philips Hue Bridge connection."""
    bridge_ip: str = ""
    bridge_port: int = 443
    username: str = ""

    @classmethod
    def from_env(cls) -> "PhilipsHueConfig":
        return cls(
            bridge_ip=os.getenv("HUE_BRIDGE_IP", ""),
            bridge_port=int(os.getenv("HUE_BRIDGE_PORT", "443")),
            username=os.getenv("HUE_USERNAME", ""),
        )


class PhilipsHueClient:
    """
    Async REST client for the Philips Hue Bridge.

    Uses HTTPS with SSL verification disabled (Hue Bridge uses a self-signed cert).
    """

    def __init__(self, config: PhilipsHueConfig):
        self.config = config
        self.base_url = f"https://{config.bridge_ip}:{config.bridge_port}/api/{config.username}"
        self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        # Name-to-ID lookup maps populated by discover()
        self._light_names: Dict[str, str] = {}
        self._group_names: Dict[str, str] = {}
        self._discovered = False

    async def discover(self):
        """Fetch all lights and groups, building name-to-ID lookup maps."""
        lights = await self.get_lights()
        groups = await self.get_groups()

        self._light_names = {
            info.get("name", "").lower(): lid
            for lid, info in lights.items()
        }
        self._group_names = {
            info.get("name", "").lower(): gid
            for gid, info in groups.items()
        }
        self._discovered = True
        logger.info(
            "Hue discovery complete: %d lights, %d groups",
            len(self._light_names),
            len(self._group_names),
        )

    async def get_lights(self) -> Dict[str, Any]:
        """GET /lights — returns all lights keyed by ID."""
        import aiohttp

        url = f"{self.base_url}/lights"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=self._ssl_context) as resp:
                data = await resp.json()
                self._check_hue_errors(data)
                return data

    async def get_groups(self) -> Dict[str, Any]:
        """GET /groups — returns all groups/rooms keyed by ID."""
        import aiohttp

        url = f"{self.base_url}/groups"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=self._ssl_context) as resp:
                data = await resp.json()
                self._check_hue_errors(data)
                return data

    async def set_light_state(self, light_id: str, state: Dict[str, Any]) -> List[Dict]:
        """PUT /lights/{id}/state — set state for a single light."""
        import aiohttp

        url = f"{self.base_url}/lights/{light_id}/state"
        async with aiohttp.ClientSession() as session:
            async with session.put(url, json=state, ssl=self._ssl_context) as resp:
                data = await resp.json()
                self._check_hue_errors(data)
                return data

    async def set_group_action(self, group_id: str, action: Dict[str, Any]) -> List[Dict]:
        """PUT /groups/{id}/action — set action for a group/room."""
        import aiohttp

        url = f"{self.base_url}/groups/{group_id}/action"
        async with aiohttp.ClientSession() as session:
            async with session.put(url, json=action, ssl=self._ssl_context) as resp:
                data = await resp.json()
                self._check_hue_errors(data)
                return data

    def find_light_id(self, name: str) -> Optional[str]:
        """Case-insensitive light name lookup."""
        return self._light_names.get(name.lower())

    def find_group_id(self, name: str) -> Optional[str]:
        """Case-insensitive group/room name lookup."""
        return self._group_names.get(name.lower())

    @staticmethod
    def _check_hue_errors(data: Any):
        """
        Hue API returns HTTP 200 even for errors. The response is a list
        containing dicts with an 'error' key when something goes wrong.
        """
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "error" in item:
                    err = item["error"]
                    raise RuntimeError(
                        f"Hue API error {err.get('type', '?')}: {err.get('description', 'unknown')}"
                    )


class PhilipsHueLightsWorkflow(Workflow):
    """
    Control lights through a Philips Hue Bridge.

    Prefers group (room) control and falls back to individual light lookup.
    Group "0" is the special Hue group representing all lights.
    Supports mood-based lighting scenes.
    """

    # Mood-to-Hue-state mappings.
    # bri: 0-254, ct: color temp in mireds (153=cool daylight, 500=warm candlelight),
    # hue: 0-65535, sat: 0-254
    MOODS: Dict[str, Dict[str, Any]] = {
        "romantic": {"on": True, "bri": 77, "ct": 500},            # dim, warm candlelight
        "relax": {"on": True, "bri": 127, "ct": 447},              # medium-low, warm
        "energize": {"on": True, "bri": 254, "ct": 250},           # full bright, cool white
        "party": {"on": True, "bri": 254, "sat": 254, "hue": 47000},  # bright, vibrant color
        "bedtime": {"on": True, "bri": 25, "ct": 500},             # very dim, warmest
        "focus": {"on": True, "bri": 254, "ct": 300},              # bright, neutral white
        "movie": {"on": True, "bri": 38, "ct": 400},               # low, warm-ish
        "morning": {"on": True, "bri": 200, "ct": 350},            # bright-ish, neutral warm
    }

    MOOD_RESPONSES: Dict[str, str] = {
        "romantic": "I've set the mood for a romantic evening, sir. Dim and warm, as one does.",
        "relax": "The lights are now set for relaxation, sir. Do take it easy.",
        "energize": "Bright and invigorating, sir. Ready to conquer the day.",
        "party": "Party mode engaged, sir. I trust the neighbours have been forewarned.",
        "bedtime": "The lights are dimmed for bedtime, sir. Sweet dreams.",
        "focus": "Bright and focused, sir. Productivity awaits.",
        "movie": "Lights dimmed for cinema mode, sir. Popcorn is your department.",
        "morning": "Good morning, sir. The lights are set to ease you into the day.",
    }

    def __init__(self, client: Optional[PhilipsHueClient] = None):
        self.client = client or self._create_default_client()

    def _create_default_client(self) -> Optional[PhilipsHueClient]:
        """Create client from environment variables."""
        config = PhilipsHueConfig.from_env()
        if config.bridge_ip and config.username:
            return PhilipsHueClient(config)
        return None

    @property
    def name(self) -> str:
        return "hue_lights"

    @property
    def description(self) -> str:
        return "Control lights through Philips Hue Bridge"

    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=[
                "light", "lights", "lamp", "illuminate", "bright", "dim",
                "romantic", "romance", "relax", "chill", "calm",
                "energize", "energetic", "party", "celebrate",
                "bedtime", "sleep", "going to bed", "good night",
                "focus", "concentrate", "study",
                "movie", "cinema", "movie night",
                "morning", "wake up", "good morning",
                "mood",
            ],
            patterns=[
                r"turn (on|off) .*(light|lamp)",
                r"(light|lamp).* (on|off)",
                r"dim .*(light|lamp)",
                r"set .*(light|lamp).* to",
                r"(bright|dark)en",
                r"(romantic|relax|party|bedtime|focus|movie|morning|energize)\s*(mood|mode|setting|vibe)?",
                r"(i am |i'm |feeling |in a ).*(romantic|relaxed|sleepy|party|focused)",
                r"(going to |time for ).*(bed|sleep)",
                r"(good\s*(night|morning))",
            ],
            examples=[
                "Turn on the living room lights",
                "Dim the bedroom lights to 50%",
                "Turn off all the lights",
                "I am in a romantic mood",
                "Set the lights to party mode",
                "I'm going to bed",
                "Movie night",
                "Good morning",
            ],
        )

    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        if self.client is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="Philips Hue is not configured. Please set HUE_BRIDGE_IP and HUE_USERNAME environment variables.",
                error="No Hue client",
            )

        # Lazy discovery on first call
        if not self.client._discovered:
            try:
                await self.client.discover()
            except Exception as e:
                return WorkflowResult(
                    status=WorkflowStatus.FAILURE,
                    message=f"I was unable to reach the Hue Bridge, sir. The error was: {e}",
                    error=str(e),
                )

        action = entities.get("action", "toggle")
        room = entities.get("room", "all").lower()
        brightness = entities.get("brightness")
        mood = entities.get("mood")

        try:
            # Build the state/action payload
            if action == "mood" and mood:
                state = self.MOODS.get(mood)
                if state is None:
                    return WorkflowResult(
                        status=WorkflowStatus.FAILURE,
                        message=f"I don't have a lighting preset for '{mood}', sir.",
                        error=f"Unknown mood: {mood}",
                    )
                state = dict(state)  # copy so we don't mutate the class-level dict
            elif action == "on":
                state = {"on": True}
                if brightness is not None:
                    state["bri"] = self._pct_to_bri(brightness)
            elif action == "off":
                state = {"on": False}
            elif action == "dim" and brightness is not None:
                state = {"on": True, "bri": self._pct_to_bri(brightness)}
            else:
                # Toggle: Hue doesn't have a native toggle on groups, but we
                # can send on=True as a reasonable default for "toggle"
                state = {"on": True}

            # Determine target: prefer group (room), fallback to individual light
            # Mood commands with no room specified target all lights
            if room in ("all", "everywhere", "everything", "every room"):
                await self.client.set_group_action("0", state)
            else:
                group_id = self.client.find_group_id(room)
                if group_id is not None:
                    await self.client.set_group_action(group_id, state)
                else:
                    light_id = self.client.find_light_id(room)
                    if light_id is not None:
                        await self.client.set_light_state(light_id, state)
                    else:
                        return WorkflowResult(
                            status=WorkflowStatus.FAILURE,
                            message=f"I could not find a light or room called '{room}', sir.",
                            error=f"Unknown light/group: {room}",
                        )

            # Build butler-style response
            if action == "mood" and mood:
                message = self.MOOD_RESPONSES.get(mood, f"I've set the {mood} mood, sir.")
            else:
                message = self._build_response(action, room, brightness)

            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=message,
                data={"room": room, "action": action, "brightness": brightness, "mood": mood},
            )

        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=f"I encountered difficulty with the lights, sir. The error was: {e}",
                error=str(e),
            )

    @staticmethod
    def _pct_to_bri(pct: int) -> int:
        """Convert brightness 0-100% to Hue's 0-254 scale."""
        return max(0, min(254, round(pct * 254 / 100)))

    @staticmethod
    def _build_response(action: str, room: str, brightness: Optional[int]) -> str:
        if action == "on":
            if brightness is not None:
                return f"I've set the {room} lights to {brightness}%, sir."
            return f"I've illuminated the {room}, sir."
        elif action == "off":
            return f"The {room} is now dark, sir. Do try not to stub your toe."
        elif action == "dim" and brightness is not None:
            return f"I've dimmed the {room} lights to {brightness}%, sir."
        return f"I've toggled the {room} lights, sir."
