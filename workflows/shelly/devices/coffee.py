"""
Coffee Machine device built on ShellyPlug.

Adds power-profile-based brew state detection, which the CoffeeMachineWorkflow
uses to decide when a brew cycle has finished and the machine is idling.

Typical power profiles (adjust thresholds to match your machine):
  Brewing / heating:  >= 800 W
  Idle / warming:       5 – 799 W  (display, keep-warm plate)
  Off / standby:       <  5 W
"""

from enum import Enum
from typing import Any, Dict, Optional

from .plug import ShellyPlug


class BrewState(Enum):
    OFF = "off"        # Machine switched off (or deep standby)
    IDLE = "idle"      # On but not brewing — warm, ready, or just sitting there
    BREWING = "brewing"  # Actively heating water


class CoffeeMachine(ShellyPlug):
    """
    Shelly Plug controlling a coffee machine.

    Inherits all on/off/stats behaviour from ShellyPlug and adds:
      - Power-state classification (off / idle / brewing)
      - Coffee-specific stats formatting
    """

    # Watts — above this the machine is considered to be actively heating/brewing
    POWER_BREWING_THRESHOLD: float = 800.0
    # Watts — above this the machine is on (display, warm plate); below is standby/off
    POWER_IDLE_THRESHOLD: float = 5.0

    def __init__(self, host: str, name: str = "coffee machine", **kwargs):
        super().__init__(host, name=name, **kwargs)

    @property
    def device_type(self) -> str:
        return "coffee_machine"

    # ------------------------------------------------------------------
    # Brew-state classification
    # ------------------------------------------------------------------

    def classify_power(self, watts: Optional[float]) -> BrewState:
        """Classify a wattage reading into a brew state."""
        if watts is None or watts < self.POWER_IDLE_THRESHOLD:
            return BrewState.OFF
        if watts >= self.POWER_BREWING_THRESHOLD:
            return BrewState.BREWING
        return BrewState.IDLE

    async def get_brew_state(self) -> tuple[BrewState, Optional[float]]:
        """
        Query the device and return (BrewState, current_watts).
        Returns (OFF, None) if the output is off regardless of power reading.
        """
        status = await self.get_switch_status()
        if not status.output:
            return BrewState.OFF, status.apower
        state = self.classify_power(status.apower)
        return state, status.apower

    # ------------------------------------------------------------------
    # Formatted output
    # ------------------------------------------------------------------

    def format_stats(self, stats: Dict[str, Any]) -> str:
        label = self.name or self.host
        power = stats.get("power_w")

        if not stats.get("output"):
            return f"The {label} is currently off, sir."

        brew_state = self.classify_power(power)
        parts: list[str] = []

        if brew_state == BrewState.BREWING:
            parts.append(f"The {label} is busy brewing, sir.")
        elif brew_state == BrewState.IDLE:
            parts.append(f"The {label} is on and idling — warmed up and ready, sir.")
        else:
            parts.append(f"The {label} is on, sir.")

        if power is not None:
            parts.append(f"Current draw: {power} W.")
        if "voltage_v" in stats:
            parts.append(f"Voltage: {stats['voltage_v']} V.")
        if "energy_wh" in stats:
            kwh = round(stats["energy_wh"] / 1000, 3)
            parts.append(f"Total energy consumed since last reset: {kwh} kWh.")
        if "temp_c" in stats:
            parts.append(f"Device temperature: {stats['temp_c']}°C.")

        return " ".join(parts)
