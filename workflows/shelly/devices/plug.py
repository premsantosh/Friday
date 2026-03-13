"""
Shelly Plug device (Gen2+).

Covers any Shelly plug/outlet that exposes a Switch component with built-in
power metering — e.g. Shelly Plug US Gen4, Shelly Plus Plug US/UK/IT, etc.

Power metrics come from Switch.GetStatus; the Switch component on metered
plugs includes: apower (W), voltage (V), current (A), freq (Hz), aenergy
(Wh cumulative), and temperature.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import ShellyDevice


@dataclass
class SwitchStatus:
    """Parsed result of Switch.GetStatus for a single switch channel."""

    output: bool
    """True = output is ON."""

    apower: Optional[float] = None
    """Instantaneous active power in Watts (if metered)."""

    voltage: Optional[float] = None
    """RMS voltage in Volts."""

    current: Optional[float] = None
    """RMS current in Amperes."""

    freq: Optional[float] = None
    """Mains frequency in Hz."""

    energy_wh: Optional[float] = None
    """Total energy consumed since last counter reset (Watt-hours)."""

    temperature_c: Optional[float] = None
    """Device internal temperature in °C (if available)."""

    raw: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}

    @classmethod
    def from_rpc(cls, data: Dict[str, Any]) -> "SwitchStatus":
        aenergy = data.get("aenergy") or {}
        temp = data.get("temperature") or {}
        return cls(
            output=bool(data.get("output", False)),
            apower=data.get("apower"),
            voltage=data.get("voltage"),
            current=data.get("current"),
            freq=data.get("freq"),
            energy_wh=aenergy.get("total"),
            temperature_c=temp.get("tC"),
            raw=data,
        )


class ShellyPlug(ShellyDevice):
    """
    Controls a Shelly metered plug (Gen2+).

    Most Shelly plugs expose a single switch channel at id=0.
    ``switch_id`` can be set to 1 for dual-outlet models in the future.
    """

    def __init__(
        self,
        host: str,
        name: str = "",
        switch_id: int = 0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0,
    ):
        super().__init__(host, name=name, username=username, password=password, timeout=timeout)
        self.switch_id = switch_id

    @property
    def device_type(self) -> str:
        return "plug"

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def turn_on(self) -> bool:
        result = await self.client.call("Switch.Set", {"id": self.switch_id, "on": True})
        return True

    async def turn_off(self) -> bool:
        result = await self.client.call("Switch.Set", {"id": self.switch_id, "on": False})
        return True

    async def toggle(self) -> bool:
        """Toggle the output; returns the new output state (True = on)."""
        result = await self.client.call("Switch.Toggle", {"id": self.switch_id})
        # was_on tells us the *previous* state, so new state is the opposite
        return not bool((result or {}).get("was_on", False))

    # ------------------------------------------------------------------
    # Status & statistics
    # ------------------------------------------------------------------

    async def get_switch_status(self) -> SwitchStatus:
        """Fetch full Switch.GetStatus for this channel."""
        data = await self.client.call("Switch.GetStatus", {"id": self.switch_id})
        return SwitchStatus.from_rpc(data or {})

    async def get_stats(self) -> Dict[str, Any]:
        """Return power statistics as a plain dict."""
        status = await self.get_switch_status()
        stats: Dict[str, Any] = {"output": status.output}
        if status.apower is not None:
            stats["power_w"] = round(status.apower, 2)
        if status.voltage is not None:
            stats["voltage_v"] = round(status.voltage, 1)
        if status.current is not None:
            stats["current_a"] = round(status.current, 3)
        if status.freq is not None:
            stats["freq_hz"] = round(status.freq, 1)
        if status.energy_wh is not None:
            stats["energy_wh"] = round(status.energy_wh, 3)
        if status.temperature_c is not None:
            stats["temp_c"] = round(status.temperature_c, 1)
        return stats

    def format_stats(self, stats: Dict[str, Any]) -> str:
        """Return a natural-language summary of the stats dict."""
        label = self.name or self.host
        state = "on" if stats.get("output") else "off"
        parts = [f"The {label} is currently {state}."]

        if "power_w" in stats:
            parts.append(f"It is drawing {stats['power_w']} watts")
            if "voltage_v" in stats:
                parts[-1] += f" at {stats['voltage_v']} volts"
            if "current_a" in stats:
                parts[-1] += f" and {stats['current_a']} amps"
            parts[-1] += "."

        if "energy_wh" in stats:
            kwh = round(stats["energy_wh"] / 1000, 3)
            parts.append(f"Total energy consumed since last reset: {kwh} kWh.")

        if "temp_c" in stats:
            parts.append(f"Internal temperature: {stats['temp_c']}°C.")

        return " ".join(parts)

    async def reset_energy_counters(self) -> Dict[str, Any]:
        """Reset the cumulative energy counter. Returns the previous counter values."""
        result = await self.client.call("Switch.ResetCounters", {"id": self.switch_id})
        return result or {}
