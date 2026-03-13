"""
Base class for all Shelly Gen2+ devices.

Every device type subclasses ShellyDevice and adds component-specific methods.
The base class exposes generic Shelly RPC namespaces (Shelly.*, System.*).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..client import ShellyClient


@dataclass
class DeviceInfo:
    """Parsed result of Shelly.GetDeviceInfo."""

    name: str
    mac: str
    model: str
    app: str
    fw_version: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rpc(cls, data: Dict[str, Any]) -> "DeviceInfo":
        return cls(
            name=data.get("name") or data.get("id", ""),
            mac=data.get("mac", ""),
            model=data.get("model", ""),
            app=data.get("app", ""),
            fw_version=data.get("ver", ""),
            raw=data,
        )


class ShellyDevice(ABC):
    """
    Abstract base class for a Shelly Gen2+ device.

    Provides generic RPC helpers and common Shelly.* methods.
    Subclasses add component-specific capabilities (Switch, PM1, etc.).
    """

    def __init__(
        self,
        host: str,
        name: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.host = host
        self.name = name  # Human-readable label (e.g. "desk lamp")
        self.client = ShellyClient(host, username=username, password=password, timeout=timeout)

    # ------------------------------------------------------------------
    # Abstract interface every device must expose
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def device_type(self) -> str:
        """Short slug identifying the device category, e.g. 'plug'."""

    @abstractmethod
    async def turn_on(self) -> bool:
        """Turn the primary output on. Returns True on success."""

    @abstractmethod
    async def turn_off(self) -> bool:
        """Turn the primary output off. Returns True on success."""

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Return a dict of relevant device statistics."""

    @abstractmethod
    def format_stats(self, stats: Dict[str, Any]) -> str:
        """Format stats dict into a human-readable string for the assistant to speak."""

    # ------------------------------------------------------------------
    # Generic Shelly.* RPC helpers available on all Gen2 devices
    # ------------------------------------------------------------------

    async def get_device_info(self) -> DeviceInfo:
        """Fetch identity information (model, firmware, MAC, …)."""
        data = await self.client.call("Shelly.GetDeviceInfo")
        return DeviceInfo.from_rpc(data)

    async def get_status(self) -> Dict[str, Any]:
        """Return the full Shelly.GetStatus response (all components)."""
        return await self.client.call("Shelly.GetStatus")

    async def reboot(self) -> None:
        """Reboot the device."""
        await self.client.call("Shelly.Reboot")
