"""
Shelly Smart Plug Workflow

Allows the voice assistant to control Shelly Gen2+ plugs by name and query
their power consumption stats.

Configuration — set the SHELLY_DEVICES environment variable to a JSON list:

    export SHELLY_DEVICES='[
        {"name": "desk lamp",  "ip": "192.168.1.50", "type": "plug"},
        {"name": "3d printer", "ip": "192.168.1.51", "type": "plug"}
    ]'

Optional per-device auth:
    {"name": "...", "ip": "...", "type": "plug", "username": "admin", "password": "secret"}

Adding a new device type:
  1. Create workflows/shelly/devices/<type>.py with a class that extends ShellyDevice.
  2. Register the type slug → class in DEVICE_REGISTRY below.
  3. That's it — the workflow will instantiate it automatically.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from ..base import Workflow, WorkflowResult, WorkflowStatus, WorkflowTrigger
from .devices.base import ShellyDevice
from .devices.plug import ShellyPlug

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: add new device types here
# ---------------------------------------------------------------------------

DEVICE_REGISTRY: Dict[str, type] = {
    "plug": ShellyPlug,
    "outlet": ShellyPlug,  # alias
}


def _load_devices_from_env() -> List[ShellyDevice]:
    """Parse SHELLY_DEVICES env var and instantiate device objects."""
    raw = os.getenv("SHELLY_DEVICES", "")
    if not raw:
        return []

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("SHELLY_DEVICES is not valid JSON: %s", exc)
        return []

    devices: List[ShellyDevice] = []
    for entry in entries:
        device_type = entry.get("type", "plug").lower()
        cls = DEVICE_REGISTRY.get(device_type)
        if cls is None:
            logger.warning("Unknown Shelly device type '%s', skipping.", device_type)
            continue

        try:
            device = cls(
                host=entry["ip"],
                name=entry.get("name", entry["ip"]),
                username=entry.get("username"),
                password=entry.get("password"),
            )
            devices.append(device)
            logger.info("Registered Shelly device '%s' (%s) at %s", device.name, device_type, entry["ip"])
        except Exception as exc:
            logger.error("Failed to create Shelly device from %s: %s", entry, exc)

    return devices


class ShellyWorkflow(Workflow):
    """
    Voice workflow for controlling Shelly Gen2+ smart plugs.

    Supports:
      - Turning devices on/off by name
      - Querying power consumption / statistics
    """

    def __init__(self, devices: Optional[List[ShellyDevice]] = None):
        self._devices: List[ShellyDevice] = devices if devices is not None else _load_devices_from_env()
        # Build a lowercase name → device lookup map
        self._by_name: Dict[str, ShellyDevice] = {d.name.lower(): d for d in self._devices}

    # ------------------------------------------------------------------
    # Workflow interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "shelly"

    @property
    def description(self) -> str:
        device_list = ", ".join(d.name for d in self._devices) or "none configured"
        return f"Control Shelly smart plugs and query their power usage. Devices: {device_list}."

    @property
    def trigger(self) -> WorkflowTrigger:
        # Build device-name keywords dynamically so the trigger stays accurate
        # even as devices are added at runtime.
        device_keywords = [word for d in self._devices for word in d.name.lower().split()]

        return WorkflowTrigger(
            keywords=[
                "plug", "outlet", "socket", "power", "watt", "energy",
                "consumption", "electricity", "shelly",
                *device_keywords,
            ],
            patterns=[
                r"turn (on|off) (?:the )?(.+?)(?:\s+plug|\s+outlet|\s+socket)?$",
                r"switch (on|off) (?:the )?(.+)",
                r"(power|energy|consumption|stats|statistics|usage) (?:for|of) (?:the )?(.+)",
                r"how (?:much power|many watts) (?:is|does) (?:the )?(.+?) (?:using|drawing|consuming)",
                r"(?:is|are) (?:the )?(.+?) (?:on|off)\??$",
            ],
            examples=[
                "Turn on the desk lamp",
                "Turn off the 3D printer",
                "How much power is the desk lamp using?",
                "Power stats for the 3D printer",
                "Is the desk lamp on?",
            ],
        )

    async def execute(self, intent: str, entities: Dict) -> WorkflowResult:
        if not self._devices:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="No Shelly devices have been configured, sir. Please set the SHELLY_DEVICES environment variable.",
                error="No devices configured",
            )

        action = entities.get("action", "status")
        device_name = entities.get("device", "").lower().strip()

        device = self._find_device(device_name, intent)
        if device is None:
            known = ", ".join(d.name for d in self._devices)
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=f"I could not identify which device you mean, sir. The configured devices are: {known}.",
                error=f"Device not found: {device_name!r}",
            )

        try:
            if action == "on":
                await device.turn_on()
                return WorkflowResult(
                    status=WorkflowStatus.SUCCESS,
                    message=f"The {device.name} is now on, sir.",
                    data={"device": device.name, "action": "on"},
                )

            elif action == "off":
                await device.turn_off()
                return WorkflowResult(
                    status=WorkflowStatus.SUCCESS,
                    message=f"The {device.name} has been switched off, sir.",
                    data={"device": device.name, "action": "off"},
                )

            elif action in ("toggle",):
                new_state = await device.toggle()  # type: ignore[attr-defined]
                state_word = "on" if new_state else "off"
                return WorkflowResult(
                    status=WorkflowStatus.SUCCESS,
                    message=f"The {device.name} is now {state_word}, sir.",
                    data={"device": device.name, "action": "toggle", "state": state_word},
                )

            else:  # status / stats
                stats = await device.get_stats()
                message = device.format_stats(stats)
                return WorkflowResult(
                    status=WorkflowStatus.SUCCESS,
                    message=message,
                    data={"device": device.name, "stats": stats},
                )

        except Exception as exc:
            logger.exception("Shelly workflow error for device '%s'", device.name)
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=f"I was unable to reach the {device.name}, sir. Please check that it is on the network.",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_device(self, name_hint: str, full_intent: str = "") -> Optional[ShellyDevice]:
        """
        Fuzzy match a device by name.

        Priority:
          1. Exact match on normalised name
          2. Device whose name is fully contained in the hint
          3. Device whose name words are a subset of the intent words
        """
        # No hint: only unambiguous when a single device is configured —
        # an empty hint used to substring-match the first device in the list.
        if not name_hint:
            if len(self._devices) == 1:
                return self._devices[0]
        else:
            # 1. Exact
            if name_hint in self._by_name:
                return self._by_name[name_hint]

            # 2. Contained substring
            for dev_name, device in self._by_name.items():
                if dev_name in name_hint or name_hint in dev_name:
                    return device

        # 3. Word overlap against the full intent
        intent_words = set(re.findall(r"\w+", full_intent.lower()))
        best_device: Optional[ShellyDevice] = None
        best_overlap = 0
        for dev_name, device in self._by_name.items():
            dev_words = set(re.findall(r"\w+", dev_name))
            overlap = len(dev_words & intent_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_device = device

        return best_device if best_overlap > 0 else None

    def register_device(self, device: ShellyDevice) -> None:
        """Programmatically add a device at runtime."""
        self._devices.append(device)
        self._by_name[device.name.lower()] = device
        logger.info("Registered Shelly device '%s' at %s", device.name, device.host)
