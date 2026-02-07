"""
Philips Hue Integration (API v2 / CLIP v2)

This module provides integration with Philips Hue Bridge for controlling
smart lights directly via the Hue CLIP v2 REST API.

Prerequisites:
- Philips Hue Bridge accessible on the network
- Application key (press the bridge button and create one via the API)
- pip install aiohttp

Configuration:
- Set HUE_BRIDGE_IP environment variable (e.g., 10.0.0.242)
- Set HUE_APPLICATION_KEY environment variable (your API application key)
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
    application_key: str = ""

    @classmethod
    def from_env(cls) -> "PhilipsHueConfig":
        return cls(
            bridge_ip=os.getenv("HUE_BRIDGE_IP", ""),
            application_key=os.getenv("HUE_APPLICATION_KEY", os.getenv("HUE_USERNAME", "")),
        )


class PhilipsHueClient:
    """
    Async REST client for the Philips Hue Bridge (API v2 / CLIP v2).

    Uses HTTPS with SSL verification disabled (Hue Bridge uses a self-signed cert).
    Authentication is via the `hue-application-key` header.
    """

    def __init__(self, config: PhilipsHueConfig):
        self.config = config
        self.base_url = f"https://{config.bridge_ip}/clip/v2/resource"
        self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        self._headers = {"hue-application-key": config.application_key}
        # Name-to-ID lookup maps populated by discover()
        self._light_names: Dict[str, str] = {}
        self._group_names: Dict[str, str] = {}  # room name -> grouped_light ID
        self._all_lights_group_id: Optional[str] = None  # bridge_home grouped_light
        self._discovered = False

    async def discover(self):
        """Fetch all lights, rooms, and grouped_lights, building name-to-ID lookup maps."""
        lights = await self.get_lights()
        rooms = await self.get_rooms()
        grouped_lights = await self.get_grouped_lights()

        # Build light name -> light resource ID
        self._light_names = {}
        for light in lights:
            name = light.get("metadata", {}).get("name", "")
            if name:
                self._light_names[name.lower()] = light["id"]

        # Build a service rid -> grouped_light ID lookup
        gl_by_id = {gl["id"]: gl for gl in grouped_lights}

        # Build room name -> grouped_light ID
        self._group_names = {}
        for room in rooms:
            room_name = room.get("metadata", {}).get("name", "")
            if not room_name:
                continue
            # Each room has services; find the grouped_light service
            for svc in room.get("services", []):
                if svc.get("rtype") == "grouped_light" and svc.get("rid") in gl_by_id:
                    self._group_names[room_name.lower()] = svc["rid"]
                    break

        # Find the bridge_home grouped_light for "all lights" control
        bridge_homes = await self._get_resource("bridge_home")
        for bh in bridge_homes:
            for svc in bh.get("services", []):
                if svc.get("rtype") == "grouped_light":
                    self._all_lights_group_id = svc["rid"]
                    break
            if self._all_lights_group_id:
                break

        self._discovered = True
        logger.info(
            "Hue discovery complete: %d lights, %d rooms",
            len(self._light_names),
            len(self._group_names),
        )

    async def _get_resource(self, resource_type: str) -> List[Dict[str, Any]]:
        """GET /clip/v2/resource/{type} — returns the data array."""
        import aiohttp

        url = f"{self.base_url}/{resource_type}"
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(url, ssl=self._ssl_context) as resp:
                result = await resp.json(content_type=None)
                self._check_hue_errors(result)
                return result.get("data", [])

    async def get_lights(self) -> List[Dict[str, Any]]:
        """GET /clip/v2/resource/light — returns all light resources."""
        return await self._get_resource("light")

    async def get_rooms(self) -> List[Dict[str, Any]]:
        """GET /clip/v2/resource/room — returns all room resources."""
        return await self._get_resource("room")

    async def get_grouped_lights(self) -> List[Dict[str, Any]]:
        """GET /clip/v2/resource/grouped_light — returns all grouped_light resources."""
        return await self._get_resource("grouped_light")

    async def set_light_state(self, light_id: str, state: Dict[str, Any]) -> List[Dict]:
        """PUT /clip/v2/resource/light/{id} — set state for a single light."""
        import aiohttp

        url = f"{self.base_url}/light/{light_id}"
        logger.debug("PUT %s body=%s", url, state)
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.put(url, json=state, ssl=self._ssl_context) as resp:
                data = await resp.json(content_type=None)
                logger.debug("Hue response: %s", data)
                self._check_hue_errors(data)
                return data.get("data", [])

    async def set_group_action(self, grouped_light_id: str, action: Dict[str, Any]) -> List[Dict]:
        """PUT /clip/v2/resource/grouped_light/{id} — set action for a grouped light."""
        import aiohttp

        url = f"{self.base_url}/grouped_light/{grouped_light_id}"
        logger.debug("PUT %s body=%s", url, action)
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.put(url, json=action, ssl=self._ssl_context) as resp:
                data = await resp.json(content_type=None)
                logger.debug("Hue response: %s", data)
                self._check_hue_errors(data)
                return data.get("data", [])

    def find_light_id(self, name: str) -> Optional[str]:
        """Case-insensitive light name lookup."""
        return self._light_names.get(name.lower())

    def find_group_id(self, name: str) -> Optional[str]:
        """Case-insensitive room name lookup, returns the grouped_light ID."""
        return self._group_names.get(name.lower())

    @staticmethod
    def _check_hue_errors(data: Any):
        """
        Hue API v2 returns errors in an `errors` array within the response.
        """
        if not isinstance(data, dict):
            return
        errors = data.get("errors", [])
        if errors:
            descriptions = "; ".join(
                e.get("description", "unknown error") for e in errors
            )
            raise RuntimeError(f"Hue API error: {descriptions}")


class PhilipsHueLightsWorkflow(Workflow):
    """
    Control lights through a Philips Hue Bridge (API v2).

    Prefers group (room) control and falls back to individual light lookup.
    Uses bridge_home grouped_light for "all lights" control.
    Supports mood-based lighting scenes.
    """

    # Mood-to-Hue-state mappings (API v2 format).
    # brightness: 0.0-100.0, mirek: color temp in mirek (153=cool daylight, 500=warm candlelight)
    # color: xy color space
    MOODS: Dict[str, Dict[str, Any]] = {
        "romantic": {
            "on": {"on": True},
            "dimming": {"brightness": 30.3},
            "color_temperature": {"mirek": 500},
        },
        "relax": {
            "on": {"on": True},
            "dimming": {"brightness": 50.0},
            "color_temperature": {"mirek": 447},
        },
        "energize": {
            "on": {"on": True},
            "dimming": {"brightness": 100.0},
            "color_temperature": {"mirek": 250},
        },
        "party": {
            "on": {"on": True},
            "dimming": {"brightness": 100.0},
            "color": {"xy": {"x": 0.1532, "y": 0.0475}},
        },
        "bedtime": {
            "on": {"on": True},
            "dimming": {"brightness": 9.8},
            "color_temperature": {"mirek": 500},
        },
        "focus": {
            "on": {"on": True},
            "dimming": {"brightness": 100.0},
            "color_temperature": {"mirek": 300},
        },
        "movie": {
            "on": {"on": True},
            "dimming": {"brightness": 15.0},
            "color_temperature": {"mirek": 400},
        },
        "morning": {
            "on": {"on": True},
            "dimming": {"brightness": 78.7},
            "color_temperature": {"mirek": 350},
        },
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
        if config.bridge_ip and config.application_key:
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
                message="Philips Hue is not configured. Please set HUE_BRIDGE_IP and HUE_APPLICATION_KEY environment variables.",
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
            # Build the state/action payload (API v2 nested format)
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
                state: Dict[str, Any] = {"on": {"on": True}}
                if brightness is not None:
                    state["dimming"] = {"brightness": self._pct_to_bri(brightness)}
            elif action == "off":
                state = {"on": {"on": False}}
            elif action == "dim" and brightness is not None:
                state = {"on": {"on": True}, "dimming": {"brightness": self._pct_to_bri(brightness)}}
            else:
                # Toggle: send on=True as a reasonable default
                state = {"on": {"on": True}}

            # Determine target: prefer group (room), fallback to individual light
            if room in ("all", "everywhere", "everything", "every room"):
                if self.client._all_lights_group_id:
                    await self.client.set_group_action(self.client._all_lights_group_id, state)
                else:
                    return WorkflowResult(
                        status=WorkflowStatus.FAILURE,
                        message="I could not find the bridge home group for all lights, sir.",
                        error="No bridge_home grouped_light found",
                    )
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
    def _pct_to_bri(pct: int) -> float:
        """Convert brightness 0-100% to Hue API v2's 0.0-100.0 scale."""
        return max(0.0, min(100.0, float(pct)))

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
