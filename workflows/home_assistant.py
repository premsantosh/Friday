"""
Home Assistant Integration

This module provides integration with Home Assistant for controlling
smart home devices. This is a more complete example of how to build
a real workflow integration.

Prerequisites:
- Home Assistant running and accessible
- Long-lived access token from Home Assistant
- pip install aiohttp

Configuration:
- Set HASS_URL environment variable (e.g., http://homeassistant.local:8123)
- Set HASS_TOKEN environment variable (your long-lived access token)
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .base import Workflow, WorkflowResult, WorkflowStatus, WorkflowTrigger


@dataclass
class HomeAssistantConfig:
    """Configuration for Home Assistant connection."""
    url: str = ""  # e.g., http://homeassistant.local:8123
    token: str = ""  # Long-lived access token
    
    @classmethod
    def from_env(cls) -> "HomeAssistantConfig":
        return cls(
            url=os.getenv("HASS_URL", "http://localhost:8123"),
            token=os.getenv("HASS_TOKEN", ""),
        )


class HomeAssistantClient:
    """
    Client for interacting with Home Assistant REST API.
    """
    
    def __init__(self, config: HomeAssistantConfig):
        self.config = config
        self.base_url = config.url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json",
        }
    
    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call a Home Assistant service.
        
        Args:
            domain: Service domain (e.g., "light", "switch", "lock")
            service: Service name (e.g., "turn_on", "turn_off", "lock")
            entity_id: Target entity (e.g., "light.living_room")
            data: Additional service data
        """
        import aiohttp
        
        url = f"{self.base_url}/api/services/{domain}/{service}"
        
        payload = data or {}
        if entity_id:
            payload["entity_id"] = entity_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as resp:
                resp.raise_for_status()
                return await resp.json()
    
    async def get_states(self) -> List[Dict[str, Any]]:
        """Get all entity states."""
        import aiohttp
        
        url = f"{self.base_url}/api/states"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                resp.raise_for_status()
                return await resp.json()
    
    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        """Get state of a specific entity."""
        import aiohttp
        
        url = f"{self.base_url}/api/states/{entity_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                resp.raise_for_status()
                return await resp.json()


class HomeAssistantLightsWorkflow(Workflow):
    """
    Control lights through Home Assistant.
    
    This is a production-ready example showing how to integrate
    with a real smart home system.
    """
    
    def __init__(self, client: Optional[HomeAssistantClient] = None):
        self.client = client or self._create_default_client()
        
        # Map room names to entity IDs
        # Customize this for your setup
        self.room_mapping = {
            "living room": "light.living_room",
            "bedroom": "light.bedroom",
            "kitchen": "light.kitchen",
            "bathroom": "light.bathroom",
            "office": "light.office",
            "all": "all",  # Special case
        }
    
    def _create_default_client(self) -> Optional[HomeAssistantClient]:
        """Create client from environment variables."""
        config = HomeAssistantConfig.from_env()
        if config.token:
            return HomeAssistantClient(config)
        return None
    
    @property
    def name(self) -> str:
        return "hass_lights"
    
    @property
    def description(self) -> str:
        return "Control lights through Home Assistant"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["light", "lights", "lamp", "illuminate", "bright", "dim"],
            patterns=[
                r"turn (on|off) .*(light|lamp)",
                r"(light|lamp).* (on|off)",
                r"dim .*(light|lamp)",
                r"set .*(light|lamp).* to",
            ],
            examples=[
                "Turn on the living room lights",
                "Dim the bedroom lights to 50%",
                "Turn off all the lights",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        if self.client is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="Home Assistant is not configured. Please set HASS_URL and HASS_TOKEN environment variables.",
                error="No Home Assistant client"
            )
        
        action = entities.get("action", "toggle")
        room = entities.get("room", "living room").lower()
        brightness = entities.get("brightness")
        
        # Get entity ID
        entity_id = self.room_mapping.get(room)
        if entity_id is None:
            # Try to find a matching entity
            entity_id = f"light.{room.replace(' ', '_')}"
        
        try:
            if action == "on":
                service_data = {}
                if brightness:
                    service_data["brightness_pct"] = brightness
                
                if entity_id == "all":
                    await self.client.call_service("light", "turn_on", data={"entity_id": "all"})
                else:
                    await self.client.call_service("light", "turn_on", entity_id, service_data)
                
                message = f"I've illuminated the {room}, sir."
                if brightness:
                    message = f"I've set the {room} lights to {brightness}%, sir."
            
            elif action == "off":
                if entity_id == "all":
                    await self.client.call_service("light", "turn_off", data={"entity_id": "all"})
                else:
                    await self.client.call_service("light", "turn_off", entity_id)
                
                message = f"The {room} is now dark, sir. Do try not to stub your toe."
            
            elif action == "dim" and brightness is not None:
                await self.client.call_service(
                    "light", "turn_on", entity_id,
                    {"brightness_pct": brightness}
                )
                message = f"I've dimmed the {room} lights to {brightness}%, sir."
            
            else:
                # Toggle
                await self.client.call_service("light", "toggle", entity_id)
                message = f"I've toggled the {room} lights, sir."
            
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=message,
                data={"room": room, "action": action, "brightness": brightness}
            )
        
        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=f"I encountered difficulty with the lights, sir. The error was: {str(e)}",
                error=str(e)
            )


class HomeAssistantLockWorkflow(Workflow):
    """Control door locks through Home Assistant."""
    
    def __init__(self, client: Optional[HomeAssistantClient] = None):
        self.client = client or self._create_default_client()
        
        # Map door names to entity IDs
        self.lock_mapping = {
            "front": "lock.front_door",
            "back": "lock.back_door",
            "garage": "lock.garage_door",
            "side": "lock.side_door",
        }
    
    def _create_default_client(self) -> Optional[HomeAssistantClient]:
        config = HomeAssistantConfig.from_env()
        if config.token:
            return HomeAssistantClient(config)
        return None
    
    @property
    def name(self) -> str:
        return "hass_locks"
    
    @property
    def description(self) -> str:
        return "Control door locks through Home Assistant"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["lock", "unlock", "door", "secure"],
            patterns=[
                r"(lock|unlock).*(door|entrance)",
                r"(secure|unsecure)",
                r"is .* (locked|unlocked)",
            ],
            examples=[
                "Lock the front door",
                "Unlock the back door",
                "Is the garage locked?",
                "Secure all doors",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        if self.client is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="Home Assistant is not configured for lock control, sir.",
                error="No Home Assistant client"
            )
        
        action = entities.get("action", "lock")
        door = entities.get("door", "front").lower()
        
        entity_id = self.lock_mapping.get(door, f"lock.{door}_door")
        
        try:
            if action == "lock":
                await self.client.call_service("lock", "lock", entity_id)
                message = f"The {door} door is now secured, sir."
            
            elif action == "unlock":
                await self.client.call_service("lock", "unlock", entity_id)
                message = f"I've unlocked the {door} door, sir. Do exercise appropriate caution."
            
            elif action == "check":
                state = await self.client.get_state(entity_id)
                is_locked = state.get("state") == "locked"
                if is_locked:
                    message = f"The {door} door is securely locked, sir."
                else:
                    message = f"The {door} door is currently unlocked, sir. Shall I secure it?"
            
            else:
                message = f"Lock action completed, sir."
            
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=message,
                data={"door": door, "action": action}
            )
        
        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=f"There was a complication with the door lock, sir.",
                error=str(e)
            )


class HomeAssistantClimateWorkflow(Workflow):
    """Control thermostat through Home Assistant."""
    
    def __init__(self, client: Optional[HomeAssistantClient] = None, climate_entity: str = "climate.thermostat"):
        self.client = client or self._create_default_client()
        self.climate_entity = climate_entity
    
    def _create_default_client(self) -> Optional[HomeAssistantClient]:
        config = HomeAssistantConfig.from_env()
        if config.token:
            return HomeAssistantClient(config)
        return None
    
    @property
    def name(self) -> str:
        return "hass_climate"
    
    @property
    def description(self) -> str:
        return "Control thermostat through Home Assistant"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["temperature", "thermostat", "heat", "cool", "AC", "warm", "cold"],
            patterns=[
                r"set.*(temp|temperature|thermostat)",
                r"(warm|heat|cool).*up",
                r"turn (on|off).*(heat|AC)",
                r"make it (warm|cool|cold)",
            ],
            examples=[
                "Set the temperature to 72",
                "Turn on the AC",
                "Make it warmer in here",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        if self.client is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="Home Assistant climate control is not configured, sir.",
                error="No Home Assistant client"
            )
        
        temperature = entities.get("temperature")
        mode = entities.get("mode")  # heat, cool, auto, off
        
        try:
            if temperature:
                await self.client.call_service(
                    "climate", "set_temperature",
                    self.climate_entity,
                    {"temperature": temperature}
                )
                message = f"I've set the temperature to {temperature} degrees, sir. Comfort is en route."
            
            elif mode:
                await self.client.call_service(
                    "climate", "set_hvac_mode",
                    self.climate_entity,
                    {"hvac_mode": mode}
                )
                message = f"The climate system is now in {mode} mode, sir."
            
            else:
                state = await self.client.get_state(self.climate_entity)
                current_temp = state.get("attributes", {}).get("current_temperature", "unknown")
                target_temp = state.get("attributes", {}).get("temperature", "unknown")
                message = f"The current temperature is {current_temp} degrees, with the target set to {target_temp}, sir."
            
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=message,
                data={"temperature": temperature, "mode": mode}
            )
        
        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="The climate system is being uncooperative, sir.",
                error=str(e)
            )
