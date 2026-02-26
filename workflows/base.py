"""
Extensible Workflow System

This module provides the foundation for adding agentic workflows to your assistant.
Each workflow is a self-contained capability that can:
- Be triggered by specific intents/commands
- Execute actions (smart home, APIs, etc.)
- Return results to the assistant

To add a new workflow:
1. Create a new class that inherits from Workflow
2. Implement the required methods
3. Register it in the WORKFLOW_REGISTRY

Example workflows are provided for common use cases.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re


class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Some actions succeeded
    PENDING = "pending"  # Waiting for external response


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    status: WorkflowStatus
    message: str  # Human-readable message for the assistant to speak
    data: Optional[Dict[str, Any]] = None  # Structured data for further processing
    error: Optional[str] = None  # Error details if failed


@dataclass
class WorkflowTrigger:
    """Defines how a workflow is triggered."""
    # Keywords that suggest this workflow
    keywords: List[str] = field(default_factory=list)
    
    # Regex patterns for more complex matching
    patterns: List[str] = field(default_factory=list)
    
    # Example phrases (used for LLM context)
    examples: List[str] = field(default_factory=list)


class Workflow(ABC):
    """
    Base class for all workflows.
    Implement this to add new capabilities to your assistant.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this workflow."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this workflow does."""
        pass
    
    @property
    @abstractmethod
    def trigger(self) -> WorkflowTrigger:
        """Define how this workflow is triggered."""
        pass
    
    @abstractmethod
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the workflow.
        
        Args:
            intent: The detected intent/command
            entities: Extracted entities (e.g., room name, device name)
            
        Returns:
            WorkflowResult with status and message
        """
        pass
    
    def matches(self, text: str) -> bool:
        """Check if this workflow should handle the given text."""
        text_lower = text.lower()
        
        # Check keywords
        for keyword in self.trigger.keywords:
            if keyword.lower() in text_lower:
                return True
        
        # Check patterns
        for pattern in self.trigger.patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def get_context_for_llm(self) -> str:
        """
        Return context about this workflow for the LLM.
        This helps the LLM understand what the assistant can do.
        """
        examples_text = "\n".join(f"  - {ex}" for ex in self.trigger.examples)
        return f"""
Workflow: {self.name}
Description: {self.description}
Example commands:
{examples_text}
"""


# =============================================================================
# EXAMPLE WORKFLOWS - Use these as templates for your own
# =============================================================================

class DoorbellWorkflow(Workflow):
    """Handle doorbell events and door lock control."""
    
    def __init__(self, doorbell_controller=None, lock_controller=None):
        self.doorbell = doorbell_controller
        self.lock = lock_controller
    
    @property
    def name(self) -> str:
        return "doorbell"
    
    @property
    def description(self) -> str:
        return "Check doorbell camera, see who's at the door, lock/unlock doors"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["door", "doorbell", "lock", "unlock", "visitor", "entrance"],
            patterns=[
                r"who.*(at|the) door",
                r"(lock|unlock).*(door|entrance)",
                r"check.*(door|entrance|visitor)",
                r"let .* in",
                r"open .* door",
            ],
            examples=[
                "Who's at the door?",
                "Lock the front door",
                "Unlock the back door",
                "Check the doorbell camera",
                "Let them in",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        action = entities.get("action", "check")  # check, lock, unlock
        door = entities.get("door", "front")
        
        if self.lock is None and self.doorbell is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="The door systems are not yet configured, sir. I suggest we remedy that.",
                error="No door controller configured"
            )
        
        try:
            if action == "check":
                # Would check doorbell camera
                message = "I'm checking the door camera now, sir."
            elif action == "lock":
                # Would lock the door
                message = f"The {door} door is now secured, sir."
            elif action == "unlock":
                # Would unlock the door
                message = f"I've unlocked the {door} door, sir. Do try not to let in anyone unsavory."
            else:
                message = f"Door action completed, sir."
            
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=message,
                data={"door": door, "action": action}
            )
        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message=f"There was a complication with the door, sir.",
                error=str(e)
            )


class ThermostatWorkflow(Workflow):
    """Control home thermostat and temperature."""
    
    def __init__(self, thermostat_controller=None):
        self.controller = thermostat_controller
    
    @property
    def name(self) -> str:
        return "thermostat"
    
    @property
    def description(self) -> str:
        return "Control thermostat - set temperature, change modes"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["temperature", "thermostat", "heat", "cool", "warm", "cold", "AC", "heating"],
            patterns=[
                r"set.*(temp|temperature|thermostat)",
                r"(warm|heat|cool).*up",
                r"turn (on|off).*(heat|AC|cooling|heating)",
                r"(too|it's).*(hot|cold|warm)",
                r"make it (warm|cool|cold)",
            ],
            examples=[
                "Set the temperature to 72 degrees",
                "It's too cold in here",
                "Turn on the AC",
                "Warm it up a bit",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        action = entities.get("action", "adjust")
        target_temp = entities.get("temperature")
        mode = entities.get("mode")  # heat, cool, auto
        
        if self.controller is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="The climate control system awaits configuration, sir.",
                error="No thermostat controller configured"
            )
        
        try:
            if target_temp:
                message = f"I've set the temperature to {target_temp} degrees, sir. Comfort should arrive shortly."
            elif mode:
                message = f"The climate system is now in {mode} mode, sir."
            else:
                message = "I've adjusted the temperature to something more agreeable, sir."
            
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=message,
                data={"temperature": target_temp, "mode": mode}
            )
        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="The climate control system is being uncooperative, sir.",
                error=str(e)
            )


class MediaWorkflow(Workflow):
    """Control TV, speakers, and media playback."""
    
    def __init__(self, media_controller=None):
        self.controller = media_controller
    
    @property
    def name(self) -> str:
        return "media"
    
    @property
    def description(self) -> str:
        return "Control TV, speakers, music playback"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["TV", "television", "music", "play", "pause", "volume", "speaker", "movie"],
            patterns=[
                r"turn (on|off).*(TV|television)",
                r"play .*(music|song|movie)",
                r"(pause|stop|resume)",
                r"(volume|louder|quieter)",
                r"(mute|unmute)",
                r"watch .*(netflix|youtube|movie)",
            ],
            examples=[
                "Turn on the TV",
                "Play some jazz music",
                "Pause the movie",
                "Turn up the volume",
                "Put on Netflix",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        action = entities.get("action", "toggle")
        device = entities.get("device", "TV")
        content = entities.get("content")
        
        if self.controller is None:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="The entertainment systems require configuration, sir.",
                error="No media controller configured"
            )
        
        try:
            if action == "on":
                message = f"Powering on the {device}, sir."
            elif action == "off":
                message = f"The {device} is now off, sir. Perhaps a book instead?"
            elif action == "play" and content:
                message = f"Now playing {content}, sir."
            elif action == "pause":
                message = "Paused, sir."
            elif action == "volume_up":
                message = "Increasing volume, sir."
            elif action == "volume_down":
                message = "Lowering volume, sir."
            else:
                message = "Media command executed, sir."
            
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message=message,
                data={"device": device, "action": action, "content": content}
            )
        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILURE,
                message="The entertainment system is being temperamental, sir.",
                error=str(e)
            )


class WeatherWorkflow(Workflow):
    """Get weather information."""
    
    def __init__(self, weather_api_key: Optional[str] = None):
        self.api_key = weather_api_key
    
    @property
    def name(self) -> str:
        return "weather"
    
    @property
    def description(self) -> str:
        return "Get current weather and forecasts"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["weather", "temperature", "rain", "sunny", "forecast", "outside"],
            patterns=[
                r"(what|how).*(weather|outside)",
                r"is it.*(rain|sunny|cold|hot)",
                r"(will|going to) .*(rain|snow)",
                r"forecast",
                r"should I .*(umbrella|jacket)",
            ],
            examples=[
                "What's the weather like?",
                "Is it going to rain today?",
                "What's the forecast for tomorrow?",
                "Should I bring an umbrella?",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        location = entities.get("location", "current")
        timeframe = entities.get("timeframe", "now")  # now, today, tomorrow, week
        
        # In a real implementation, you'd call a weather API here
        # For now, return a placeholder
        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message="I would check the weather, sir, but I haven't been connected to a weather service yet.",
            data={"location": location, "timeframe": timeframe}
        )


class TimerWorkflow(Workflow):
    """Set timers and reminders."""
    
    def __init__(self):
        self.active_timers: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return "timer"
    
    @property
    def description(self) -> str:
        return "Set timers, alarms, and reminders"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["timer", "alarm", "remind", "reminder", "wake", "minutes", "hours"],
            patterns=[
                r"set.*(timer|alarm|reminder)",
                r"remind me",
                r"in \d+ (minute|hour|second)",
                r"wake me",
                r"(cancel|stop).*(timer|alarm)",
            ],
            examples=[
                "Set a timer for 10 minutes",
                "Remind me in an hour",
                "Wake me up at 7am",
                "Cancel the timer",
            ]
        )
    
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        action = entities.get("action", "set")
        duration = entities.get("duration")
        label = entities.get("label", "timer")
        
        if action == "set":
            message = f"I've set a {duration} timer, sir. I shall alert you when it expires."
        elif action == "cancel":
            message = "Timer cancelled, sir."
        else:
            message = "Timer action completed, sir."
        
        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=message,
            data={"action": action, "duration": duration, "label": label}
        )


# =============================================================================
# WORKFLOW MANAGER
# =============================================================================

class WorkflowManager:
    """
    Manages all registered workflows.
    Use this to add, remove, and execute workflows.
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
    
    def register(self, workflow: Workflow):
        """Register a new workflow."""
        self.workflows[workflow.name] = workflow
    
    def unregister(self, name: str):
        """Unregister a workflow by name."""
        if name in self.workflows:
            del self.workflows[name]
    
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """Get a workflow by name."""
        return self.workflows.get(name)
    
    def find_matching_workflow(self, text: str) -> Optional[Workflow]:
        """Find a workflow that matches the given text.

        When multiple workflows match, prefer the one registered later since
        real integrations (Philips Hue, Home Assistant) are registered after
        the default template workflows and should take priority.
        """
        matched = None
        for workflow in self.workflows.values():
            if workflow.matches(text):
                matched = workflow
        return matched
    
    def get_all_context_for_llm(self) -> str:
        """Get context about all workflows for the LLM."""
        if not self.workflows:
            return "No special capabilities are currently configured."
        
        contexts = [w.get_context_for_llm() for w in self.workflows.values()]
        return "\n".join(contexts)
    
    def list_workflows(self) -> List[str]:
        """List all registered workflow names."""
        return list(self.workflows.keys())


def create_default_workflow_manager() -> WorkflowManager:
    """
    Create a workflow manager with example workflows.
    Modify this to add your actual smart home integrations.
    """
    manager = WorkflowManager()

    # Register example workflows (without actual controllers)
    # Replace None with your actual controller implementations
    manager.register(DoorbellWorkflow(doorbell_controller=None, lock_controller=None))
    manager.register(ThermostatWorkflow(thermostat_controller=None))
    manager.register(WeatherWorkflow())
    manager.register(TimerWorkflow())

    # NOTE: MediaWorkflow is not registered by default because its trigger
    # keywords ("movie", "play", "pause") overlap heavily with other workflows
    # (e.g. Philips Hue mood "movie night"). Register it explicitly when you
    # have a real media controller configured.

    return manager
