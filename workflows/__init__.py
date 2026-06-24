from .base import (
    Workflow,
    WorkflowStatus,
    WorkflowResult,
    WorkflowTrigger,
    WorkflowManager,
    ConversationalWorkflow,
    InterruptPolicy,
    DoorbellWorkflow,
    ThermostatWorkflow,
    WeatherWorkflow,
    TimerWorkflow,
    create_default_workflow_manager,
)

from .home_assistant import (
    HomeAssistantConfig,
    HomeAssistantClient,
    HomeAssistantLockWorkflow,
    HomeAssistantClimateWorkflow,
)

from .philips_hue import (
    PhilipsHueConfig,
    PhilipsHueClient,
    PhilipsHueLightsWorkflow,
)

from .time import TimeWorkflow

from .shelly import (
    ShellyClient,
    ShellyRPCError,
    ShellyDevice,
    DeviceInfo,
    ShellyPlug,
    SwitchStatus,
    CoffeeMachine,
    BrewState,
    ShellyWorkflow,
    DEVICE_REGISTRY,
)

from .coffee_machine import (
    CoffeeSchedule,
    CoffeeMonitorConfig,
    CoffeeMachineMonitor,
    CoffeeMachineWorkflow,
)

__all__ = [
    # Base classes
    "Workflow",
    "WorkflowStatus",
    "WorkflowResult",
    "WorkflowTrigger",
    "WorkflowManager",
    "ConversationalWorkflow",
    "InterruptPolicy",

    # Example workflows
    "DoorbellWorkflow",
    "ThermostatWorkflow",
    "WeatherWorkflow",
    "TimerWorkflow",
    "create_default_workflow_manager",

    # Home Assistant
    "HomeAssistantConfig",
    "HomeAssistantClient",
    "HomeAssistantLockWorkflow",
    "HomeAssistantClimateWorkflow",

    # Philips Hue
    "PhilipsHueConfig",
    "PhilipsHueClient",
    "PhilipsHueLightsWorkflow",

    # Time
    "TimeWorkflow",

    # Shelly
    "ShellyClient",
    "ShellyRPCError",
    "ShellyDevice",
    "DeviceInfo",
    "ShellyPlug",
    "SwitchStatus",
    "CoffeeMachine",
    "BrewState",
    "ShellyWorkflow",
    "DEVICE_REGISTRY",

    # Coffee Machine
    "CoffeeSchedule",
    "CoffeeMonitorConfig",
    "CoffeeMachineMonitor",
    "CoffeeMachineWorkflow",
]
