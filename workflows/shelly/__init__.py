from .client import ShellyClient, ShellyRPCError
from .devices import ShellyDevice, DeviceInfo, ShellyPlug, SwitchStatus, CoffeeMachine, BrewState
from .workflow import ShellyWorkflow, DEVICE_REGISTRY

# Register coffee_machine type so it can be used via SHELLY_DEVICES JSON as well
DEVICE_REGISTRY["coffee_machine"] = CoffeeMachine

__all__ = [
    # Low-level client
    "ShellyClient",
    "ShellyRPCError",
    # Device base
    "ShellyDevice",
    "DeviceInfo",
    # Device implementations
    "ShellyPlug",
    "SwitchStatus",
    "CoffeeMachine",
    "BrewState",
    # Workflow
    "ShellyWorkflow",
    "DEVICE_REGISTRY",
]
