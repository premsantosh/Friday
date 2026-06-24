"""Reservation channels (M2): the booking strategies behind a common interface."""

from .base import (
    Availability,
    AvailabilityStatus,
    BookingResult,
    BrowserChannel,
    CommitPlan,
    ReservationChannel,
    kill_switch_on,
)
from .email import EmailChannel, EmailOutcome, EmailReply
from .generic_web import GenericWebChannel
from .opentable import OpenTableChannel
from .phone import (
    BlandClient,
    CallResult,
    DialAndBridgeChannel,
    PhoneChannel,
    PhoneOutcome,
)
from .resy import ResyChannel
from .sandbox_bot import BotCandidate, DockerSandbox, GitHubBotFinder, SandboxBotChannel, SandboxResult
from .yelp import YelpChannel

__all__ = [
    "ReservationChannel",
    "BrowserChannel",
    "Availability",
    "AvailabilityStatus",
    "CommitPlan",
    "BookingResult",
    "kill_switch_on",
    "GenericWebChannel",
    "OpenTableChannel",
    "ResyChannel",
    "YelpChannel",
    "PhoneChannel",
    "DialAndBridgeChannel",
    "BlandClient",
    "CallResult",
    "PhoneOutcome",
    "EmailChannel",
    "EmailOutcome",
    "EmailReply",
    "SandboxBotChannel",
    "GitHubBotFinder",
    "DockerSandbox",
    "BotCandidate",
    "SandboxResult",
]
