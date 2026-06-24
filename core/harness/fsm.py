"""
Declarative state machine for conversational workflows (harness spec §6).

A workflow declares its dialogue states, the legal (state, event) transitions,
and which states are *gate states* — the only states from which a user's "yes"
may be recorded as an approval. The machine is a closed set: an undeclared
transition raises `IllegalTransition` instead of silently doing something, and
`ApprovalPolicy(gate_states=machine.gate_states)` makes approving from a
non-gate state structurally impossible.

The machine itself is pure data + validation; the current state lives in
`session.fsm_state` exactly as before (no framework schema change).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple


class IllegalTransition(Exception):
    def __init__(self, machine: str, state: str, event: str):
        self.machine, self.state, self.event = machine, state, event
        super().__init__(
            f"machine {machine!r}: no transition from state {state!r} on event {event!r}")


@dataclass(frozen=True)
class Machine:
    name: str
    states: frozenset
    initial: str
    transitions: Mapping[Tuple[str, str], str]
    gate_states: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        if self.initial not in self.states:
            raise ValueError(f"initial state {self.initial!r} not in states")
        if not self.gate_states <= self.states:
            raise ValueError(f"gate states {self.gate_states - self.states} not in states")
        for (src, event), dst in self.transitions.items():
            if src not in self.states or dst not in self.states:
                raise ValueError(f"transition ({src!r}, {event!r}) -> {dst!r} "
                                 f"references an undeclared state")

    def next(self, state: str, event: str) -> str:
        """The state after `event` fires in `state`. Undeclared -> IllegalTransition."""
        try:
            return self.transitions[(state, event)]
        except KeyError:
            raise IllegalTransition(self.name, state, event) from None

    def events_from(self, state: str) -> frozenset:
        return frozenset(e for (s, e) in self.transitions if s == state)
