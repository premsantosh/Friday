"""Friday's self-awareness substrate: read-only introspection over its own state.

Everything Friday knows about itself flows through this package: the
`self_status` workflow, the `python -m research doctor` CLI, and the nightly
failure alert all call the same providers. See README.md in this directory for
the extension contract — the short version is that coverage is discovered, not
enumerated, so new abilities are reported automatically.
"""

from introspection.registry import (
    CheckResult,
    CheckStatus,
    Paths,
    Probes,
    StatusProvider,
    format_report,
    gather_status,
    iter_providers,
    register_provider,
    run_doctor,
    summarize,
)

__all__ = [
    "CheckResult",
    "CheckStatus",
    "Paths",
    "Probes",
    "StatusProvider",
    "format_report",
    "gather_status",
    "iter_providers",
    "register_provider",
    "run_doctor",
    "summarize",
]
