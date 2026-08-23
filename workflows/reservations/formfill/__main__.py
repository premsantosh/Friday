"""Dry-run CLI: `python -m workflows.reservations.formfill --url <page>`.

Lives in its own module so running it doesn't re-import a package that's
already in sys.modules (which is what `python -m ...formfill.agent` warns about).
"""

from .agent import _main

raise SystemExit(_main())
