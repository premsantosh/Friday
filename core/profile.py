"""
Durable facts about the user — the things a booking or intake form asks for.

Deliberately a *separate table* from `memory.facts`: that one is auto-written by
the local fact extractor at 0.4 confidence (memory/extractor.py), and a
hallucinated guess must never end up typed into a clinic intake form. Every row
here is user-asserted (or seeded from an env var the user set themselves).

The split that matters:

  - `descriptors()` returns keys + labels + descriptions and **no values**. It
    is the only thing that may go to an LLM — the field mapper decides *which*
    fact fills a form field, never learning what that fact is.
  - `values()` returns the actual data, for local substitution only.

Storage lives in the same SQLite file as the rest of Friday's memory
(~/.friday/memory.db), chmod 0600 — this table holds date of birth and
insurance IDs, which the other tables don't.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "~/.friday/memory.db"
PROFILE_DB_ENV = "FRIDAY_PROFILE_DB"   # override; tests point this at a tmp path

# Personas. A fact set under a context overrides the same fact under 'default';
# anything not overridden falls through, so an email or date of birth is stored
# once and shared. The name you give a dentist is not the name you give a
# barber, and putting the wrong one on a medical form is a real problem.
DEFAULT_CONTEXT = "default"
FORMAL_CONTEXT = "formal"     # medical, legal, financial, government, insurance
CASUAL_CONTEXT = "casual"     # restaurants, salons, spas, everyday appointments
CONTEXTS = (DEFAULT_CONTEXT, FORMAL_CONTEXT, CASUAL_CONTEXT)


class UnknownProfileContext(ValueError):
    """Refused: contexts are a closed set, so a typo can't silently create one."""


def _clean_context(context: Optional[str]) -> str:
    value = (context or DEFAULT_CONTEXT).strip().lower()
    if value not in CONTEXTS:
        raise UnknownProfileContext(
            f"{context!r} is not one of {', '.join(CONTEXTS)}")
    return value


@dataclass(frozen=True)
class ProfileField:
    """A fact we know how to hold, and how to describe it to a model."""
    key: str
    label: str
    description: str
    # Extra label fragments for the deterministic (no-LLM) form matcher.
    aliases: tuple = ()
    derived: bool = False   # computed from another field when not set explicitly


# The closed vocabulary. A mapper may only name keys from this list; anything
# else is dropped, so a model can't invent a fact source.
PROFILE_FIELDS: tuple = (
    ProfileField("full_name", "Full name", "The user's full name",
                 aliases=("name", "your name", "full name", "contact name")),
    ProfileField("first_name", "First name", "Given name only",
                 aliases=("first", "given", "fname", "first name"), derived=True),
    ProfileField("last_name", "Last name", "Family name / surname only",
                 aliases=("last", "surname", "family", "lname", "last name"), derived=True),
    ProfileField("email", "Email address", "Email address for confirmations and replies",
                 aliases=("email", "e-mail", "email address")),
    ProfileField("phone", "Phone number", "Phone number for callbacks and SMS confirmations",
                 aliases=("phone", "mobile", "cell", "telephone", "tel", "contact number")),
    ProfileField("street_address", "Street address", "Street address, without city or state",
                 aliases=("address", "street", "address line 1", "addr")),
    ProfileField("city", "City", "City of residence", aliases=("city", "town")),
    ProfileField("state", "State", "State or province", aliases=("state", "province", "region")),
    ProfileField("postal_code", "ZIP / postal code", "ZIP or postal code",
                 aliases=("zip", "postal", "postcode", "zip code")),
    ProfileField("country", "Country", "Country of residence", aliases=("country",)),
    ProfileField("date_of_birth", "Date of birth",
                 "Date of birth, as asked for on medical and intake forms",
                 aliases=("date of birth", "birth date", "dob", "birthday")),
    ProfileField("insurance_provider", "Insurance provider",
                 "Health insurance carrier name",
                 aliases=("insurance", "carrier", "insurance provider", "health plan")),
    ProfileField("insurance_member_id", "Insurance member ID",
                 "Health insurance member or policy number",
                 aliases=("member id", "policy number", "subscriber id", "insurance id")),
    ProfileField("employer", "Employer", "Current employer or company name",
                 aliases=("employer", "company", "organization", "organisation")),
    ProfileField("referral_source", "Referral source",
                 "How the user typically hears about a business, for "
                 "'how did you hear about us' fields",
                 aliases=("how did you hear", "referral", "referred by", "hear about us")),
    ProfileField("preferred_contact_method", "Preferred contact method",
                 "How the user prefers to be contacted: 'email' or 'phone'",
                 aliases=("preferred contact", "contact preference", "best way to reach")),
)

FIELD_BY_KEY: Dict[str, ProfileField] = {f.key: f for f in PROFILE_FIELDS}
PROFILE_KEYS: frozenset = frozenset(FIELD_BY_KEY)

# Env vars the user already sets for the reservations agent. Seeding closes the
# gap where RESERVATION_USER_EMAIL was documented in .env.example but read
# nowhere in the code.
_ENV_SEEDS = {
    "RESERVATION_GUEST_NAME": "full_name",
    "RESERVATION_USER_PHONE": "phone",
    "RESERVATION_USER_EMAIL": "email",
}


class UnknownProfileKey(KeyError):
    """Refused: the key isn't in the declared registry."""


class UserProfile:
    """Durable, user-asserted facts. Safe to construct per use; cheap."""

    def __init__(self, db_path: Optional[str] = None):
        raw = db_path or os.getenv(PROFILE_DB_ENV) or DEFAULT_DB_PATH
        self.db_path = Path(raw).expanduser()
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_table()
        self._protect()

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS profile (
            key TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'default',
            value TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'user',
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (key, context)
        )
    """

    def _init_table(self) -> None:
        self.conn.execute(self._SCHEMA)
        self._migrate_to_contexts()
        self.conn.commit()

    def _migrate_to_contexts(self) -> None:
        """Add the context dimension to a v1 table (PK on key alone).

        The first version of this table held one value per key. Personas need
        (key, context), which means a new primary key — SQLite can't alter one
        in place, so rebuild and carry the rows over as 'default'.
        """
        columns = {row[1] for row in
                   self.conn.execute("PRAGMA table_info(profile)").fetchall()}
        if "context" in columns:
            return
        logger.info("Migrating profile table to per-context values")
        self.conn.executescript(f"""
            ALTER TABLE profile RENAME TO profile_v1;
            {self._SCHEMA};
            INSERT INTO profile (key, context, value, source, updated_at)
                SELECT key, 'default', value, source, updated_at FROM profile_v1;
            DROP TABLE profile_v1;
        """)

    def _protect(self) -> None:
        """0600 the db file — this table holds DOB and insurance IDs, which the
        other tables in memory.db don't. Matches sessions.db / audit.db."""
        try:
            if self.db_path.exists():
                os.chmod(self.db_path, 0o600)
        except OSError:
            logger.warning("Couldn't restrict permissions on %s", self.db_path, exc_info=True)

    # ------------------------------------------------------------------ CRUD
    def set(self, key: str, value: str, *, context: str = DEFAULT_CONTEXT,
            source: str = "user") -> None:
        if key not in PROFILE_KEYS:
            raise UnknownProfileKey(key)
        context = _clean_context(context)
        value = (value or "").strip()
        if not value:
            self.delete(key, context=context)
            return
        self.conn.execute("""
            INSERT INTO profile (key, context, value, source, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(key, context) DO UPDATE SET
                value=excluded.value, source=excluded.source, updated_at=datetime('now')
        """, (key, context, value, source))
        self.conn.commit()

    def get(self, key: str, context: str = DEFAULT_CONTEXT) -> Optional[str]:
        return self.all(context).get(key)

    def delete(self, key: str, context: str = DEFAULT_CONTEXT) -> bool:
        cur = self.conn.execute(
            "DELETE FROM profile WHERE key = ? AND context = ?",
            (key, _clean_context(context)))
        self.conn.commit()
        return cur.rowcount > 0

    def forget(self, key: str) -> int:
        """Drop a fact from every context."""
        cur = self.conn.execute("DELETE FROM profile WHERE key = ?", (key,))
        self.conn.commit()
        return cur.rowcount

    def stored(self, context: str = DEFAULT_CONTEXT) -> Dict[str, str]:
        """What's in the table for this context, overlaid on 'default'.

        No derived values — this is the raw storage view.
        """
        context = _clean_context(context)
        out = {k: v for k, v in self.conn.execute(
            "SELECT key, value FROM profile WHERE context = ?",
            (DEFAULT_CONTEXT,)).fetchall()}
        if context != DEFAULT_CONTEXT:
            out.update({k: v for k, v in self.conn.execute(
                "SELECT key, value FROM profile WHERE context = ?", (context,)).fetchall()})
        return out

    def sources(self, context: str = DEFAULT_CONTEXT) -> Dict[str, str]:
        context = _clean_context(context)
        out = {k: s for k, s in self.conn.execute(
            "SELECT key, source FROM profile WHERE context = ?",
            (DEFAULT_CONTEXT,)).fetchall()}
        if context != DEFAULT_CONTEXT:
            out.update({k: s for k, s in self.conn.execute(
                "SELECT key, source FROM profile WHERE context = ?", (context,)).fetchall()})
        return out

    def rows(self) -> List[tuple]:
        """(context, key, value, source) for everything stored — for the CLI."""
        return self.conn.execute(
            "SELECT context, key, value, source FROM profile "
            "ORDER BY context != 'default', context, key").fetchall()

    def contexts(self) -> List[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT context FROM profile ORDER BY context != 'default', context")]

    def all(self, context: str = DEFAULT_CONTEXT) -> Dict[str, str]:
        """Everything we can supply for this context, stored plus derived.

        A context is an *overlay* on 'default': set a name and phone under
        'formal' and everything else (email, date of birth, insurance) still
        comes from the shared default, so facts aren't duplicated per persona.
        """
        stored = self.stored(context)
        out: Dict[str, str] = {}
        for field in PROFILE_FIELDS:
            value = stored.get(field.key) or self._derive(field.key, stored)
            if value:
                out[field.key] = value
        return out

    def known_keys(self, context: str = DEFAULT_CONTEXT) -> List[str]:
        return list(self.all(context))

    @staticmethod
    def _derive(key: str, stored: Dict[str, str]) -> Optional[str]:
        """First/last name from the full name when not set explicitly.

        Fixes the long-standing duplication in the OpenTable channel, which put
        the whole guest name into both the first- and last-name inputs.
        """
        if key not in ("first_name", "last_name"):
            return None
        parts = (stored.get("full_name") or "").split()
        if len(parts) < 2:
            # A single-token name is a first name; guessing a surname is worse
            # than leaving the field for the user.
            return parts[0] if parts and key == "first_name" else None
        return parts[0] if key == "first_name" else " ".join(parts[1:])

    # ------------------------------------------------------- LLM-facing view
    def descriptors(self, context: str = DEFAULT_CONTEXT) -> List[Dict[str, str]]:
        """Keys, labels and descriptions for the facts we hold — **no values**.

        This is the only projection of the profile that may cross an egress
        sink. A mapper picks a key from it; the value is substituted here.
        The context never leaves this process either — the model has no idea
        which persona it's filling in.
        """
        have = self.all(context)
        return [
            {"key": f.key, "label": f.label, "description": f.description}
            for f in PROFILE_FIELDS if f.key in have
        ]

    def values(self, keys, context: str = DEFAULT_CONTEXT) -> Dict[str, str]:
        """Actual values for the named keys. Local substitution only — never
        put the result of this into an LLM prompt or any other sink."""
        have = self.all(context)
        return {k: have[k] for k in keys if k in have}

    # ----------------------------------------------------------------- seeding
    def seed_from_env(self) -> List[str]:
        """Adopt values from the reservation env vars the user already sets.

        Env vars are a bootstrap, not a source of truth: once the user has
        asserted a fact themselves — in *any* context — env seeding leaves that
        key alone for good. Otherwise a stale RESERVATION_GUEST_NAME would keep
        reinstating itself alongside the personas that replaced it.
        """
        asserted = {k for (k,) in self.conn.execute(
            "SELECT DISTINCT key FROM profile WHERE source = 'user'")}
        stored = self.stored()
        seeded: List[str] = []
        for env_var, key in _ENV_SEEDS.items():
            value = (os.getenv(env_var) or "").strip()
            if not value or key in asserted or stored.get(key) == value:
                continue
            self.set(key, value, source="env")
            seeded.append(key)
        return seeded

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# Businesses that keep records under your legal name: anything medical,
# clinical, legal, financial, insurance-bearing or governmental.
_FORMAL_RE = re.compile(
    r"\b(clinic|clinical|medical|medicine|health|healthcare|hospital|patient|"
    r"doctor|dr\.?|physician|physical therapy|physio|physiotherap\w*|therapy|"
    r"therapist|psychiatr\w*|psycholog\w*|counsel\w*|chiropract\w*|orthopa?ed\w*|"
    r"dental|dentist|orthodont\w*|optometr\w*|ophthalmolog\w*|dermatolog\w*|"
    r"podiatr\w*|pediatric\w*|obgyn|urgent care|surgery|surgical|radiolog\w*|"
    r"lab work|blood work|vaccination|immuni[sz]ation|prescription|"
    r"insurance|insurer|medicare|medicaid|copay|deductible|"
    r"attorney|lawyer|legal|law firm|solicitor|notary|"
    r"bank|mortgage|loan|financial advis\w*|accountant|tax|"
    r"dmv|passport|visa appointment|consulate|embassy|government)\b", re.I)

# Form fields only a records-keeping business asks for. Stronger evidence than
# any guess from the business name, because it comes from the form itself.
_FORMAL_FIELD_KEYS = frozenset({
    "date_of_birth", "insurance_provider", "insurance_member_id"})


def choose_context(text: str = "", field_keys=()) -> str:
    """Which persona to fill a form with: formal or casual.

    Two signals, strongest first:

    1. The form asks for a date of birth or insurance details. Only somewhere
       that keeps records under your legal name needs those, and this is read
       off the actual form rather than inferred.
    2. The business or service reads as medical/legal/financial/governmental.

    Everything else is casual — restaurants, salons, spas, the everyday
    bookings that outnumber the formal ones.
    """
    if _FORMAL_FIELD_KEYS & set(field_keys or ()):
        return FORMAL_CONTEXT
    if _FORMAL_RE.search(text or ""):
        return FORMAL_CONTEXT
    return CASUAL_CONTEXT


def match_key(text: str) -> Optional[str]:
    """Deterministic label → profile key matching (the no-LLM fallback path).

    Whole words only. A plain substring test reads "Tell Us About Your
    Condition" as the `tel` alias and types a phone number into the free-text
    box — which is exactly what it did before this used word boundaries.

    Longest alias first, so "first name" beats full_name's bare "name" alias.
    """
    low = (text or "").strip().lower()
    if not low:
        return None
    for alias, key in _ALIAS_INDEX:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", low):
            return key
    return None


# Longest alias first; built once, since the registry is a module constant.
_ALIAS_INDEX = sorted(
    ((alias, f.key) for f in PROFILE_FIELDS
     for alias in {*f.aliases, f.label.lower()}),
    key=lambda pair: len(pair[0]), reverse=True,
)


# --------------------------------------------------------------------- CLI
def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m core.profile",
        description="Inspect and edit the durable facts Friday knows about you.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show every stored fact, by persona")
    sub.add_parser("fields", help="show every fact key Friday can hold")
    sub.add_parser("seed", help="adopt values from the RESERVATION_* env vars")

    def with_context(p):
        p.add_argument("--context", "-c", default=DEFAULT_CONTEXT, choices=CONTEXTS,
                       help="which persona (default: %(default)s)")
        return p

    p_show = with_context(sub.add_parser(
        "show", help="show the facts a form would be filled with, for one persona"))
    p_get = with_context(sub.add_parser("get", help="print one fact"))
    p_get.add_argument("key")
    p_set = with_context(sub.add_parser("set", help="store one fact"))
    p_set.add_argument("key")
    p_set.add_argument("value", nargs="+")
    p_del = with_context(sub.add_parser("delete", help="forget one fact"))
    p_del.add_argument("key")
    p_del.add_argument("--all-contexts", action="store_true",
                       help="forget it under every persona")
    p_which = sub.add_parser(
        "which", help="show which persona a given business or request would use")
    p_which.add_argument("text", nargs="+")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass

    profile = UserProfile()

    if args.cmd == "fields":
        for f in PROFILE_FIELDS:
            suffix = "  (derived from full_name)" if f.derived else ""
            print(f"{f.key:24} {f.description}{suffix}")
        return 0

    if args.cmd == "list":
        profile.seed_from_env()
        rows = profile.rows()
        if not rows:
            print("Nothing stored yet. Try: python -m core.profile set full_name \"Your Name\"")
            return 0
        current = None
        for context, key, value, source in rows:
            if context != current:
                current = context
                header = {DEFAULT_CONTEXT: "default  (shared by every persona)",
                          FORMAL_CONTEXT: "formal   (medical, legal, financial, government)",
                          CASUAL_CONTEXT: "casual   (restaurants, salons, spas, everyday)"}
                print(f"\n[{header.get(context, context)}]")
            print(f"  {key:24} {value}   [{source}]")
        print()
        return 0

    if args.cmd == "show":
        resolved = profile.all(args.context)
        if not resolved:
            print(f"(nothing resolves for the {args.context} persona)")
            return 1
        print(f"\nWhat a {args.context} form would be filled with:\n")
        for key, value in resolved.items():
            print(f"  {key:24} {value}")
        print()
        return 0

    if args.cmd == "which":
        text = " ".join(args.text)
        print(choose_context(text))
        return 0

    if args.cmd == "seed":
        seeded = profile.seed_from_env()
        print(f"Seeded from env: {', '.join(seeded)}" if seeded else "Nothing new to seed.")
        return 0

    if args.cmd == "get":
        value = profile.get(args.key, args.context)
        if value is None:
            print(f"(not set: {args.key})")
            return 1
        print(value)
        return 0

    if args.cmd == "set":
        try:
            profile.set(args.key, " ".join(args.value), context=args.context)
        except UnknownProfileKey:
            print(f"Unknown key {args.key!r}. See: python -m core.profile fields")
            return 2
        where = "" if args.context == DEFAULT_CONTEXT else f" ({args.context})"
        print(f"Stored {args.key}{where}.")
        return 0

    if args.cmd == "delete":
        if args.all_contexts:
            count = profile.forget(args.key)
            print(f"Forgotten from {count} persona(s)." if count else f"(not set: {args.key})")
            return 0
        print("Forgotten." if profile.delete(args.key, args.context)
              else f"(not set: {args.key})")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
