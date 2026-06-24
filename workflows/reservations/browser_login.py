"""
One-time login bootstrap for the reservation browser channels.

    python -m workflows.reservations.browser_login opentable
    python -m workflows.reservations.browser_login resy
    python -m workflows.reservations.browser_login yelp
    python -m workflows.reservations.browser_login generic https://some-booking-site.com

Opens a visible Chromium window on the platform's sign-in page using the same
persistent profile the booking channels use (RESERVATION_BROWSER_DIR, default
~/.friday/browser/<platform>). Log in once — including any 2FA — then press Enter
in this terminal (leave the browser open). We snapshot the session cookies to a
0600 storage-state file the booking channel replays, so session cookies survive
relaunches (the persistent profile alone drops them). No passwords are ever
stored by Friday (spec §8 L3: profile dir and state file are chmod 0700/0600).
"""

from __future__ import annotations

import os
import stat
import sys

from .channels.base import persistent_context_options, storage_state_path

LOGIN_URLS = {
    "opentable": "https://www.opentable.com/signin",
    "resy": "https://resy.com/",          # sign-in is a modal on the home page
    "yelp": "https://www.yelp.com/login",
}


def _profile_dir(platform: str) -> str:
    root = os.path.expanduser(os.getenv("RESERVATION_BROWSER_DIR", "~/.friday/browser"))
    path = os.path.join(root, platform)
    os.makedirs(path, exist_ok=True)
    os.chmod(root, stat.S_IRWXU)   # 0700 — session cookies are account-takeover tokens
    os.chmod(path, stat.S_IRWXU)
    return path


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in (*LOGIN_URLS, "generic"):
        print(__doc__)
        return 2
    platform = argv[0]
    if platform == "generic":
        if len(argv) < 2:
            print("generic needs a URL: python -m workflows.reservations.browser_login generic <url>")
            return 2
        url = argv[1]
    else:
        url = LOGIN_URLS[platform]

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright isn't installed: pip install playwright && playwright install chromium")
        return 1

    profile = _profile_dir(platform)
    state_path = storage_state_path(profile)
    opts = persistent_context_options(profile)
    print(f"Opening {url}\nProfile: {profile}")
    if opts.get("channel"):
        print(f"Using your installed '{opts['channel']}' browser.")
    print("Log in (and finish any 2FA / human check) in the browser window.")

    saved = False
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(**opts)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url)
        try:
            # Snapshot must happen while the context is alive, so we wait on a
            # terminal keypress rather than the window close — closing the window
            # tears the context down before we can read storage_state.
            input("\nPress Enter HERE once you're logged in — keep the browser open... ")
            ctx.storage_state(path=state_path)
            os.chmod(state_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — auth tokens (§8 L3)
            saved = True
        except Exception as exc:
            print(f"Couldn't save the session: {exc}\n"
                  "Re-run and press Enter only after you've finished logging in "
                  "(don't close the browser first).")
        finally:
            try:
                ctx.close()
            except Exception:
                pass
    if not saved:
        return 1
    print(f"Done — {platform} session saved to {state_path}. The booking channel will replay it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
