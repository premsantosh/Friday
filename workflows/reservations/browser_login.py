"""
One-time login bootstrap for the reservation browser channels.

    python -m workflows.reservations.browser_login opentable
    python -m workflows.reservations.browser_login resy
    python -m workflows.reservations.browser_login yelp
    python -m workflows.reservations.browser_login generic https://some-booking-site.com

Opens a visible Chromium window on the platform's sign-in page using the same
persistent profile the booking channels use (RESERVATION_BROWSER_DIR, default
~/.friday/browser/<platform>). Log in once — including any 2FA — then close the
window; the session cookies persist and the channel reuses them. No passwords
are ever stored by Friday (spec §8 L3: the profile dir is chmod 0700).
"""

from __future__ import annotations

import os
import stat
import sys
import time

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
    print(f"Opening {url}\nProfile: {profile}")
    print("Log in (and finish any 2FA), then close the browser window.")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(profile, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url)
        closed = False
        try:
            while ctx.pages:
                time.sleep(1)
            closed = True
        except Exception:
            closed = True  # window closed under us — that's the expected exit
        finally:
            try:
                ctx.close()
            except Exception:
                pass
    if closed:
        print(f"Done — {platform} session saved. The booking channel will reuse it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
