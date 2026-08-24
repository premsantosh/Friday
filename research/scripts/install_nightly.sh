#!/bin/bash
# Install (or reinstall) the nightly learning job as a launchd user agent.
# Runs `python -m research nightly` at 03:30 daily. The Mac must be awake
# (plugged in with `caffeinate` or a pmset wake schedule, e.g.:
#   sudo pmset repeat wakeorpoweron MTWRFSU 03:25:00 )
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TEMPLATE="$REPO/research/scripts/com.friday.nightly.plist"
TARGET="$HOME/Library/LaunchAgents/com.friday.nightly.plist"
LABEL="com.friday.nightly"

mkdir -p "$HOME/.friday/research/logs"

sed -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" "$TEMPLATE" > "$TARGET"

# Reload cleanly whether or not it was already installed.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"

echo "Installed $LABEL (03:30 daily)."
echo "  logs:   ~/.friday/research/logs/nightly.{out,err}.log"
echo "  run now: launchctl kickstart gui/$(id -u)/$LABEL"
echo "  remove:  launchctl bootout gui/$(id -u)/$LABEL && rm '$TARGET'"
