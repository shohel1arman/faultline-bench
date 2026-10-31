#!/bin/zsh
# Stop starting the queue at login. Does not stop a queue that is already running.
PLIST="$HOME/Library/LaunchAgents/faultline.queue.plist"
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "removed $PLIST"
