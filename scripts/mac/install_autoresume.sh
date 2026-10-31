#!/bin/zsh
# Start scripts/mac/queue.sh at every login, so the queue resumes after a restart.
# Needs automatic login (System Settings > Users & Groups) to resume without a person present.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/faultline.queue.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>faultline.queue</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>$REPO/scripts/mac/queue.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$REPO/outputs/agent.log</string>
  <key>StandardErrorPath</key><string>$REPO/outputs/agent.log</string>
</dict>
</plist>
PLISTEOF
chmod +x "$REPO/scripts/mac/queue.sh"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "installed $PLIST; the queue starts now and after every login"
echo "follow progress with: tail -f $REPO/outputs/queue.log"
