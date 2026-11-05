#!/bin/bash
# Start the Spark queue automatically after every boot (for example after a power cut),
# by adding one @reboot line to this user's crontab. Remove it with uninstall_autoresume.sh
# once the runs are finished. The queue refuses to start twice, so this is safe to combine
# with starting it by hand.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LINE="@reboot sleep 120 && cd $REPO && bash scripts/spark/queue.sh  # faultline-autoresume"
( crontab -l 2> /dev/null | grep -v "faultline-autoresume"; echo "$LINE" ) | crontab -
echo "installed: the queue starts 2 minutes after every boot"
crontab -l | grep faultline-autoresume
