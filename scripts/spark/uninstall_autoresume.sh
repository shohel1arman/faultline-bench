#!/bin/bash
# Remove the @reboot line added by install_autoresume.sh.
( crontab -l 2> /dev/null | grep -v "faultline-autoresume" ) | crontab -
echo "removed the faultline auto-resume line from crontab"
