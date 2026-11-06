#!/bin/bash
# --pre-run wrapper for the Maestri routine "Olivia node watch".
#
# Does the real work (node_notify.sh posts to Slack on a state transition) and
# then EXITS NON-ZERO WHEN THERE WAS NOTHING TO SAY. That is deliberate: Maestri
# skips a fire whose pre-run exits non-zero, so the routine's desktop reminder
# only appears on a genuine transition instead of every 15 minutes forever.
#
# Echoes the notifier's line so the routine's {{output}} placeholder can carry it
# into the reminder text.
#
# Note the inversion: a non-zero exit here means "normal, nothing to report", not
# "error". That is unusual, and is why this lives in its own file with this
# comment rather than being inlined into the routine definition.
cd "$(dirname "$0")" || exit 1
out=$(./node_notify.sh 2>&1)
[[ -n "$out" ]] && echo "$out"
[[ "$out" == posted:* ]]
