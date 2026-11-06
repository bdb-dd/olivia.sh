#!/bin/bash
# Posts to Slack when GH200 node availability MEANINGFULLY CHANGES.
#
# Companion to node_watch.sh, which samples availability on the Olivia login node
# every 10 minutes into node_avail.tsv. That sampler is durable but SILENT — it
# cannot reach Slack, because the bridge and its token live on the workstation,
# not the cluster. This script closes that gap: run it from a schedule on the
# workstation and it reads the cluster's time series and speaks up.
#
# WHY TREND AND NOT JUST "A NODE IS FREE": by the time usable>=1 the window is
# already open, and a 4-GPU single-node shape is the hardest allocation we ask
# for — it can close before anyone reads the message. The leading indicator is
# `ahead`, the number of PENDING jobs outranking a freshly submitted devel job
# (devel scores ~20000). That was 33 on Friday evening; as it falls our odds rise
# well before a node actually frees.
#
# Posts at most once per state transition, tracked in a state file, so a long
# quiet spell does not become a stream of identical messages.
#
# Usage:  node_notify.sh          (reads $BRIDGE_CANVAS, defaults to Olivia)
#         STATE=/path/to/state node_notify.sh    (override, e.g. for testing)
set -uo pipefail

BH="${MAESTRI_BRIDGE_HOME:-$HOME/dev/itonomi/maestri-slack-bridge/main}"
CANVAS="${BRIDGE_CANVAS:-Olivia}"
STATE="${STATE:-$HOME/.cache/olivia_nodewatch_state}"
REMOTE="${REMOTE_SAMPLE_DIR:-/cluster/work/projects/nn10104k/containers}"
mkdir -p "$(dirname "$STATE")"

# Exit quietly on any unreadable sample. A notifier that guesses is worse than
# one that says nothing, and the sampler keeps its own record regardless.
row=$(timeout 60 ssh -o BatchMode=yes olivia "tail -1 $REMOTE/node_avail.tsv" 2>/dev/null) || exit 0
[[ -z "$row" || "$row" == timestamp* ]] && exit 0
IFS=$'\t' read -r ts dow hr usable ge4 ahead names <<<"$row"
[[ "$usable" =~ ^[0-9]+$ ]] || exit 0
[[ "$ahead"  =~ ^[0-9]+$ ]] || ahead=999

# Best `ahead` over the last 12 samples (~2 h), so a single lucky sample is not
# mistaken for a downtrend.
recent_min=$(timeout 60 ssh -o BatchMode=yes olivia \
  "tail -12 $REMOTE/node_avail.tsv | awk -F'\t' '\$6 ~ /^[0-9]+\$/ {print \$6}' | sort -n | head -1" 2>/dev/null)
[[ "$recent_min" =~ ^[0-9]+$ ]] || recent_min=$ahead

if   (( usable >= 1 )); then state=OPEN
elif (( ahead  <= 10 )); then state=CLOSE
elif (( ahead  <= 20 )); then state=IMPROVING
else                         state=BUSY
fi

prev=$(cat "$STATE" 2>/dev/null || echo NONE)
echo "$state" > "$STATE"
[[ "$state" == "$prev" ]] && exit 0

# Do not announce BUSY as the first thing ever said — "back to busy" reads as a
# non-sequitur with no earlier trend message to contrast against.
if [[ "$state" == "BUSY" && "$prev" == "NONE" ]]; then
    echo "baseline recorded: BUSY (ahead=$ahead) — not posting an opener"
    exit 0
fi

case "$state" in
  OPEN)      intent=--blocker
             msg="*NODE WINDOW OPEN — act now* ($ts, $dow ${hr}h)
*$usable* full 4-GPU node(s) usable by us right now: \`$names\`
Pending jobs outranking a fresh devel job: *$ahead* (was 33 on Friday evening)

This is the window you asked me to catch, and a 4-GPU single-node shape is the hardest allocation we request — it may not last. Say the word and I will submit, or go yourself:
\`\`\`
cd ~/dev/sigma2/olivia/main && QOS=devel TIME_LIMIT=02:00:00 ./olivia.sh server start <preset> -d
\`\`\`" ;;
  CLOSE)     intent=--ask
             msg="*Nodes are close* ($ts, $dow ${hr}h)
Pending jobs ahead of a fresh devel job down to *$ahead* (from 33 Friday evening); best over the last ~2 h was *$recent_min*. $ge4 node(s) have ≥4 free GPUs, $usable usable so far.

At this queue depth a devel job has a real chance of starting promptly. Want me to queue something now so we are first in line when one frees?" ;;
  IMPROVING) intent=--fyi
             msg="*Availability trending our way* ($ts, $dow ${hr}h)
Pending jobs ahead of a fresh devel job: *$ahead* (from 33 Friday evening), best over the last ~2 h *$recent_min*. Still $usable usable nodes, so nothing to grab yet — this is the first sign the queue is draining. No action needed." ;;
  BUSY)      intent=--fyi
             msg="*Back to busy* ($ts, $dow ${hr}h) — pending ahead is back up to *$ahead*, $usable usable nodes. Flagging so the earlier trend message is not left looking current." ;;
esac

if "$BH/bin/post-to-pi.sh" "$intent" --canvas "$CANVAS" "$msg" >/dev/null 2>&1; then
    echo "posted: $state (ahead=$ahead usable=$usable recent_min=$recent_min)"
else
    # Roll the state back so the next run retries rather than silently skipping
    # the transition we failed to announce.
    echo "$prev" > "$STATE"
    echo "POST FAILED for $state — state rolled back to $prev so the next run retries"
    exit 1
fi
