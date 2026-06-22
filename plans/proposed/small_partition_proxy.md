# A durable LLM proxy on Olivia's `small` partition

**Status:** Option A implemented (built + unit/e2e-tested locally; on-cluster validation pending)
**Date:** 2026-06-20
**Branch:** `add-small-partition-proxy`

## TL;DR

Olivia's `small` partition (CPU-only, partial-node allocation, **7-day max
walltime**, default partition) is a good home for a long-lived proxy that
front-ends our ephemeral GPU (`accel`) inference jobs. It is the *sanctioned*
way to run a durable in-cluster process — through the queue system, not on the
login node.

But it does **not** eliminate the SSH tunnel. Olivia's compute nodes are not on
the public internet (only login nodes are), so the laptop still has to tunnel in
through the login node. What the `small`-partition proxy buys is:

1. A **stable in-cluster endpoint** for up to 7 days, decoupled from the
   8-hour-ish GPU job lifecycle — the laptop tunnel target stops changing every
   time the GPU job restarts or moves nodes.
2. A **policy-clean replacement** for today's opt-in `LOGIN_PROXY` relay, which
   runs a long-lived daemon on the shared login node (against NRIS policy, as
   the code itself flags).
3. A place to run the **SSE batching proxy next to the backend** (over the fast
   internal Slingshot fabric) so the slow hop the laptop crosses is already
   batched.

**Recommendation:** build a `small`-partition proxy job and point the laptop
tunnel at it. Request *minimal* CPU/memory (partial node) and accept the 7-day
cap (resubmit weekly). Do **not** pursue an outbound reverse-tunnel to escape
the SSH-tunnel requirement — that likely violates the acceptable-use policy.

---

## 1. What's fragile today

The current path from a laptop to a model (see `CLAUDE.md` and
`anthropic_proxy.py:13`):

```
Claude Code ──/v1/messages──► anthropic_proxy ──/v1/chat/completions──► vllm:8000
 (laptop)      (Anthropic)      (laptop :8002)         │              (GPU node, accel)
                                                       │
                          localhost:8003 ──SSH tunnel──┘ (via login node)
```

- **vLLM runs on `accel`** with `#SBATCH --time=08:00:00`
  (`run_vllm_server.sh:6`). When the job ends or is requeued, the endpoint
  disappears and usually comes back on a *different* node name.
- **The laptop SSH tunnel** (`olivia.sh`) forwards `localhost:8003` → GPU
  node:8000 through the login node. It dies on laptop sleep / network drops, and
  Olivia requires **password + OTP on every new SSH connection** (keys don't
  bypass 2FA), so a genuine drop means re-authenticating. `autossh` is
  deliberately not used for exactly this reason (`olivia.sh:82`).
- **The durability hack we already have** — opt-in `LOGIN_PROXY`
  (`olivia.sh:243`) — runs a tiny user-space TCP relay (`~/.olivia/relay.py`) on
  the **login node**, listening on a fixed port (18000) and following the GPU
  node across job moves. It works, but the code carries an explicit warning:
  *"this runs a long-lived process on the shared login node — check the NRIS
  acceptable-use policy."* That warning is correct: see §3.

So the durability problem is really two problems: (a) the **endpoint** moves
with the GPU job, and (b) the **ingress** (laptop → cluster) is a fragile,
2FA-gated SSH tunnel. The `small` partition can fix (a) cleanly. It can only
*improve*, not remove, (b).

## 2. Sigma2 / Olivia policy findings

Sources (Sigma2 documentation):

- Job types on Olivia — <https://documentation.sigma2.no/jobs/job_types/olivia_job_types.html>
- Olivia system architecture — <https://documentation.sigma2.no/hpc_machines/olivia/overview.html>
- Using shared resources responsibly — <https://documentation.sigma2.no/computing/responsible-use.html>
- Login vs compute node internet access — <https://documentation.sigma2.no/jobs/internet-login-compute-nodes.html>
- Projects and accounting — <https://documentation.sigma2.no/jobs/projects_accounting.html>
- Acceptable use policy — <https://www.sigma2.no/acceptable-use-policy>

### 2.1 The `small` partition

| Property | Value |
|---|---|
| Hardware | CPU-only (AMD EPYC Turin, 256 cores / ~741 GiB usable per node) |
| Partial-node allocation | **Yes** — billed per requested CPU + memory |
| Max walltime | **7 days** |
| Per-job cap | 256 billing units, **max 1 node** |
| Nodes | 88 |
| Default partition | **Yes** (`--partition=small` can be omitted) |
| Intended use | memory-intensive CPU work needing < 1 node; small / dev / test jobs |

A few-CPU, few-GiB proxy is exactly the kind of sub-node job partial allocation
is designed for, and 7 days is far more durable than the GPU job's ~8 h.

### 2.2 Network model — the decisive constraint

- **Login nodes are on the public internet; compute nodes are not.** You reach a
  service on a compute node only from inside the cluster (or by tunnelling
  through a login node). There is no inbound path from the internet to a `small`
  node.
- **Compute nodes' only outbound internet is via an HTTP proxy**
  (`http_proxy=http://10.63.2.48:3128/`), intended for `pip`/`conda`. This is
  not a general bidirectional channel.

Consequence: a proxy on `small` is reachable by us only via the login node. The
SSH tunnel does not go away — but its *target* can now be a stable `small` node
instead of the moving GPU node, and the login node reverts to being a plain SSH
jump host (allowed) rather than a persistent-daemon host (not allowed).

### 2.3 Login-node and responsible-use rules

- Login nodes are **only** for "file transfer, compilation, editing, job
  submission and short tests." Don't run interactive calculations or persistent
  services on them. **"Always use the queue system for running jobs."**
- Responsible use: **"Do not ask for a lot more memory or number of cores or
  time than you need"** — it depletes quota and delays others. Don't use
  `--exclusive` unless told to.

This is the crux: the `small`-partition proxy is *the policy-correct version* of
the `LOGIN_PROXY` relay. We move the long-lived process off the shared login
node and into the scheduler, which is precisely what the docs ask for.

### 2.4 Accounting

- You're billed for resources **requested**, not used (unused *time* is
  refunded, but the CPU/memory reservation is billed for the wallclock the job
  is up). Billing units = max(CPU cost, memory cost).
- A proxy that idles 99% of the time still bills its small reservation
  continuously. At, say, 2 CPU it's ≈ 2 × 24 × 7 = **~336 billing-unit-hours per
  week** — small, but non-zero, and it sits in mild tension with "don't request
  more than you need" if left running while no one is using a model. Keep the
  request minimal and stop the job when idle for long stretches.

### 2.5 Acceptable use

Resources may be used only "consistent with the stated goals … of the project,"
and you must not "breach or circumvent any administrative or security controls."
A proxy that front-ends *our own* inference experiments is within project scope.
A general always-on public gateway, or anything that tunnels *out* to bypass the
network controls (see §4, option C), is not.

## 3. What a `small`-partition proxy actually buys (and doesn't)

**Buys:**

- ✅ **Stable endpoint, 7 days.** Clients point at a fixed `small`-node host:port.
  The GPU job can restart/move; the proxy re-points internally. No re-tunnelling
  on every GPU job.
- ✅ **Policy-clean durability.** Replaces the login-node relay daemon with a
  queue-system job — exactly what NRIS asks for.
- ✅ **Batching next to the backend.** `small → accel` is on the internal
  Slingshot fabric (fast). Run `vllm_proxy.py` (our SSE batcher, today colocated
  on the GPU node) on the `small` node instead, so only the *already-batched*
  stream crosses the slow laptop tunnel. Same effect, but now on a durable host.
- ✅ **Separation of concerns / cost.** The cheap, durable CPU front-end is
  decoupled from the expensive, ephemeral GPU back-end. You can cancel and
  resubmit GPU jobs (different models, restarts) without touching client config.

**Does not buy:**

- ❌ **No tunnel-free access.** Compute nodes aren't internet-reachable; the
  laptop still tunnels through the login node. The 2FA-on-new-connection
  fragility remains — but the tunnel target is now stable for 7 days, so
  re-tunnelling is rare and `ControlMaster`/`ServerAlive`/`ControlPersist`
  (already in `olivia.sh`) do more for us.
- ❌ **Not "permanent."** 7-day cap means weekly resubmission. Durable, not
  forever.

## 4. Options surveyed

### Option A — `small`-partition proxy + laptop tunnel to it  ✅ recommended

```
Claude Code ─► anthropic_proxy (laptop :8002)
                     │  SSH tunnel (login node as jump), target STABLE for 7 days
                     ▼
            small-node proxy  ── internal Slingshot ──►  accel GPU node : vLLM:8000
            (vllm_proxy batcher + follow-the-node relay,
             --partition=small, --time=7-00:00:00, ~2 CPU / ~4 GiB)
```

- Login node is used only as an SSH jump (allowed). No login-node daemon.
- The follow-the-node relay logic we already have moves *into* the `small` job.
- Tunnel target = the `small` node, which doesn't move for 7 days.
- Clean against every policy in §2. Mild accounting cost (§2.4) — mitigate by
  requesting minimal resources and stopping when idle.

### Option B — keep `accel`-only, just harden the tunnel  ⬇ status quo+

Improve keepalives / reconnect only. Doesn't address endpoint churn or the
login-node-relay policy issue. This is roughly where we are; not the ask.

### Option C — outbound reverse tunnel to a user VPS  ⛔ not recommended

Have the `small`-node proxy dial *out* (via the `10.63.2.48:3128` HTTP proxy) to
a user-controlled public host, giving a truly persistent public endpoint with no
laptop SSH tunnel. Rejected because: (a) outbound is HTTP-proxy-only and not a
general tunnel channel — a persistent reverse tunnel over it is unreliable and
likely blocked; (b) it is plausibly "circumventing administrative/network
controls," which the acceptable-use policy forbids. If a tunnel-free public
endpoint is genuinely wanted, **ask NRIS** rather than engineer around the
controls.

### Option D — ask NRIS for a sanctioned persistent-service mechanism  📨 parallel track

Some HPC centres expose Open OnDemand / interactive portals / service nodes for
exactly this. I found no evidence Olivia offers one yet (it's in its pilot
period). Worth a support ticket asking: *is there a supported way to run a small
persistent service endpoint, and is a multi-day `small` proxy job acceptable use?*
This de-risks Option A and might unlock something better.

## 5. Implementation (Option A — built 2026-06-20)

Built on this branch. It went a step beyond the original sketch: instead of a
single-upstream relay, the `small`-node process is a **multi-model router** that
selects the backend per request, so one endpoint serves every running model.

**New files**

- **`presets.json`** — single source of truth for the preset table (models,
  aliases, container prefix/index, node/gpu/pp shape). Extracted from the
  `olivia.sh` `preset_field`/`normalize_preset` case statements so the bash CLI
  and the router agree on the table. (Directly answers the "extract the case data
  into a data file" ask.)
- **`presets.py`** — stdlib-only accessor over `presets.json`. Used by the router
  (`import presets`) and by `olivia.sh` via a small CLI (`normalize` / `field` /
  `match` / `container` / ...). Semantics verified identical to the old bash
  accessors (aliases, `resources`, case-preserved custom prefixes, routing by
  repo id).
- **`model_router.py`** — the durable router. Routes `/v1/chat/completions`,
  `/v1/completions`, `/v1/models` by the request's `model` field. Discovery is
  **job-name-independent**: it lists the user's running `vllm-*` jobs via
  `squeue` and **probes each node's `/v1/models`** to learn what it serves,
  building a {served model → node} map (15 s TTL, retry once on a stale node),
  then resolves a preset name/alias/repo-id to a live backend. Reuses the proven
  SSE batching from `vllm_proxy.py`. Includes the **auto-spindown watchdog** and
  optional bearer-token auth (`OLIVIA_PROXY_TOKEN`).
- **`run_proxy.sh`** — the SLURM job: `--partition=small --cpus-per-task=2
  --mem=4G --time=7-00:00:00`. Provisions a tiny aiohttp venv (via the NRIS
  compute-node HTTP proxy, mirroring the prefetch venv pattern) and launches the
  router with `--backend-port`/`--empty-timeout`.

**Changed files**

- **`vllm_proxy.py`** — extracted the SSE batching loop into reusable
  `forward_stream` / `forward_nonstream` module functions (the router and the
  single-upstream proxy now share one implementation; behaviour unchanged).
- **`olivia.sh`** —
  - `preset_field` / `normalize_preset` now delegate to `presets.py` (the giant
    case statements are gone);
  - new `proxy start|status|stop|logs|deploy|tunnel` command group, plus a
    standalone `find_job_node` (exact-match) to locate the router node, and
    `reconnect` is router-tunnel-aware;
  - **`LOGIN_PROXY` (the login-node relay) removed entirely** — vars, the
    `relay.py` deploy/ensure/stop functions, the `setup_tunnel`/`kill_tunnel`
    branches, and the `--login-proxy` flags on `tunnel`/`chat`/`reconnect`. The
    `proxy` module is its policy-clean replacement.

**Design choices worth noting**

- **Job-name-independent discovery (probe `/v1/models`).** This was the key
  decision after reconciling with `main`'s commit `0ea204e`, which renames each
  branch's server `vllm-<DEPLOY_KEY>` (per-agent isolation) with exact-match
  resolution. Rather than fight that with a competing per-preset naming, the
  router asks each running server what it serves. So it composes with `main`'s
  scheme, needs **no change to `start_server_job`/`find_vllm_node`** (smaller
  merge surface), and is more robust.
- **All servers serve on port 8000.** Each vLLM job owns its GPU node(s)
  exclusively, so no port collision; the router probes `node:8000`. `--backend-port`
  overrides if that ever changes.
- **SLURM + live probe are the source of truth** (no registry file to go stale):
  a dead job is absent from `squeue`; a loading server simply isn't in the map.
- **Auto-spindown.** The router exits after `--empty-timeout` (default 1800 s =
  30 min) with no GPU servers running, ending the SLURM job and freeing the
  `small` allocation — addresses the idle-billing tension in §2.4. Startup grace
  + transient-error tolerance built in.
- **`anthropic_proxy.py` is unchanged** — point its `--upstream` at the router
  tunnel (`http://localhost:8003`) and pick a model with `--model <preset>`.

**Reconciliation with `main` (checked 2026-06-20)**

`main` advanced 4 commits during the work (→ `54bc013`). Findings: preset data in
`preset_field`/`normalize_preset` is **identical** (so `presets.json` is
complete); `513eff4`/`93d3335` don't touch `olivia.sh`; only `0ea204e` (per-branch
job name + exact-match resolution) overlapped. The rework above (probe-based
discovery + reverting the job-name/`find_vllm_node` edits) makes this branch
merge cleanly onto `main`.

**Verified locally**

- `presets.py` output matches the old bash accessors across all presets, aliases,
  and a custom name (case-preserved prefix).
- End-to-end router test (fake upstream serving `/v1/models` + `/v1/chat/completions`,
  stubbed `squeue`): routing by preset name / alias / repo id, model→served-id
  rewrite, SSE streaming reassembly, `/v1/models`, 404 (unknown model), 503
  (known preset not running).
- Auto-spindown watchdog raises `SIGTERM` after the empty window (and is reset by
  servers reappearing / by transient errors).
- `bash -n` clean on `olivia.sh` and `run_proxy.sh`; cross-module imports OK; all
  `LOGIN_PROXY` references removed.

## 6. On-cluster validation

Validated live 2026-06-22 by the agentic-evals harness (cross-model SWE sweep
routed through the router on `c1-4:8080`):

- [x] `python3` + `venv` + `pip` work on a `small` node and the NRIS HTTP proxy
      reaches PyPI — **after** the fix below. (First bring-up failed: the small
      nodes' system `python3` is 3.6.15, too old for `model_router.py`; fixed by
      `run_proxy.sh` autodetecting a ≥3.9 interpreter — commit `a3d4954`.)
- [x] The router reaches `gpu-node:8000/v1/models` over the internal fabric
      (probe discovery surfaced the laguna backend), and a `small` node's socket
      is reachable from other compute nodes (the eval jobs ran on `small` nodes
      and hit `c1-4:8080`).
- [x] Multi-model routing, **concurrent**: glm52 + kimi27 served at once, both
      eval jobs routed through the one endpoint by model name (glm52 7/12,
      laguna 5/12, kimi27 3/12).

Still to confirm:

- [ ] Auto-spindown actually fires 30 min after the last server stops (and not
      while one is up). The `no_proxy` hardening in `run_proxy.sh` lands first so
      the next run exercises a clean router.
- [ ] The laptop → login-node → `proxy tunnel` → router path (the 2026-06-22
      sweep was entirely cluster-side, so this human/Claude-Code path is untested).
- [ ] Token auth (`OLIVIA_PROXY_TOKEN`).
- [ ] Ask NRIS (Option D): is a multi-day `small` proxy job acceptable use, and
      is there a supported persistent-service path?

## 7. References

All under <https://documentation.sigma2.no> unless noted:

- Job types on Olivia · `jobs/job_types/olivia_job_types.html`
- Olivia architecture · `hpc_machines/olivia/overview.html`
- Responsible use · `computing/responsible-use.html`
- Login vs compute internet · `jobs/internet-login-compute-nodes.html`
- Projects & accounting · `jobs/projects_accounting.html`
- SSH · `getting_started/ssh.html`
- Acceptable use policy · <https://www.sigma2.no/acceptable-use-policy>
- Codebase: `olivia.sh:243` (LOGIN_PROXY relay), `run_vllm_server.sh:6`
  (accel/8h), `anthropic_proxy.py:13` (architecture), `vllm_proxy.py` (SSE
  batcher)
