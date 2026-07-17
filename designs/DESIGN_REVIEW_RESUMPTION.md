# Design: Review Resumption and Stage Pacing

## Problem

A review that fails part-way through discards everything it has done. The next
attempt starts again at stage 1 and pays for every stage a second time.

The failure modes that trigger this are routine rather than exotic:

- `review.max_total_output_tokens` aborts a review that has generated too much.
  The abort fires *because* the review is expensive, which means the work being
  thrown away is the expensive kind.
- The AI provider fails: rate limits, a subscription window closing mid-review,
  a transport error. Subscription-backed providers (`claude-cli`, `copilot-cli`)
  hit this on a schedule, not at random.
- The worker process dies, or the daemon restarts.

`reviewer.rs` then retries up to `review.max_retries` times, and each attempt
redoes the same work at the same cost. Worse, the second attempt frequently
fails the same way the first did, so the retries compound the loss instead of
recovering from it.

Two properties of the pipeline make this expensive:

1. **Stages 1-7 ran with unbounded parallelism** (`try_join_all`), so a single
   large patch could have seven LLM sessions in flight at once. A failure at any
   point discarded all seven partial results.
2. **Nothing was persisted until the review finished.** Stage results lived in
   the worker process and died with it.

Concretely: a review can burn most of its budget on the expensive early
stages, trip the abort near the end, and lose all of it — the retry then
begins at stage 1 with a full budget to burn again.

## Goals

- A retried review must not re-run stages that already succeeded.
- A checkpoint must never change a review's findings, only its cost.
- A stale checkpoint must be impossible to reuse.
- The failure of a checkpoint must never fail a review that otherwise worked.
- Operators must be able to trade wall-clock time for a gentler token burn.

## Non-goals

- Resuming *within* a stage. Stages are single logical LLM conversations; the
  interesting checkpoint boundary is stage completion. Turn-level checkpointing
  would have to persist provider-side conversation state and is out of scope.
- Checkpointing stages 8-11 (dedup, conflict resolution, verification, report).
  They are sequential, comparatively cheap, and consume the stage 1-7 output, so
  resuming into the middle of them saves little and complicates invalidation.
- Sharing checkpoints between machines or reviews. Checkpoints are a local cache
  keyed to one review on one host.

## Design

### 1. Stage pacing: `ai.stage_concurrency`

Stages 1-7 now run through a bounded `futures::stream::buffered(n)` instead of
`try_join_all`. `buffered` preserves input order, so result consolidation is
unchanged and remains deterministic.

The stages share one review-level output-token budget
(`review.max_total_output_tokens`) and one provider. When that budget is
exceeded, or the provider becomes unstable, every stage in flight at that
moment fails together and the tokens it has already generated are wasted --
and for the token budget, the review overshoots the ceiling by whatever the
in-flight stages spend before the abort propagates. The loss scales with how
many stages run at once.

`ai.stage_concurrency` caps how many stages are in flight. The default of `0`
means unbounded: every planned stage runs at once, reproducing the previous
`try_join_all` behaviour exactly and independently of how many stages exist,
so adding a stage later never silently caps the fan-out. A value of `1` runs
stages serially; any `N` caps them at N, bounding both the blast radius of a
shared-fate failure and the token-budget overshoot.

This is deliberately a per-review control, distinct from two nearby mechanisms:

- the global LLM request semaphore in `reviewer.rs` bounds in-flight *requests*
  across all reviews (throughput), not how many stages one review keeps open;
- stage checkpointing (below) avoids re-running *completed* stages on a retry,
  but cannot help the stages that were in flight when a shared-fate failure
  hit. Only bounding concurrency limits that loss, so this stands on its own.

At `stage_concurrency = 1` an interrupted review loses at most one stage.

### 2. Stage checkpoints

Each analysis stage writes its `StageExecutionResult` to
`<checkpoint_dir>/<scope>/stage-<N>.json` as soon as it completes. Before
running a stage, the worker looks for a matching checkpoint and reuses it
instead of calling the model.

Checkpoints are removed when the review completes, since only a retry can use
them.

### 3. Invalidation

Reuse is only safe if the stored result is what the current run would have
produced. Two mechanisms enforce this:

**Scope** (`local_review.rs::checkpoint_scope`) — a directory name derived from
the patch sha, the baseline sha, the model and the patch index. Different
reviews land in different directories and cannot see each other's stages.

**Fingerprint** (`prompts.rs::stage_fingerprint`) — stored inside each
checkpoint file: a hash of the stage number and the fully rendered system
prompt. The system prompt already contains the patch, the baseline log, the
subsystem prompts selected by the Phase 0 pre-screen, and any custom prompt, so
hashing it captures every input that feeds the stage. A checkpoint whose
fingerprint does not match is ignored, not reused.

The fingerprint is what makes this safe rather than merely convenient. Phase 0
prompt selection is itself an LLM call and can legitimately differ between
attempts; when it does, the fingerprints differ, the checkpoints are ignored,
and the stages re-run. Resumption is a best-effort optimisation that silently
degrades to the current behaviour, which is the correct bias.

### 4. Failure handling

Checkpointing must not introduce a new way to fail a review:

- Writes go to a temporary file and are renamed into place, so an interrupted
  write cannot leave a half-written checkpoint for a later run to parse.
- An unreadable or malformed checkpoint is logged and ignored.
- A failed write is logged and swallowed; the review continues.
- Cleanup failures are logged and swallowed.

### 5. Location

Checkpoints live under `review.checkpoint_dir` (default `review_checkpoints`),
deliberately **not** under `worktree_dir`: `reviewer.rs` calls `remove_dir_all`
on the worktree directory at daemon startup, which would defeat resumption
across exactly the restart it is meant to survive.

The daemon passes `--checkpoint-dir` to the `review` binary, mirroring
`--worktree-dir`. `sashiko-cli local` resolves it from settings. An explicit
flag wins over configuration; an empty configured value disables checkpointing.

## Configuration

| Setting | Default | Effect |
|---|---|---|
| `ai.stage_concurrency` | `0` | Analysis stages in flight per review. `0` = unbounded (all at once), `1` = serial, `N` = capped at N. |
| `review.checkpoint_dir` | `review_checkpoints` | Where checkpoints live. Empty disables. |

Both defaults preserve existing behaviour, except that a retry now resumes.

## Alternatives considered

**Store checkpoints in the database.** The review worker is a separate process
that deliberately does not hold a database handle; the daemon owns the DB. Adding
one to the worker would widen its interface and its blast radius for a cache.
Files under a configured directory keep the worker's contract unchanged.

**Key checkpoints by patch sha alone.** Cheaper, but wrong: it ignores the model
and the prompt bundle, so changing either would silently reuse results computed
from something else.

**Hash the patch instead of the system prompt.** Not sufficient. Two runs on the
same patch can select different subsystem prompts in Phase 0, which changes the
stage's actual input. Hashing the rendered system prompt covers the patch and
everything else that reaches the model.

**Checkpoint every turn within a stage.** Would salvage more from a mid-stage
failure, but requires persisting and replaying provider-side conversation state
(including thought signatures), which is provider-specific and fragile. Stage
granularity captures most of the value for a fraction of the complexity.

**Make resumption opt-in.** Rejected: the safety comes from the fingerprint, not
from the setting, and a knob defaulted to off would mostly serve to keep the
feature untested. It remains disableable for operators who want strict
determinism per attempt.

## Testing

- `checkpoint_scope_is_stable_for_identical_inputs` — the same review resolves
  to the same scope.
- `checkpoint_scope_changes_with_every_input` — patch sha, baseline, model,
  patch index and a missing sha each produce a distinct scope.
- Existing worker tests cover the unchanged consolidation path; `buffered`
  preserves ordering, so stage results are consolidated in the same order as
  before.

Manual verification: run a review with `stage_concurrency = 1`, kill the worker
mid-run, and confirm the next attempt logs "resumed from checkpoint" for the
completed stages and re-runs only the interrupted one.

## Future work

- Age out orphaned checkpoint directories (a review that never retries leaves
  its scope behind). A startup sweep with an mtime bound would do it.
- Surface resumed stages in the web UI, so an operator can see that a retry
  reused work rather than repeating it.
- Consider checkpointing stages 8-10 if verification costs grow.
