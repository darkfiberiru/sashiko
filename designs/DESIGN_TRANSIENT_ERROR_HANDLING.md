# Design: Transient Error and Session-Limit Handling

## Problem

The `claude-cli` provider collapsed every non-spawn, non-timeout CLI error into
`ClaudeCliError::Cli(String) -> AiErrorClass::Fatal`. That one bucket contained
two conditions that are not fatal at all:

- **Transient overload** — the CLI prints `API Error: Overloaded` (an HTTP 529).
  This clears in seconds; the correct response is a short backoff and retry.
- **Session limit** — the CLI prints `You've hit your session limit · resets 9pm
  (America/New_York)`. This is a subscription cap that clears at a specific
  wall-clock time, often hours away.

Both were treated as fatal, so:

- An overload aborted the review's current attempt. With `max_retries` the
  daemon retried the whole review, frequently hitting the same overload and
  compounding the loss.
- A session limit killed the review outright. Retrying was pointless — the limit
  does not clear for hours — but nothing told the retry logic that.

Neither condition warrants discarding work. An overload wants a brief wait; a
session limit wants a pause until the reset.

## Goals

- Retry transient overloads instead of failing.
- On a session limit, pause until just after the reset time, then continue,
  rather than failing or polling a doomed request.
- Keep the existing safety valve that stops a misbehaving provider from
  blocking indefinitely.
- Do not let an authoritative session-limit pause count against the small
  retry budgets used by short helper calls (Phase 0, planning).

## Non-goals

- Parsing arbitrary IANA timezones from the message. `chrono-tz` is not a
  dependency, and the CLI reports the reset in the machine's local zone, so the
  time is interpreted as local. If a machine's local zone ever differs from the
  zone the CLI reports, the pause could be off by the zone offset; the
  self-correcting retry (below) recovers even then. Adding `chrono-tz` and
  reading the parenthesised zone is the clean upstream fix.
- Cross-provider generalisation. The numeric HTTP codes (429/500/529) are
  already classified correctly for the API `claude` provider via
  `classify_status_code`. This change is specific to the CLI provider, which
  surfaces these conditions as free text.

## Design

### 1. A distinct error class

`AiErrorClass` gains a third recoverable variant alongside `RateLimit` and
`Transient`:

```rust
SessionLimit { retry_after: Duration }
```

It is separate from `RateLimit` on purpose. A rate limit is a guess bounded by a
short safety cap; a session limit is an *authoritative* wait the provider told
us about. The two deserve different ceilings and different budget treatment, and
a distinct variant states that at the type level. The enum is internally tagged
(`tag = "class"`), so the new variant crosses the worker/daemon stdio protocol
as `"class": "session_limit"` with no ambiguity.

### 2. Classifying the CLI message

`ClaudeCliError::Cli(msg)` is now classified by `classify_cli_message(msg, now)`:

- contains `session limit` -> parse the reset time -> `SessionLimit`;
- contains `overloaded` or `529` -> `Transient` (30s seed backoff);
- otherwise -> `Fatal` (unchanged).

`now` is injected rather than read inside the function so the reset arithmetic
is unit-testable.

`parse_session_reset` extracts `resets <time>` (`9pm`, `10:30am`, `12am`,
`12pm`) and returns how long to wait, keyed on where that local time falls
relative to `now`:

- **still ahead today** → the gap until it. `classify_cli_message` adds a
  5-minute buffer so the retry lands just after the reset instead of racing it.
- **just passed (within a 30-minute grace)** → a ~5-minute poll. This is the
  "hit the timer, still limited" case: we waited for the reported reset, retried,
  and got the *same* reset time back, which means the reset is running late. A
  short poll recovers within minutes. Without this, an already-passed time would
  be read as *tomorrow's* reset — a reset delayed by 5 minutes would otherwise
  put the review to sleep for ~12 hours (the `report_scheduled_block` ceiling).
- **well in the past (beyond the grace)** → the same time tomorrow, the next
  cycle. Being limited more than half an hour past the reported reset time is a
  fresh limit for the next window, not a delayed one.

If the time cannot be parsed, it falls back to a 30-minute pause — long enough
not to hammer the CLI, short enough to recover.

The 30-minute grace is a heuristic, not a proof: a reset delayed by *more* than
30 minutes falls into the tomorrow branch and over-waits (bounded by the 12-hour
ceiling and checkpoint-resume). The fully robust alternative — tracking "did we
just wait for this exact reset?" in the retry loop — is noted in Future work.

DST is handled: an ambiguous fall-back hour resolves to the earliest instant, a
skipped spring-forward hour returns `None` and takes the 30-minute fallback.

### 3. Honouring an authoritative wait

The quota manager capped every block at `MAX_RETRY_AFTER` (5 minutes). That is
right for a guessed rate-limit backoff but wrong for a known reset hours away —
it would turn one pause into dozens of doomed 5-minute retries.

`report_scheduled_block(retry_after)` is added alongside `report_quota_error`.
It caps at `MAX_SCHEDULED_BLOCK` (12 hours) instead of 5 minutes: long enough to
honour any real session reset, still bounded so a misparsed time cannot block
forever. The 12-hour ceiling is the self-correction backstop — if the parse is
wrong, the review resumes within 12 hours and re-evaluates.

### 4. Wiring

Three call sites match on `AiErrorClass`:

- **`reviewer.rs`** (main stage loop) — `SessionLimit` logs the pause and calls
  `report_scheduled_block`, then continues. The existing loop already extends
  the review's active-time deadline by whatever `wait_for_access` sleeps, so a
  multi-hour pause does not trip the review timeout, and there is no separate
  hard process kill to fight.
- **`proxy.rs`** (Gemini proxy) — mirrors `RateLimit`, using
  `report_scheduled_block`.
- **`session.rs`** (helper-call runner) — `SessionLimit` sleeps for
  `retry_after` **without** incrementing `transient_retries`. A session limit is
  not a flaky error, so it must not exhaust the small transient budget that
  guards Phase 0 and planning.

## Interaction with review resumption

This pairs with the stage-checkpoint work (DESIGN_REVIEW_RESUMPTION.md). If a
session limit is hit and, for any reason, the pause does not survive (a daemon
restart during the wait), the retried review resumes from its last completed
stage rather than restarting. Pause-in-place is the fast path; checkpoint-resume
is the backstop.

## Alternatives considered

**Reuse `RateLimit` and raise `MAX_RETRY_AFTER`.** Fewer lines, but it weakens
the 5-minute safety cap for *all* rate-limit paths to serve the one authoritative
case. A distinct variant keeps the guessed and known waits separate.

**Fail the review and reschedule it via the daemon after the reset.** More
robust to restarts and frees the worktree/slot during the wait, but needs
persistent per-patchset "retry not before T" state and a scheduler tick. It is
also unnecessary here: a session limit is account-wide, so nothing else could
run during the pause anyway — holding the slot costs nothing. Pause-in-place is
simpler and the checkpoint work already covers the restart case.

**Poll every 5 minutes (classify session limit as `RateLimit`).** Requires no
quota change but fires a doomed request every 5 minutes for hours and floods the
log. Rejected in favour of a single honoured pause, with the 12-hour ceiling and
checkpoint-resume as the safety nets.

## Testing

Unit tests in `claude_cli.rs` cover: overload and `529` -> `Transient`; unknown
message -> `Fatal`; `resets 9pm` from 3pm -> 6h + 5m; a reset already past today
rolling to tomorrow; minutes, noon and midnight parsing; and the unparseable
fallback.

Manual verification: hit a real session limit (or inject the message) and
confirm the log shows a single "Pausing … until reset" line and the review
resumes shortly after the reset time.

## Future work

- Add `chrono-tz`, read the parenthesised zone, and interpret the reset in that
  zone rather than local.
- Honour a `retry-after` header from the API `claude` provider the same way,
  so an authoritative wait there also escapes the 5-minute cap.
- Surface an active pause in the web UI (e.g. "paused until 21:05") so an
  operator sees why a review is idle rather than assuming it hung.
