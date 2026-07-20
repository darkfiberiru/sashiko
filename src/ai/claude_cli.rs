// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! AI provider that shells out to the `claude` CLI instead of calling the API directly.
//! This uses the local Claude Code installation (subscription auth) rather than API credits.
//!
//! ## Safety
//!
//! The `claude --print` flag runs in text-completion mode: no tools, no file
//! access, no session persistence, no network calls. The CLI reads a prompt
//! from stdin and writes a response to stdout — it cannot modify the
//! filesystem or execute commands. This makes it inherently safe for use as
//! a completion backend without any additional sandboxing.

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Local, TimeZone};
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

use crate::ai::{
    AiErrorClass, AiProvider, AiRequest, AiResponse, AiRole, AiUsage, ClassifyAiError,
    ProviderCapabilities, ToolCall,
};

#[derive(Debug, thiserror::Error)]
pub enum ClaudeCliError {
    #[error("Failed to spawn claude CLI: {0}")]
    Spawn(String),
    #[error("claude CLI timed out after 10 minutes")]
    Timeout,
    #[error("claude CLI wait error: {0}")]
    Wait(String),
    #[error("claude CLI error: {0}")]
    Cli(String),
    #[error("Failed to parse claude CLI JSON output: {0}")]
    Parse(String),
}

impl ClassifyAiError for ClaudeCliError {
    fn ai_error_class(&self) -> AiErrorClass {
        match self {
            ClaudeCliError::Spawn(_) => AiErrorClass::Fatal,
            ClaudeCliError::Timeout => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            ClaudeCliError::Wait(_) => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            ClaudeCliError::Cli(msg) => classify_cli_message(msg, Local::now()),
            ClaudeCliError::Parse(_) => AiErrorClass::Fatal,
        }
    }
}

/// Buffer added past a session-limit reset time before retrying, so we resume
/// just after access returns rather than racing the reset to the second.
const SESSION_RESET_BUFFER: Duration = Duration::from_secs(5 * 60);

/// Fallback pause when a session limit is reported but its reset time cannot be
/// parsed. Long enough to avoid hammering the CLI, short enough to recover if
/// the guess is wrong.
const SESSION_RESET_FALLBACK: Duration = Duration::from_secs(30 * 60);

/// Grace window past a reported reset within which a repeated session-limit
/// message is treated as a *late* reset (retry soon) rather than the next day's
/// cycle. See [`parse_session_reset`] for the rationale.
const SESSION_RESET_GRACE_MINUTES: i64 = 30;

/// Classifies a `claude` CLI error message.
///
/// The CLI reports two recoverable conditions as plain text that would
/// otherwise be treated as fatal:
///   - a subscription session limit ("You've hit your session limit · resets
///     9pm (America/New_York)"), which clears at a known wall-clock time;
///   - a transient overload ("API Error: Overloaded", i.e. an HTTP 529).
///
/// Everything else remains fatal.
///
/// `now` is injected so the reset-time arithmetic is testable. The reset time
/// is interpreted in the machine's local timezone; the CLI currently reports
/// times in the local zone, so no zone conversion is done here (see
/// DESIGN_TRANSIENT_ERROR_HANDLING.md).
fn classify_cli_message(msg: &str, now: DateTime<Local>) -> AiErrorClass {
    let lower = msg.to_lowercase();

    if lower.contains("session limit") {
        let retry_after = parse_session_reset(&lower, now)
            .map(|d| d + SESSION_RESET_BUFFER)
            .unwrap_or(SESSION_RESET_FALLBACK);
        return AiErrorClass::SessionLimit { retry_after };
    }

    if lower.contains("overloaded") || lower.contains("529") {
        return AiErrorClass::Transient {
            retry_after: Duration::from_secs(30),
        };
    }

    AiErrorClass::Fatal
}

/// Extracts the duration until we should next retry after a "resets <time>"
/// message.
///
/// Handles `9pm`, `10:30am`, `12am` (midnight) and `12pm` (noon). Behaviour by
/// where the reset time falls relative to `now`:
///   - still ahead today  -> wait until it (the exact reset);
///   - just passed (within [`SESSION_RESET_GRACE_MINUTES`]) -> `ZERO`, so the
///     caller's buffer yields a ~5-minute poll. This is the "hit the timer,
///     still limited" case: the reset is late, so retry soon rather than waiting
///     a whole day;
///   - well in the past    -> the same time tomorrow (the next cycle).
///
/// Returns `None` if no reset time is present or it cannot be parsed.
fn parse_session_reset(lower_msg: &str, now: DateTime<Local>) -> Option<Duration> {
    let re = regex::Regex::new(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)").ok()?;
    let caps = re.captures(lower_msg)?;

    let hour12: u32 = caps.get(1)?.as_str().parse().ok()?;
    if !(1..=12).contains(&hour12) {
        return None;
    }
    let minute: u32 = match caps.get(2) {
        Some(m) => m.as_str().parse().ok()?,
        None => 0,
    };
    if minute > 59 {
        return None;
    }
    let is_pm = caps.get(3)?.as_str() == "pm";

    // 12am -> 0, 12pm -> 12, otherwise add 12 for pm.
    let hour24 = match (hour12, is_pm) {
        (12, false) => 0,
        (12, true) => 12,
        (h, false) => h,
        (h, true) => h + 12,
    };

    let today = now.date_naive().and_hms_opt(hour24, minute, 0)?;
    match resolve_local(today) {
        // Still ahead today: wait until it.
        Some(dt) if dt > now => (dt - now).to_std().ok(),
        // Only just passed: the reset is running late, so retry soon.
        Some(dt) if (now - dt).num_minutes() <= SESSION_RESET_GRACE_MINUTES => Some(Duration::ZERO),
        // Well in the past (or a DST gap today): next day's cycle.
        _ => {
            let tomorrow = now
                .date_naive()
                .succ_opt()?
                .and_hms_opt(hour24, minute, 0)?;
            (resolve_local(tomorrow)? - now).to_std().ok()
        }
    }
}

/// Resolves a naive local datetime to a concrete instant, choosing the earliest
/// valid one if the wall-clock time is ambiguous (fall-back DST hour). Returns
/// `None` if the time does not exist (spring-forward gap); the caller then uses
/// a safe fallback pause.
fn resolve_local(naive: chrono::NaiveDateTime) -> Option<DateTime<Local>> {
    use chrono::offset::LocalResult;
    match Local.from_local_datetime(&naive) {
        LocalResult::Single(dt) => Some(dt),
        LocalResult::Ambiguous(earliest, _) => Some(earliest),
        LocalResult::None => None,
    }
}

pub struct ClaudeCliProvider {
    pub model: String,
    pub effort: Option<String>,
}

#[async_trait]
impl AiProvider for ClaudeCliProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = build_prompt(&request);

        debug!("claude-cli prompt length: {} chars", prompt.len());

        let mut args = vec![
            "--print".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
            "--no-session-persistence".to_string(),
        ];

        args.push("--model".to_string());
        args.push(self.model.clone());

        if let Some(effort) = &self.effort {
            args.push("--effort".to_string());
            args.push(effort.clone());
        }

        let mut child = Command::new("claude")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| ClaudeCliError::Spawn(e.to_string()))?;

        // Write prompt to stdin then close it
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
            stdin.flush().await?;
        }

        // 10-minute timeout per CLI call — a hung claude process won't block forever
        let output = timeout(Duration::from_secs(600), child.wait_with_output())
            .await
            .map_err(|_| ClaudeCliError::Timeout)?
            .map_err(|e| ClaudeCliError::Wait(e.to_string()))?;

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                if !line.trim().is_empty() {
                    debug!("[claude-cli stderr] {}", line);
                }
            }
        }

        let raw = String::from_utf8_lossy(&output.stdout);

        if !output.status.success() {
            // Try to extract the actual error message from the JSON output.
            // The CLI emits a JSON object with is_error=true and the reason
            // in the "result" field even when it exits non-zero.
            if let Ok(outer) = serde_json::from_str::<Value>(&raw)
                && let Some(msg) = outer["result"].as_str()
            {
                return Err(ClaudeCliError::Cli(msg.to_string()).into());
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ClaudeCliError::Cli(format!(
                "exited with {}: {}",
                output.status,
                stderr.trim()
            ))
            .into());
        }
        let outer: Value = serde_json::from_str(&raw).map_err(|e| {
            ClaudeCliError::Parse(format!("{}\nRaw: {}", e, &raw[..raw.len().min(200)]))
        })?;

        if outer["is_error"].as_bool().unwrap_or(false) {
            return Err(ClaudeCliError::Cli(
                outer["result"]
                    .as_str()
                    .unwrap_or("unknown error")
                    .to_string(),
            )
            .into());
        }

        let result_text = outer["result"].as_str().unwrap_or("").trim().to_string();

        // Parse usage from the outer JSON
        let usage = parse_usage(&outer);

        // Parse the inner response — tool calls or content
        parse_inner_response(&result_text, usage)
    }

    fn estimate_tokens(&self, request: &AiRequest) -> usize {
        let chars: usize = request
            .messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|c| c.len())
            .sum();
        chars / 4
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: self.model.clone(),
            context_window_size: context_window_for_model(&self.model),
        }
    }
}

/// Pick the context window to advertise for a given model name. The value is
/// currently metadata only (no consumer gates on it), so the mapping is coarse.
/// Opus 4.7 ships with a 1M window by default via Claude Code, and any model
/// can be selected with the `[1m]` suffix to opt into the 1M variant.
/// Verified against Claude Code 2.1.132 for opus-4-7, sonnet-4-6, sonnet-4-6[1m],
/// haiku-4-5.
fn context_window_for_model(model: &str) -> usize {
    if model.contains("[1m]") || model.contains("opus-4-7") {
        1_000_000
    } else {
        200_000
    }
}

/// Build the full text prompt from the AiRequest.
/// Embeds system prompt, conversation history, tool definitions, and instructions.
pub fn build_prompt(request: &AiRequest) -> String {
    let mut out = String::new();

    // System prompt
    if let Some(sys) = &request.system {
        out.push_str("<system>\n");
        out.push_str(sys);
        out.push_str("\n</system>\n\n");
    }

    // Conversation history
    for msg in &request.messages {
        match &msg.role {
            AiRole::System => {
                // Already handled above; skip embedded system messages
            }
            AiRole::User => {
                out.push_str("<user>\n");
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                out.push_str("\n</user>\n\n");
            }
            AiRole::Assistant => {
                out.push_str("<assistant>\n");
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        out.push_str(&format!(
                            "<tool_call id=\"{}\" name=\"{}\">\n{}\n</tool_call>\n",
                            call.id, call.function_name, call.arguments
                        ));
                    }
                }
                out.push_str("</assistant>\n\n");
            }
            AiRole::Tool => {
                let id = msg.tool_call_id.as_deref().unwrap_or("?");
                out.push_str(&format!("<tool_result id=\"{}\">\n", id));
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                out.push_str("\n</tool_result>\n\n");
            }
        }
    }

    // Tool definitions and response instructions
    if let Some(tools) = &request.tools
        && !tools.is_empty()
    {
        out.push_str("<available_tools>\n");
        for tool in tools {
            out.push_str(&format!(
                "- name: {}\n  description: {}\n  parameters: {}\n\n",
                tool.name, tool.description, tool.parameters
            ));
        }
        out.push_str("</available_tools>\n\n");
        out.push_str(
            "RESPONSE FORMAT: You MUST respond with a SINGLE valid JSON object only (no markdown, no explanation).\n\
             To call tools: {\"tool_calls\": [{\"id\": \"c1\", \"function_name\": \"TOOL_NAME\", \"arguments\": {ARGS}}, {\"id\": \"c2\", \"function_name\": \"OTHER_TOOL\", \"arguments\": {ARGS2}}]}\n\
             Put ALL tool calls in ONE tool_calls array. Do NOT output multiple JSON objects.\n\
             For your final answer: {\"content\": \"YOUR RESPONSE\"}\n\
             Do not mix both. Output exactly one JSON object.\n",
        );
    } else if let Some(instruction) = request
        .response_format
        .as_ref()
        .and_then(|f| f.format_json_schema_instruction())
    {
        out.push_str(&instruction);
        out.push('\n');
    }

    out
}

fn parse_usage(outer: &Value) -> Option<AiUsage> {
    let u = &outer["usage"];
    if u.is_null() {
        return None;
    }
    let input = u["input_tokens"].as_u64().unwrap_or(0) as usize;
    let output = u["output_tokens"].as_u64().unwrap_or(0) as usize;
    let cached = u["cache_read_input_tokens"].as_u64().unwrap_or(0) as usize;
    Some(AiUsage {
        prompt_tokens: input,
        completion_tokens: output,
        total_tokens: input + output,
        cached_tokens: Some(cached),
    })
}

pub fn parse_inner_response(text: &str, usage: Option<AiUsage>) -> Result<AiResponse> {
    // Try extracting JSON (might be in a markdown code block)
    let json_str = extract_json(text);

    if let Ok(v) = serde_json::from_str::<Value>(&json_str) {
        return parse_single_json(&v, &json_str, usage);
    }

    // Try JSONL: multiple JSON objects on separate lines (model sometimes emits
    // separate tool_calls objects per line instead of one combined object)
    let mut merged_tool_calls: Vec<ToolCall> = Vec::new();
    let mut had_json = false;
    for line in json_str.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            had_json = true;
            if let Some(calls) = v["tool_calls"].as_array() {
                for c in calls {
                    if let Some(tc) = parse_tool_call(c) {
                        merged_tool_calls.push(tc);
                    }
                }
            }
        }
    }

    if !merged_tool_calls.is_empty() {
        debug!(
            "claude-cli: merged {} tool calls from JSONL response",
            merged_tool_calls.len()
        );
        return Ok(AiResponse {
            content: None,
            thought: None,
            thought_signature: None,
            tool_calls: Some(merged_tool_calls),
            usage,
            truncated: false,
        });
    }

    if had_json {
        // Had valid JSON lines but no tool calls — return original text as content
        // (json_str from extract_json may be mangled if text had multiple objects)
        return Ok(AiResponse {
            content: Some(text.to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            usage,
            truncated: false,
        });
    }

    // Not parseable as JSON — return raw text
    warn!("claude-cli response not valid JSON, returning as raw content");
    Ok(AiResponse {
        content: Some(text.to_string()),
        thought: None,
        thought_signature: None,
        tool_calls: None,
        usage,
        truncated: false,
    })
}

fn parse_tool_call(c: &Value) -> Option<ToolCall> {
    let id = c["id"].as_str().unwrap_or("c1").to_string();
    let name = c["function_name"].as_str()?.to_string();
    let args = c["arguments"].clone();
    Some(ToolCall {
        id,
        function_name: name,
        arguments: args,
        thought_signature: None,
    })
}

fn parse_single_json(v: &Value, json_str: &str, usage: Option<AiUsage>) -> Result<AiResponse> {
    // Tool calls?
    if let Some(calls) = v["tool_calls"].as_array() {
        let tool_calls: Vec<ToolCall> = calls.iter().filter_map(parse_tool_call).collect();

        if !tool_calls.is_empty() {
            return Ok(AiResponse {
                content: None,
                thought: None,
                thought_signature: None,
                tool_calls: Some(tool_calls),
                usage,
                truncated: false,
            });
        }
    }

    // Content field?
    if let Some(content) = v["content"].as_str() {
        return Ok(AiResponse {
            content: Some(content.to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            usage,
            truncated: false,
        });
    }

    // Any other JSON — return it as content string (e.g. {"concerns": [...]})
    Ok(AiResponse {
        content: Some(json_str.to_string()),
        thought: None,
        thought_signature: None,
        tool_calls: None,
        usage,
        truncated: false,
    })
}

/// Extract JSON from text that may be wrapped in markdown fences.
/// Returns the content inside the first fenced block, or the original text trimmed.
/// Does NOT try to find outermost braces — that can silently produce invalid JSON
/// when the text contains multiple objects (e.g. JSONL), which the JSONL fallback
/// in parse_inner_response handles better.
fn extract_json(text: &str) -> String {
    // Strip markdown fences — handle both LF and CRLF, and optional language tag
    let normalized = text.replace("\r\n", "\n");
    for fence_start in &["```json\n", "```JSON\n", "```\n"] {
        if let Some(start) = normalized.find(fence_start) {
            let after = &normalized[start + fence_start.len()..];
            if let Some(end) = after.find("\n```") {
                return after[..end].trim().to_string();
            }
        }
    }
    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiMessage, AiRequest, AiResponseFormat, AiRole, AiTool};
    use chrono::TimeZone;
    use serde_json::json;

    fn at(hour: u32, min: u32) -> DateTime<Local> {
        // A fixed reference day well clear of any DST boundary.
        Local
            .with_ymd_and_hms(2026, 6, 15, hour, min, 0)
            .single()
            .expect("valid local time")
    }

    #[test]
    fn overloaded_is_transient() {
        let c = classify_cli_message("API Error: Overloaded", at(15, 0));
        assert!(matches!(c, AiErrorClass::Transient { .. }));
    }

    #[test]
    fn http_529_is_transient() {
        let c = classify_cli_message("claude CLI error: 529 overloaded_error", at(15, 0));
        assert!(matches!(c, AiErrorClass::Transient { .. }));
    }

    #[test]
    fn unknown_cli_error_stays_fatal() {
        let c = classify_cli_message("some unrecognized failure", at(15, 0));
        assert!(matches!(c, AiErrorClass::Fatal));
    }

    #[test]
    fn session_limit_waits_until_reset_plus_buffer() {
        // At 3pm, "resets 9pm" -> 6h until reset, + 5 min buffer.
        let msg = "You've hit your session limit · resets 9pm (America/New_York)";
        let c = classify_cli_message(msg, at(15, 0));
        match c {
            AiErrorClass::SessionLimit { retry_after } => {
                let secs = retry_after.as_secs();
                assert_eq!(secs, 6 * 3600 + 5 * 60, "6h to 9pm plus 5 min buffer");
            }
            other => panic!("expected SessionLimit, got {:?}", other),
        }
    }

    #[test]
    fn session_reset_rolls_to_tomorrow_when_past() {
        // 10pm vs "resets 9pm": an hour past, beyond the grace window.
        let d = parse_session_reset("resets 9pm", at(22, 0)).expect("parsed");
        assert_eq!(d.as_secs(), 23 * 3600, "23h until tomorrow 9pm");
    }

    #[test]
    fn session_reset_just_passed_retries_soon() {
        // Reset only just passed -> ZERO; the caller's buffer makes it a poll.
        assert_eq!(
            parse_session_reset("resets 9pm", at(21, 5)),
            Some(Duration::ZERO),
            "5 min past reset should retry soon, not tomorrow"
        );
        // End to end: the classified pause is exactly the 5-minute buffer.
        match classify_cli_message("session limit · resets 9pm", at(21, 5)) {
            AiErrorClass::SessionLimit { retry_after } => {
                assert_eq!(retry_after, SESSION_RESET_BUFFER, "delayed reset -> 5 min");
            }
            other => panic!("expected SessionLimit, got {:?}", other),
        }
    }

    #[test]
    fn session_reset_grace_boundary() {
        // At exactly the grace edge (30 min past) it still counts as just-passed.
        assert_eq!(
            parse_session_reset("resets 9pm", at(21, 30)),
            Some(Duration::ZERO),
            "30 min past is within the grace window"
        );
        // Well beyond the grace window rolls to the next day.
        let d = parse_session_reset("resets 9pm", at(23, 0)).expect("parsed");
        assert_eq!(d.as_secs(), 22 * 3600, "2h past -> tomorrow 9pm (22h away)");
    }

    #[test]
    fn session_reset_handles_minutes_and_noon_midnight() {
        // 10:30am from 9am -> 1h30m.
        assert_eq!(
            parse_session_reset("resets 10:30am", at(9, 0))
                .unwrap()
                .as_secs(),
            90 * 60
        );
        // 12pm (noon) from 9am -> 3h.
        assert_eq!(
            parse_session_reset("resets 12pm", at(9, 0))
                .unwrap()
                .as_secs(),
            3 * 3600
        );
        // 12am (midnight) from 9pm -> 3h to next midnight.
        assert_eq!(
            parse_session_reset("resets 12am", at(21, 0))
                .unwrap()
                .as_secs(),
            3 * 3600
        );
    }

    #[test]
    fn session_limit_unparseable_time_uses_fallback() {
        // "session limit" present but no parseable reset -> fallback pause.
        let c = classify_cli_message("You've hit your session limit", at(15, 0));
        match c {
            AiErrorClass::SessionLimit { retry_after } => {
                assert_eq!(retry_after, SESSION_RESET_FALLBACK);
            }
            other => panic!("expected SessionLimit fallback, got {:?}", other),
        }
    }

    fn make_request(messages: Vec<AiMessage>) -> AiRequest {
        AiRequest {
            system: None,
            messages,
            tools: None,
            temperature: None,
            response_format: None,
            context_tag: None,
        }
    }

    fn simple_user_msg() -> Vec<AiMessage> {
        vec![AiMessage {
            role: AiRole::User,
            content: Some("hi".to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            tool_call_id: None,
        }]
    }

    #[test]
    fn test_build_prompt_json_format_without_tools() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Json { schema: None });

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("ONLY a valid JSON object"));
        assert!(!prompt.contains("tool_calls"));
    }

    #[test]
    fn test_build_prompt_json_format_with_schema() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Json {
            schema: Some(
                json!({"type": "object", "properties": {"selected_prompts": {"type": "array"}}}),
            ),
        });

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("selected_prompts"));
        assert!(prompt.contains("matching this schema"));
    }

    #[test]
    fn test_build_prompt_with_tools_includes_format() {
        let mut req = make_request(simple_user_msg());
        req.tools = Some(vec![AiTool {
            name: "git_log".to_string(),
            description: "Show git log".to_string(),
            parameters: json!({"type": "object"}),
        }]);

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("tool_calls"));
        assert!(prompt.contains("<available_tools>"));
    }

    #[test]
    fn test_build_prompt_text_format_no_instruction() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Text);

        let prompt = build_prompt(&req);
        assert!(!prompt.contains("RESPONSE FORMAT"));
    }

    #[test]
    fn test_build_prompt_no_format_no_instruction() {
        let req = make_request(simple_user_msg());

        let prompt = build_prompt(&req);
        assert!(!prompt.contains("RESPONSE FORMAT"));
    }
}
