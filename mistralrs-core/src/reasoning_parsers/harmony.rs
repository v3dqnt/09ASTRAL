//! OpenAI Harmony format parsing for GPT-OSS models.
//!
//! The Harmony format uses channels to separate different types of content:
//! - `analysis`: Chain-of-thought reasoning (internal, not for end users)
//! - `commentary`: Tool call preambles and explanations
//! - `final`: User-facing response content
//!
//! Tool calls in Harmony are indicated by the `recipient` field being set to:
//! - `functions.tool_name` for user-defined tools
//! - `browser.search`, `browser.open`, `browser.find` for browser tool
//! - `python` for python tool
//!
//! This module provides incremental parsing of Harmony-formatted token streams.

use uuid::Uuid;

const START: &str = "<|start|>";
const CHANNEL: &str = "<|channel|>";
const CONSTRAIN: &str = "<|constrain|>";
const MESSAGE: &str = "<|message|>";
const STOPS: [&str; 3] = ["<|end|>", "<|call|>", "<|return|>"];
const ROLES: [&str; 5] = ["system", "developer", "user", "assistant", "tool"];

#[derive(Debug, Default, Clone, PartialEq)]
struct Header {
    recipient: Option<String>,
    channel: Option<String>,
}

#[derive(Debug)]
enum ParseState {
    ExpectStart,
    Header { text: String, role_known: bool },
    Content { header: Header, bytes: Vec<u8> },
}

/// Streaming Harmony parser over the detokenized output, where real special tokens appear as their text and
/// spelled-out ones were already defused by `SpecialTextGuard`. Mirrors openai-harmony's `StreamableParser`.
struct StreamParser {
    state: ParseState,
    pending: String,
}

impl StreamParser {
    fn new() -> Self {
        Self {
            state: ParseState::Header {
                text: String::new(),
                role_known: true,
            },
            pending: String::new(),
        }
    }

    fn process(&mut self, bytes: &[u8]) {
        if let ParseState::Content { bytes: content, .. } = &mut self.state {
            if !STOPS.iter().any(|m| contains(bytes, m.as_bytes())) {
                content.extend_from_slice(bytes);
                return;
            }
        }
        self.pending.push_str(&String::from_utf8_lossy(bytes));
        loop {
            match &mut self.state {
                ParseState::ExpectStart => match self.pending.find(START) {
                    Some(i) => {
                        self.pending.drain(..i + START.len());
                        self.state = ParseState::Header {
                            text: String::new(),
                            role_known: false,
                        };
                    }
                    None => {
                        self.pending.clear();
                        return;
                    }
                },
                ParseState::Header { text, role_known } => {
                    let msg = self.pending.find(MESSAGE);
                    let stop = first_stop(&self.pending);
                    match (msg, stop) {
                        (Some(i), s) if s.is_none_or(|(j, _)| i < j) => {
                            text.push_str(&self.pending[..i]);
                            let header = parse_header(text, *role_known);
                            self.pending.drain(..i + MESSAGE.len());
                            self.state = match header {
                                Some(header) => ParseState::Content {
                                    header,
                                    bytes: Vec::new(),
                                },
                                None => ParseState::ExpectStart,
                            };
                        }
                        (_, Some((j, len))) => {
                            self.pending.drain(..j + len);
                            self.state = ParseState::ExpectStart;
                        }
                        _ => {
                            text.push_str(&self.pending);
                            self.pending.clear();
                            return;
                        }
                    }
                }
                ParseState::Content { bytes, .. } => match first_stop(&self.pending) {
                    Some((j, len)) => {
                        bytes.extend_from_slice(&self.pending.as_bytes()[..j]);
                        self.pending.drain(..j + len);
                        self.state = ParseState::ExpectStart;
                    }
                    None => {
                        bytes.extend_from_slice(self.pending.as_bytes());
                        self.pending.clear();
                        return;
                    }
                },
            }
        }
    }

    fn process_eos(&mut self) {
        if matches!(self.state, ParseState::Content { .. }) {
            self.state = ParseState::ExpectStart;
        }
    }

    fn header(&self) -> Option<&Header> {
        match &self.state {
            ParseState::Content { header, .. } => Some(header),
            _ => None,
        }
    }

    fn current_channel(&self) -> Option<String> {
        self.header()?.channel.clone()
    }

    fn current_recipient(&self) -> Option<String> {
        self.header()?.recipient.clone()
    }

    /// Content so far, holding back an incomplete trailing UTF-8 sequence.
    fn current_content(&self) -> String {
        let ParseState::Content { bytes, .. } = &self.state else {
            return String::new();
        };
        let end = match std::str::from_utf8(bytes) {
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            _ => bytes.len(),
        };
        String::from_utf8_lossy(&bytes[..end]).into_owned()
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn first_stop(s: &str) -> Option<(usize, usize)> {
    STOPS
        .iter()
        .filter_map(|m| s.find(m).map(|i| (i, m.len())))
        .min()
}

/// Port of openai-harmony's `parse_header_from_string`; `None` means a malformed header.
fn parse_header(header: &str, role_known: bool) -> Option<Header> {
    let mut header = header.to_string();
    let mut channel = None;
    if let Some(idx) = header.find(CHANNEL) {
        let after = &header[idx + CHANNEL.len()..];
        let end = after
            .find(|c: char| c.is_whitespace() || c == '<')
            .unwrap_or(after.len());
        if end == 0 {
            return None;
        }
        channel = Some(after[..end].to_string());
        header = format!("{}{}", &header[..idx], &after[end..]);
    }
    let header = header.trim().replace(CONSTRAIN, &format!(" {CONSTRAIN}"));
    let mut parts: Vec<&str> = header.split_ascii_whitespace().collect();

    let role = if role_known {
        "assistant"
    } else {
        let first = *parts.first()?;
        if ROLES.contains(&first) {
            first
        } else if parts.len() > 1 || first.starts_with("to=") {
            parts.remove(0);
            "tool"
        } else {
            return None;
        }
    };
    if parts.first() == Some(&role) {
        parts.remove(0);
    }

    let mut recipient = None;
    if let Some(last) = parts.pop() {
        if let Some(r) = last.strip_prefix("to=") {
            recipient = Some(r.to_string());
        } else if parts.is_empty() {
            recipient = Some(last.to_string());
        } else if let Some(r) = parts.pop() {
            recipient = Some(r.strip_prefix("to=").unwrap_or(r).to_string());
        }
    }
    parts.is_empty().then_some(Header { recipient, channel })
}

/// Extract the tool name from a recipient string.
/// - "functions.my_tool" -> "my_tool"
/// - "browser.search" -> "browser.search"
/// - "python" -> "python"
fn extract_tool_name(recipient: &str) -> String {
    if let Some(name) = recipient.strip_prefix("functions.") {
        name.to_string()
    } else {
        // For builtin tools like "browser.search" or "python", use the full recipient
        recipient.to_string()
    }
}

/// Channel types in Harmony format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonyChannel {
    /// Chain-of-thought reasoning (internal, not for end users)
    Analysis,
    /// Tool call preambles and explanations
    Commentary,
    /// User-facing response content
    Final,
}

impl HarmonyChannel {
    /// Parse a channel name string into a HarmonyChannel
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "analysis" => Some(Self::Analysis),
            "commentary" => Some(Self::Commentary),
            "final" => Some(Self::Final),
            _ => None,
        }
    }
}

/// Incremental delta from Harmony parsing
#[derive(Debug, Clone, Default)]
pub struct HarmonyDelta {
    /// New analysis/reasoning content since last delta
    pub analysis_delta: Option<String>,
    /// New commentary content since last delta
    pub commentary_delta: Option<String>,
    /// New final response content since last delta
    pub final_delta: Option<String>,
    /// Currently active channel
    pub current_channel: Option<HarmonyChannel>,
}

impl HarmonyDelta {
    /// Check if this delta has any content
    pub fn has_content(&self) -> bool {
        self.analysis_delta.is_some()
            || self.commentary_delta.is_some()
            || self.final_delta.is_some()
    }

    /// Get reasoning content (analysis + commentary without tool calls)
    pub fn reasoning_content(&self) -> Option<String> {
        match (&self.analysis_delta, &self.commentary_delta) {
            (Some(a), Some(c)) => Some(format!("{}{}", a, c)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(c)) => Some(c.clone()),
            (None, None) => None,
        }
    }
}

/// Accumulated content for each channel
#[derive(Debug, Clone, Default)]
pub struct HarmonyAccumulated {
    /// Accumulated analysis content
    pub analysis: String,
    /// Accumulated commentary content
    pub commentary: String,
    /// Accumulated final content
    pub final_content: String,
}

/// A tool call parsed from Harmony format
#[derive(Debug, Clone)]
pub struct HarmonyToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// The function name (extracted from recipient like "functions.tool_name")
    pub name: String,
    /// The JSON arguments as a string
    pub arguments: String,
}

impl HarmonyAccumulated {
    /// Get all reasoning content (analysis + commentary)
    pub fn reasoning_content(&self) -> Option<String> {
        let combined = format!("{}{}", self.analysis, self.commentary);
        if combined.is_empty() {
            None
        } else {
            Some(combined)
        }
    }
}

/// Context for tracking Harmony parsing state within a sequence.
///
/// Wraps the streaming parser and provides delta extraction for streaming responses.
pub struct HarmonyContext {
    parser: StreamParser,
    // Track lengths for delta extraction (for parser content)
    last_analysis_len: usize,
    last_commentary_len: usize,
    last_final_len: usize,
    // Accumulated content
    accumulated: HarmonyAccumulated,
    // Track which channel we're currently in
    channel: Option<HarmonyChannel>,
    // Track positions for streaming deltas (what has been sent)
    sent_reasoning_len: usize,
    sent_final_len: usize,
    // Tool call tracking
    tool_calls: Vec<HarmonyToolCall>,
    // Track current tool call being built (recipient, accumulated_args)
    current_tool_call: Option<(String, String)>,
    // Track how much of current tool call args have been sent
    sent_tool_args_len: usize,
    // Set when a new tool call starts, signaling that a JSON grammar
    // should be activated for the argument tokens.
    needs_grammar_activation: bool,
}

impl Default for HarmonyContext {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyContext {
    /// Create a new Harmony parsing context
    pub fn new() -> Self {
        Self {
            parser: StreamParser::new(),
            last_analysis_len: 0,
            last_commentary_len: 0,
            last_final_len: 0,
            accumulated: HarmonyAccumulated::default(),
            channel: None,
            sent_reasoning_len: 0,
            sent_final_len: 0,
            tool_calls: Vec::new(),
            current_tool_call: None,
            sent_tool_args_len: 0,
            needs_grammar_activation: false,
        }
    }

    /// Process a token's detokenized bytes (special tokens included) and return any new delta content
    pub fn process_token(&mut self, bytes: &[u8]) -> HarmonyDelta {
        self.parser.process(bytes);
        self.extract_delta()
    }

    /// Extract delta since last call
    fn extract_delta(&mut self) -> HarmonyDelta {
        let mut delta = HarmonyDelta::default();

        // Get current channel from parser
        if let Some(channel_str) = self.parser.current_channel() {
            if let Some(channel) = HarmonyChannel::parse(&channel_str) {
                self.channel = Some(channel);
                delta.current_channel = Some(channel);
            }
        }

        // Check for tool calls via recipient field
        // Recipient is set to "functions.tool_name" when making a tool call
        let current_recipient = self.parser.current_recipient();

        // Get current content and extract delta based on channel
        {
            let content = self.parser.current_content();
            // Check if this is a tool call
            // Tool calls have recipients like:
            // - "functions.tool_name" for user-defined tools
            // - "browser.search", "browser.open", "browser.find" for browser tool
            // - "python" for python tool
            if let Some(ref recipient) = current_recipient {
                let is_tool_call = recipient.starts_with("functions.")
                    || recipient.starts_with("browser.")
                    || recipient == "python";

                if is_tool_call {
                    // This is a tool call - track it
                    // Check if this is the same tool call or a different one
                    let is_same_tool_call = self
                        .current_tool_call
                        .as_ref()
                        .is_some_and(|(existing, _)| existing == recipient);

                    if is_same_tool_call {
                        // Same tool call, update arguments
                        if let Some((_, ref mut args)) = self.current_tool_call {
                            *args = content.clone();
                        }
                    } else {
                        // Different tool call or no current tool call
                        // Finalize previous tool call if any
                        if let Some((prev_recipient, prev_args)) = self.current_tool_call.take() {
                            let prev_name = extract_tool_name(&prev_recipient);
                            self.tool_calls.push(HarmonyToolCall {
                                id: format!("call_{}", Uuid::new_v4()),
                                name: prev_name,
                                arguments: prev_args,
                            });
                        }
                        // Start new tool call
                        self.current_tool_call = Some((recipient.clone(), content.clone()));
                        self.sent_tool_args_len = 0;
                        self.needs_grammar_activation = true;
                    }
                    // Don't accumulate tool call content to final_content
                    return delta;
                }
            }

            // Not a tool call, handle normally by channel
            match self.channel {
                Some(HarmonyChannel::Analysis) => {
                    if content.len() > self.last_analysis_len {
                        let new_content = content[self.last_analysis_len..].to_string();
                        self.accumulated.analysis.push_str(&new_content);
                        delta.analysis_delta = Some(new_content);
                        self.last_analysis_len = content.len();
                    }
                }
                Some(HarmonyChannel::Commentary) => {
                    if content.len() > self.last_commentary_len {
                        let new_content = content[self.last_commentary_len..].to_string();
                        self.accumulated.commentary.push_str(&new_content);
                        delta.commentary_delta = Some(new_content);
                        self.last_commentary_len = content.len();
                    }
                }
                Some(HarmonyChannel::Final) | None => {
                    // Final channel OR no channel marker - treat content as final.
                    // This handles cases where the model responds without Harmony
                    // channel markers (e.g., after tool call results).
                    if content.len() > self.last_final_len {
                        let new_content = content[self.last_final_len..].to_string();
                        self.accumulated.final_content.push_str(&new_content);
                        delta.final_delta = Some(new_content);
                        self.last_final_len = content.len();
                    }
                }
            }
        }

        delta
    }

    /// Check if grammar activation is needed for a new tool call,
    /// clearing the flag after reading.
    pub fn take_needs_grammar_activation(&mut self) -> bool {
        std::mem::replace(&mut self.needs_grammar_activation, false)
    }

    /// Get the currently active channel
    pub fn current_channel(&self) -> Option<HarmonyChannel> {
        self.channel
    }

    /// Get all accumulated content
    pub fn accumulated(&self) -> &HarmonyAccumulated {
        &self.accumulated
    }

    /// Get accumulated reasoning content (analysis + commentary)
    pub fn reasoning_content(&self) -> Option<String> {
        self.accumulated.reasoning_content()
    }

    /// Get accumulated final content
    pub fn final_content(&self) -> Option<String> {
        if self.accumulated.final_content.is_empty() {
            None
        } else {
            Some(self.accumulated.final_content.clone())
        }
    }

    /// Get the reasoning delta since last call (for streaming).
    /// Returns new reasoning content that hasn't been sent yet.
    pub fn get_reasoning_delta(&mut self) -> Option<String> {
        let reasoning = format!(
            "{}{}",
            self.accumulated.analysis, self.accumulated.commentary
        );
        if reasoning.len() > self.sent_reasoning_len {
            let delta = reasoning[self.sent_reasoning_len..].to_string();
            self.sent_reasoning_len = reasoning.len();
            if delta.is_empty() {
                None
            } else {
                Some(delta)
            }
        } else {
            None
        }
    }

    /// Get the final content delta since last call (for streaming).
    /// Returns new final content that hasn't been sent yet.
    pub fn get_final_delta(&mut self) -> Option<String> {
        if self.accumulated.final_content.len() > self.sent_final_len {
            let delta = self.accumulated.final_content[self.sent_final_len..].to_string();
            self.sent_final_len = self.accumulated.final_content.len();
            if delta.is_empty() {
                None
            } else {
                Some(delta)
            }
        } else {
            None
        }
    }

    /// Signal end of stream to the parser
    pub fn process_eos(&mut self) {
        self.parser.process_eos();

        // Finalize any pending tool call
        if let Some((recipient, args)) = self.current_tool_call.take() {
            let name = extract_tool_name(&recipient);
            self.tool_calls.push(HarmonyToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name,
                arguments: args,
            });
        }
    }

    /// Get the recipient (for tool calls) if any
    pub fn current_recipient(&self) -> Option<String> {
        self.parser.current_recipient()
    }

    /// Check if there's a tool call in progress
    pub fn has_tool_call(&self) -> bool {
        self.current_tool_call.is_some() || !self.tool_calls.is_empty()
    }

    /// Get all completed tool calls
    pub fn get_tool_calls(&self) -> &[HarmonyToolCall] {
        &self.tool_calls
    }

    /// Get the current tool call being built (if any)
    /// Returns (recipient, arguments_so_far)
    pub fn get_current_tool_call(&self) -> Option<(&str, &str)> {
        self.current_tool_call
            .as_ref()
            .map(|(recipient, args)| (recipient.as_str(), args.as_str()))
    }

    /// Finalize any pending tool call and return all tool calls.
    /// This should be called when the sequence is done.
    /// Note: This takes ownership of the tool calls, so calling it twice
    /// will return an empty vector the second time.
    pub fn finalize_tool_calls(&mut self) -> Vec<HarmonyToolCall> {
        // Finalize any pending tool call
        if let Some((recipient, args)) = self.current_tool_call.take() {
            let name = extract_tool_name(&recipient);
            self.tool_calls.push(HarmonyToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name,
                arguments: args,
            });
        }
        // Take ownership to prevent duplicate returns if called multiple times
        std::mem::take(&mut self.tool_calls)
    }
}

/// Check if a chat template uses Harmony format by looking for Harmony markers.
///
/// Returns true if the template contains Harmony-specific tokens like
/// `<|channel|>`, `<|start|>`, `<|message|>`, or `<|end|>`.
pub fn is_harmony_template(template: &str) -> bool {
    // Check for the most distinctive Harmony marker
    if template.contains("<|channel|>") {
        return true;
    }

    // Check for the combination of start/message/end which is characteristic of Harmony
    template.contains("<|start|>")
        && template.contains("<|message|>")
        && template.contains("<|end|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(ctx: &mut HarmonyContext, pieces: &[&str]) {
        for p in pieces {
            ctx.process_token(p.as_bytes());
        }
    }

    #[test]
    fn test_parse_header() {
        let h = |s, known| parse_header(s, known);
        assert_eq!(
            h("<|channel|>analysis", true),
            Some(Header {
                recipient: None,
                channel: Some("analysis".into())
            })
        );
        assert_eq!(
            h(
                " to=functions.get_weather<|channel|>commentary <|constrain|>json",
                true
            ),
            Some(Header {
                recipient: Some("functions.get_weather".into()),
                channel: Some("commentary".into())
            })
        );
        assert_eq!(
            h("assistant<|channel|>commentary to=python code", false),
            Some(Header {
                recipient: Some("python".into()),
                channel: Some("commentary".into())
            })
        );
        assert_eq!(
            h("functions.x to=assistant<|channel|>commentary", false),
            Some(Header {
                recipient: Some("assistant".into()),
                channel: Some("commentary".into())
            })
        );
        assert_eq!(h("bogus", false), None);
        assert_eq!(h("<|channel|>", true), None);
        assert_eq!(h("a b c", true), None);
    }

    #[test]
    fn test_stream_reasoning_then_final() {
        let mut ctx = HarmonyContext::new();
        let mut reasoning = String::new();
        let mut fin = String::new();
        for p in [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "Think",
            "ing.",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "Hi",
            "E9",
            "<|return|>",
        ] {
            let bytes: Vec<u8> = if p == "E9" {
                vec![b' ', 0xc3, 0xa9]
            } else {
                p.into()
            };
            let d = ctx.process_token(&bytes);
            reasoning.push_str(&d.analysis_delta.unwrap_or_default());
            fin.push_str(&d.final_delta.unwrap_or_default());
        }
        assert_eq!(reasoning, "Thinking.");
        assert_eq!(fin, "Hi \u{e9}");
        assert!(!ctx.has_tool_call());
    }

    #[test]
    fn test_stream_split_utf8() {
        let mut p = StreamParser::new();
        p.process(b"<|channel|>final<|message|>a");
        p.process(&[0xe2, 0x82]);
        assert_eq!(p.current_content(), "a");
        p.process(&[0xac]);
        assert_eq!(p.current_content(), "a\u{20ac}");
    }

    #[test]
    fn test_stream_tool_call() {
        let mut ctx = HarmonyContext::new();
        feed(
            &mut ctx,
            &[
                "<|channel|>analysis<|message|>need weather",
                "<|end|>",
                "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>",
                "{\"city\":",
                "\"Paris\"}",
                "<|call|>",
            ],
        );
        let calls = ctx.finalize_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"Paris\"}");
    }

    #[test]
    fn test_is_harmony_template() {
        // Should detect Harmony templates
        assert!(is_harmony_template(
            "<|start|>system<|message|>content<|end|>"
        ));
        assert!(is_harmony_template(
            "some prefix <|channel|>analysis<|message|>thinking"
        ));

        // Should not detect non-Harmony templates
        assert!(!is_harmony_template("<|im_start|>system<|im_end|>"));
        assert!(!is_harmony_template("regular chat template"));
    }

    #[test]
    fn test_harmony_channel_from_str() {
        assert_eq!(
            HarmonyChannel::parse("analysis"),
            Some(HarmonyChannel::Analysis)
        );
        assert_eq!(
            HarmonyChannel::parse("commentary"),
            Some(HarmonyChannel::Commentary)
        );
        assert_eq!(HarmonyChannel::parse("final"), Some(HarmonyChannel::Final));
        assert_eq!(HarmonyChannel::parse("unknown"), None);
    }

    #[test]
    fn test_harmony_delta_has_content() {
        let empty = HarmonyDelta::default();
        assert!(!empty.has_content());

        let with_analysis = HarmonyDelta {
            analysis_delta: Some("thinking".to_string()),
            ..Default::default()
        };
        assert!(with_analysis.has_content());

        let with_final = HarmonyDelta {
            final_delta: Some("response".to_string()),
            ..Default::default()
        };
        assert!(with_final.has_content());
    }

    #[test]
    fn test_harmony_delta_reasoning_content() {
        let both = HarmonyDelta {
            analysis_delta: Some("thinking ".to_string()),
            commentary_delta: Some("about tools".to_string()),
            ..Default::default()
        };
        assert_eq!(
            both.reasoning_content(),
            Some("thinking about tools".to_string())
        );

        let only_analysis = HarmonyDelta {
            analysis_delta: Some("just thinking".to_string()),
            ..Default::default()
        };
        assert_eq!(
            only_analysis.reasoning_content(),
            Some("just thinking".to_string())
        );

        let none = HarmonyDelta::default();
        assert_eq!(none.reasoning_content(), None);
    }
}
