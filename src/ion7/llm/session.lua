--- @module ion7.llm.session
--- @author  ion7 / Ion7 Project Contributors
---
--- Conversation state — message history plus the per-sequence KV
--- bookkeeping that lets multiple sessions share a single
--- `ion7.core.Context` instance.
---
--- Each `Session` owns :
---
---   - a `seq_id` — its row in the shared KV cache (assigned by the
---     `kv` layer at first prepare).
---   - a per-session `n_past` — the position of the next decode in
---     that row. Distinct from the context's global Lua-side mirror,
---     which is single-row by design.
---   - a `_seq_snapshot` — the blob returned by `ctx:seq_snapshot(seq_id)`
---     on the last successful decode. Restoring it via
---     `ctx:seq_restore(blob, seq_id)` brings the sequence back to that
---     exact KV state without touching any other session's row.
---
--- The class is deliberately small : it stores history and KV
--- coordinates, exposes message-construction helpers, and serialises
--- to JSON. Any logic that touches the model lives in `engine.lua` or
--- `kv/`.
---
---   local s = llm.Session.new({ system = "You are concise." })
---   s:add_user("What is LuaJIT ?")
---   local r = engine:chat(s)
---   s:add_assistant(r.content, { thinking = r.thinking,
---                                tool_calls = r.tool_calls })

local messages_util = require "ion7.llm.util.messages"
local json          = require "ion7.vendor.json"

local Session = {}
Session.__index = Session

local _next_id = 0
local function next_id()
    _next_id = _next_id + 1
    return _next_id
end

--- @class ion7.llm.Session
--- @field id              integer            Process-unique session id.
--- @field seq_id          integer?           KV row in the shared context.
--- @field system          string?            System prompt (single string).
--- @field messages        table[]            Canonical `{role, content, ...}` array.
--- @field n_past          integer            Per-session next-decode position.
--- @field _seq_snapshot   string?            Last `ctx:seq_snapshot(seq_id)` blob.
--- @field _dirty          boolean            Set when `messages` mutated since last decode.
--- @field _msg_kv_ends    integer[]          KV-end-position per message (for eviction).
--- @field _last_response  ion7.llm.Response? Set by `engine:chat` after each call.

--- New session.
---
--- @param  opts table?
---   `system` (string?)  System prompt. Rendered as the first message
---                       by the chat template ; not stored in
---                       `messages` directly.
---   `seq_id` (integer?) Pre-assigned KV row. Most callers leave this
---                       nil ; the `kv` layer claims a slot at first
---                       prepare.
--- @return ion7.llm.Session
function Session.new(opts)
    opts = opts or {}
    return setmetatable({
        id             = next_id(),
        seq_id         = opts.seq_id,
        system         = opts.system or nil,
        messages       = {},
        n_past         = 0,
        _seq_snapshot  = nil,
        _dirty         = false,
        _msg_kv_ends   = {},
        _last_response = nil,
    }, Session)
end

-- ── Message construction ──────────────────────────────────────────────────

--- Append a message of an explicit role. Marks the session dirty so
--- the engine knows to re-decode the suffix on next call.
---
--- @param  role    string  `"user" | "assistant" | "system" | "tool"`
--- @param  content string
--- @param  extras  table?  Pass-through fields (`thinking`,
---                         `tool_calls`, `tool_call_id`) understood by
---                         the chat template.
--- @return ion7.llm.Session self
function Session:add(role, content, extras)
    local msg = { role = role, content = content }
    if extras then
        if extras.thinking     then msg.thinking     = extras.thinking     end
        if extras.tool_calls   then msg.tool_calls   = extras.tool_calls   end
        if extras.tool_call_id then msg.tool_call_id = extras.tool_call_id end
    end
    messages_util.validate(msg)
    self.messages[#self.messages + 1] = msg
    self._dirty = true
    return self
end

--- Append a user message.
function Session:add_user(text)
    return self:add("user", text)
end

--- Append an assistant message. Pass `extras.thinking` and
--- `extras.tool_calls` from the previous response so the chat
--- template can re-render the full reasoning + tool-call envelope on
--- the next turn (interleaved-thinking pattern).
function Session:add_assistant(text, extras)
    return self:add("assistant", text, extras)
end

--- Append a tool-result message tied back to a call id.
--- @param  tool_call_id string
--- @param  content      string|table  Stringified result, or a table the
---                                    template will JSON-encode.
function Session:add_tool_result(tool_call_id, content)
    local msg = messages_util.tool_result(tool_call_id, content)
    self.messages[#self.messages + 1] = msg
    self._dirty = true
    return self
end

--- Replace the system prompt. Mutating the system text invalidates
--- the prefix cache for this session ; the `kv` layer recomputes on
--- next prepare.
--- @param  text string
function Session:set_system(text)
    if text ~= self.system then
        self.system        = text
        self._seq_snapshot = nil
        self._dirty        = true
        self.n_past        = 0
        self._msg_kv_ends  = {}
    end
    return self
end

-- ── Read-side helpers ─────────────────────────────────────────────────────

--- The full message list as the chat template expects it : system
--- (when present) followed by the chronological turns.
--- @return table[]
function Session:all_messages()
    if not self.system then return self.messages end
    local out = { { role = "system", content = self.system } }
    for _, m in ipairs(self.messages) do out[#out + 1] = m end
    return out
end

--- Just the conversational messages (no system). Same array
--- reference Session manages internally — do not mutate.
--- @return table[]
function Session:pending_messages()
    return self.messages
end

--- Last `Response` produced for this session, or nil if no chat ran
--- yet. Set by `engine:chat` / `engine:stream`.
--- @return ion7.llm.Response?
function Session:last_response()
    return self._last_response
end

--- Plain-text dump of the conversation, one `role: content` line per
--- message. Useful when feeding history into a summariser or RAG
--- chain.
--- @param  msgs table?  Defaults to `self.messages` (excludes system).
--- @return string
function Session:format(msgs)
    msgs = msgs or self.messages
    local parts = {}
    for _, m in ipairs(msgs) do
        parts[#parts + 1] = m.role .. ": " .. m.content
    end
    return table.concat(parts, "\n")
end

-- ── Lifecycle ─────────────────────────────────────────────────────────────

--- Drop every message and reset KV bookkeeping. The `seq_id` stays
--- assigned (the row is still ours) ; the `kv` layer drops the row's
--- contents through `kv_seq_rm` at next prepare when it sees
--- `n_past == 0`.
--- @return ion7.llm.Session self
function Session:reset()
    self.messages       = {}
    self.n_past         = 0
    self._seq_snapshot  = nil
    self._dirty         = false
    self._msg_kv_ends   = {}
    self._last_response = nil
    return self
end

--- Lua-side history clone. The companion KV copy (`kv_seq_cp src dst`)
--- happens inside `kv:fork(parent_session)` because that needs a
--- live context. Calling `:fork` in isolation gives you a session
--- with the same messages but an unassigned `seq_id` — useful for
--- branching exploration where you do not need shared KV.
--- @return ion7.llm.Session
function Session:fork()
    local child = Session.new({ system = self.system })
    for _, m in ipairs(self.messages) do
        local copy = { role = m.role, content = m.content }
        if m.thinking     then copy.thinking     = m.thinking     end
        if m.tool_calls   then copy.tool_calls   = m.tool_calls   end
        if m.tool_call_id then copy.tool_call_id = m.tool_call_id end
        child.messages[#child.messages + 1] = copy
    end
    -- The KV state is copyable but the snapshot blob is not — it is
    -- specific to one seq_id at one moment. The kv layer issues a
    -- `seq_cp` and re-snapshots the child after the copy.
    child.n_past = self.n_past
    for i, v in ipairs(self._msg_kv_ends) do child._msg_kv_ends[i] = v end
    return child
end

-- ── KV bookkeeping (called by engine / kv layer, not by users) ────────────

--- Set the snapshot + per-session n_past. Called after a successful
--- decode. The blob is the result of `ctx:seq_snapshot(self.seq_id)`,
--- restorable later via `ctx:seq_restore(blob, self.seq_id)` without
--- disturbing other sessions sharing the context.
--- @param  blob   string
--- @param  n_past integer
function Session:_save_seq_snapshot(blob, n_past)
    self._seq_snapshot = blob
    self.n_past        = n_past
    self._dirty        = false
end

--- True when `_save_seq_snapshot` was called at least once and the
--- session has not been mutated since.
--- @return boolean
function Session:_has_clean_snapshot()
    return self._seq_snapshot ~= nil and not self._dirty
end

--- Raw snapshot blob.
--- @return string?
function Session:_seq_snapshot_blob()
    return self._seq_snapshot
end

-- ── Persistence ──────────────────────────────────────────────────────────

--- Plain Lua table view (no KV blob — that is per-context and not
--- portable across runs).
--- @return table
function Session:serialize()
    return {
        id       = self.id,
        system   = self.system,
        messages = self.messages,
    }
end

--- Re-build a Session from a `serialize()` table. The new session
--- has no snapshot and starts dirty so the engine will re-encode the
--- whole history on first `engine:chat`.
--- @param  t table
--- @return ion7.llm.Session
function Session.deserialize(t)
    local s = Session.new({ system = t.system })
    s.id       = t.id or s.id
    s.messages = t.messages or {}
    s._dirty   = #s.messages > 0
    return s
end

--- Save to disk as JSON. Goes through `ion7.vendor.json`.
--- @param  path string
--- @return boolean ok
--- @return string? err
function Session:save(path)
    local f, ferr = io.open(path, "w")
    if not f then return false, ferr or ("cannot open " .. path) end
    f:write(json.encode(self:serialize()))
    f:close()
    return true
end

--- Inverse of `:save`. Returns `(session, nil)` on success or
--- `(nil, err)` on failure.
--- @param  path string
--- @return ion7.llm.Session?
--- @return string?
function Session.load(path)
    local f, ferr = io.open(path, "r")
    if not f then return nil, ferr or ("cannot open " .. path) end
    local data = f:read("*a") ; f:close()
    local ok, decoded = pcall(json.decode, data)
    if not ok then return nil, tostring(decoded) end
    return Session.deserialize(decoded)
end

return Session
