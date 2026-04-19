--- @module ion7.llm.context_manager
--- SPDX-License-Identifier: MIT
--- KV cache orchestration: prefix cache, session slots, overflow handling, hooks.
---
--- Features:
---   1. PREFIX CACHE - system prompt encoded once, restored per session (~15ms vs ~300ms).
---   2. OVERFLOW HANDLING - sliding window (kv_seq_shift) when supported, hard reset fallback.
---   3. ATTENTION SINK - first n_sink tokens never evicted (StreamingLLM, ICLR 2024).
---   4. SESSION SLOTS - seq_id allocation for multi-sequence contexts.
---   5. HOOKS - before_encode (RAG injection) and on_evict (summarization / external memory).
---
--- @author Ion7-Labs
--- @version 0.2.0

local ContextManager = {}
ContextManager.__index = ContextManager

--- @param  ctx    Context
--- @param  vocab  Vocab
--- @param  opts   table?
---   opts.max_sessions  number?  Max concurrent sessions (default: n_seq_max).
---   opts.headroom      number?  Tokens reserved for generation (default: 256).
---   opts.n_sink        number?  Attention sink size (default: 4). Set 0 to disable.
---   opts.eviction      string?  "message" (default) or "fifo".
--- @return ContextManager
function ContextManager.new(ctx, vocab, opts)
    opts = opts or {}
    local n_seq = tonumber(ctx:n_seq_max()) or 1
    local max_s = math.min(opts.max_sessions or n_seq, n_seq)

    local slots = {}
    for i = 0, max_s - 1 do slots[i] = true end

    return setmetatable({
        _ctx              = ctx,
        _vocab            = vocab,
        _n_seq            = n_seq,
        _max_sessions     = max_s,
        _slots            = slots,
        _prefix_snap      = nil,
        _prefix_n         = 0,
        _prefix_text      = nil,
        _hooks            = {},
        _headroom         = opts.headroom or 256,
        _n_sink           = (opts.n_sink ~= nil) and opts.n_sink or 4,
        _eviction         = opts.eviction or "message",
        _n_evictions      = 0,
        _n_tokens_evicted = 0,
        -- When true, passes enable_thinking=0 to apply_template, which lets the
        -- model's Jinja2 template disable thinking natively (Qwen3/3.5, DeepSeek-R1).
        -- Maps to -1 (model default) when false.
        _no_think         = opts.no_think or false,
    }, ContextManager)
end

-- ── Prefix cache ──────────────────────────────────────────────────────────────

--- Encode the system prompt and save a KV snapshot.
--- No-op if called again with the same text.
--- @param  system_text  string
--- @return number  token count
function ContextManager:set_system(system_text)
    assert(type(system_text) == "string",
        "[ion7.llm.context_manager] system_text must be a string")

    if self._prefix_text == system_text and self._prefix_snap then
        return self._prefix_n
    end

    local ctx, vocab = self._ctx, self._vocab

    -- Some templates (e.g. Qwen3.5 Jinja2) throw when the message list
    -- contains only a system role with no user message.
    -- Fallback: apply with an empty user turn and trim the user portion.
    local prompt
    local ok, result = pcall(vocab.apply_template, vocab,
        { { role = "system", content = system_text } }, false)
    if ok then
        prompt = result
    else
        local full = vocab:apply_template({
            { role = "system", content = system_text },
            { role = "user",   content = "" },
        }, false)
        -- Trim at the start of the user turn (ChatML, Llama-2, Phi formats)
        local user_start = full:find("\n<|im_start|>user", 1, true)
                        or full:find("\n%[INST%]", 1, false)
                        or full:find("\n<|user|>", 1, true)
        prompt = user_start and full:sub(1, user_start) or full
    end

    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)

    self._prefix_snap = ctx:snapshot()
    self._prefix_n    = n
    self._prefix_text = system_text
    return n
end

--- @return boolean
function ContextManager:has_prefix()
    return self._prefix_snap ~= nil
end

--- Restore prefix snapshot into context.
--- @return boolean
function ContextManager:restore_prefix()
    if not self._prefix_snap then return false end
    self._ctx:restore(self._prefix_snap)
    self._ctx:set_n_past(self._prefix_n)
    return true
end

-- ── Hooks ─────────────────────────────────────────────────────────────────────

--- Build a prompt string from messages.
--- Passes enable_thinking to the Jinja2 template (0 = disable, -1 = model default).
--- @param  msgs     table    Message array
--- @param  add_ass  boolean  Add generation prompt
--- @return string
function ContextManager:_prompt(msgs, add_ass)
    local et = (add_ass and self._no_think) and 0 or -1
    return self._vocab:apply_template(msgs, add_ass, et)
end

--- Install a named hook.
---
--- "before_encode"  fn(messages, session) -> messages?
---   Called before tokenizing messages. Return modified array or nil.
---
--- "on_evict"  fn(evicted_messages, session) -> messages?
---   Called when messages are dropped due to overflow. Return replacement
---   messages (e.g. summary) or nil to drop silently.
---
--- @param  name  string
--- @param  fn    function
function ContextManager:set_hook(name, fn)
    assert(type(name) == "string",   "[ion7.llm.context_manager] hook name must be a string")
    assert(type(fn)   == "function", "[ion7.llm.context_manager] hook must be a function")
    self._hooks[name] = fn
end

--- @param  name  string
function ContextManager:clear_hook(name)
    self._hooks[name] = nil
end

--- @param  name  string
--- @return function?
function ContextManager:get_hook(name)
    return self._hooks[name]
end

-- ── Slot management ───────────────────────────────────────────────────────────

function ContextManager:_acquire_slot()
    for seq_id, free in pairs(self._slots) do
        if free then
            self._slots[seq_id] = false
            return seq_id
        end
    end
    return nil
end

function ContextManager:_release_slot(seq_id)
    self._ctx:kv_seq_rm(seq_id, 0, -1)
    self._slots[seq_id] = true
end

-- ── Eviction ──────────────────────────────────────────────────────────────────

-- Per-message KV end positions (approximation via single-message tokenization).
-- Some templates (e.g. Qwen3.5) fail when given a single assistant message with
-- add_generation_prompt=false. Fall back to raw content tokenization in that case.
function ContextManager:_compute_msg_positions(msgs, base_pos)
    local vocab, pos, ends = self._vocab, base_pos, {}
    for i, msg in ipairs(msgs) do
        local ok, tmpl = pcall(vocab.apply_template, vocab, { msg }, false)
        local _, n
        if ok then
            _, n = vocab:tokenize(tmpl, false, true)
        else
            -- Fallback: tokenize raw content + small fixed overhead for role tokens
            _, n = vocab:tokenize(msg.content or "", false, true)
            n = n + 6  -- role header + newlines overhead
        end
        pos = pos + n
        ends[i] = pos
    end
    return ends
end

-- Remove oldest tokens to free at least `overflow` tokens.
-- Respects n_sink (pinned zone) and eviction strategy.
function ContextManager:_evict(seq_id, overflow, session)
    local ctx        = self._ctx
    local keep_from  = self._prefix_n + self._n_sink
    local movable    = ctx:n_past() - keep_from
    if movable <= 0 then return end

    if self._eviction == "message" and session and #session._msg_kv_ends > 0 then
        local ends = session._msg_kv_ends
        local freed, last_pos, count = 0, keep_from, 0

        for i = 1, #ends do
            local msg_end = ends[i]
            if msg_end <= keep_from then
                count   = i
                last_pos = msg_end
            else
                local start  = (i == 1) and keep_from or math.max(ends[i - 1], keep_from)
                freed   = freed + (msg_end - start)
                count   = i
                last_pos = msg_end
                if freed >= overflow then break end
            end
        end

        if freed > 0 then
            ctx:kv_seq_rm(seq_id, keep_from, last_pos)
            ctx:kv_seq_shift(seq_id, -freed, last_pos, -1)
            ctx:set_n_past(ctx:n_past() - freed)

            local new_ends = {}
            for i = count + 1, #ends do new_ends[#new_ends + 1] = ends[i] - freed end
            session._msg_kv_ends = new_ends

            self._n_evictions      = self._n_evictions + 1
            self._n_tokens_evicted = self._n_tokens_evicted + freed
        end
    else
        local shift = math.min(math.max(overflow, math.floor(movable / 2)), movable)
        if shift > 0 then
            ctx:kv_seq_rm(seq_id, keep_from, keep_from + shift)
            ctx:kv_seq_shift(seq_id, -shift, keep_from + shift, -1)
            ctx:set_n_past(ctx:n_past() - shift)

            if session then
                local new_ends = {}
                for _, e in ipairs(session._msg_kv_ends) do
                    local adj = e - shift
                    if adj > keep_from then new_ends[#new_ends + 1] = adj end
                end
                session._msg_kv_ends = new_ends
            end

            self._n_evictions      = self._n_evictions + 1
            self._n_tokens_evicted = self._n_tokens_evicted + shift
        end
    end
end

-- ── Session lifecycle ─────────────────────────────────────────────────────────

--- Prepare KV for a session's next decode. Returns n_past.
---
--- Fast path: session has a clean snapshot -> restore, done.
--- Slow path: prefill from prefix (or scratch), handle overflow, decode.
---
--- @param  session  Session
--- @return number   n_past
function ContextManager:prepare(session)
    local ctx, vocab = self._ctx, self._vocab

    if session.seq_id == nil then
        session.seq_id = self:_acquire_slot() or 0
    end

    -- Fast path: clean snapshot, no new messages
    if session:has_snapshot() and not session._dirty then
        ctx:restore(session:snapshot())
        ctx:set_n_past(session.n_past)
        return session.n_past
    end

    -- Slow path: full prefill
    if self:has_prefix() then
        self:restore_prefix()
    else
        ctx:kv_clear()
        ctx:set_n_past(0)
    end

    local all_msgs = session:all_messages()
    if #all_msgs == 0 then
        local snap = ctx:snapshot()
        session:_save_snapshot(snap, ctx:n_past())
        return ctx:n_past()
    end

    -- Strip system message already covered by prefix cache
    local msgs = all_msgs
    if self:has_prefix() and msgs[1] and msgs[1].role == "system" then
        msgs = {}
        for i = 2, #all_msgs do msgs[#msgs + 1] = all_msgs[i] end
    end

    if #msgs == 0 then
        local snap = ctx:snapshot()
        session:_save_snapshot(snap, ctx:n_past())
        return ctx:n_past()
    end

    -- before_encode hook (RAG / context injection)
    local beh = self._hooks.before_encode
    if beh then msgs = beh(msgs, session) or msgs end

    local tokens, n = vocab:tokenize(self:_prompt(msgs, true), false, true)
    local n_ctx_eff  = ctx:n_ctx_seq()

    -- Overflow handling
    if ctx:n_past() + n > n_ctx_eff - self._headroom then
        local overflow = (ctx:n_past() + n) - (n_ctx_eff - self._headroom)
        local seq_id   = session.seq_id or 0

        if ctx:kv_can_shift() then
            self:_evict(seq_id, overflow, session)
        else
            -- Hard reset: wipe KV, re-encode trimmed history from prefix
            if self:has_prefix() then
                self:restore_prefix()
            else
                ctx:kv_clear()
                ctx:set_n_past(0)
            end
            session._msg_kv_ends = {}
        end
    end

    -- Post-reset trim: history may still exceed available space
    local available = n_ctx_eff - self._headroom - ctx:n_past()
    if n > available then
        local n_before_trim = n
        local dropped = {}

        while #msgs > 1 do
            dropped[#dropped + 1] = table.remove(msgs, 1)
            tokens, n = vocab:tokenize(self:_prompt(msgs, true), false, true)
            if n <= available then break end
        end

        -- Last resort: single oversized message - keep only the last one
        if n > available then
            dropped[#dropped + 1] = msgs[1]
            msgs = { msgs[#msgs] }
            tokens, n = vocab:tokenize(self:_prompt(msgs, true), false, true)
        end

        if #dropped > 0 then
            self._n_evictions      = self._n_evictions + 1
            self._n_tokens_evicted = self._n_tokens_evicted + n_before_trim

            -- on_evict hook: allow replacement (summary, marker, etc.)
            local evict_hook = self._hooks.on_evict
            if evict_hook then
                local replacement = evict_hook(dropped, session)
                if replacement and #replacement > 0 then
                    local new_msgs = {}
                    for _, m in ipairs(replacement) do new_msgs[#new_msgs + 1] = m end
                    for _, m in ipairs(msgs)        do new_msgs[#new_msgs + 1] = m end
                    local new_tokens, new_n = vocab:tokenize(
                        self:_prompt(new_msgs, true), false, true)
                    if new_n <= available then
                        msgs, tokens, n = new_msgs, new_tokens, new_n
                    end
                end
            end

            session._msg_kv_ends = {}
        end
    end

    -- Safety clamp against template token overhead
    local actual_free = n_ctx_eff - ctx:n_past()
    if n > actual_free then n = actual_free end
    if n <= 0 then return ctx:n_past() end

    ctx:decode(tokens, n, 0, ctx:n_past())

    if #session._msg_kv_ends == 0 then
        session._msg_kv_ends = self:_compute_msg_positions(msgs, self._prefix_n)
    end

    local snap = ctx:snapshot()
    session:_save_snapshot(snap, ctx:n_past())
    return ctx:n_past()
end

--- Free a session's KV slot. Call when the session is permanently closed.
--- @param  session  Session
function ContextManager:release(session)
    if session.seq_id ~= nil and self._n_seq > 1 then
        self:_release_slot(session.seq_id)
        session.seq_id = nil
    end
end

--- Fork a session. Child shares KV up to this point via kv_seq_cp.
--- @param  session  Session
--- @return Session
function ContextManager:fork(session)
    local child = session:fork()
    if self._n_seq > 1 then
        local new_seq = self:_acquire_slot()
        if new_seq then
            child.seq_id = new_seq
            self._ctx:kv_seq_cp(session.seq_id, new_seq, 0, -1)
        end
    end
    return child
end

--- @return table  { slots_total, slots_free, prefix_tokens, prefix_cached, n_sink, eviction, n_evictions, n_tokens_evicted }
function ContextManager:stats()
    local free = 0
    for _, v in pairs(self._slots) do if v then free = free + 1 end end
    return {
        slots_total      = self._max_sessions,
        slots_free       = free,
        prefix_tokens    = self._prefix_n,
        prefix_cached    = self._prefix_snap ~= nil,
        n_sink           = self._n_sink,
        eviction         = self._eviction,
        n_evictions      = self._n_evictions,
        n_tokens_evicted = self._n_tokens_evicted,
    }
end

return ContextManager
