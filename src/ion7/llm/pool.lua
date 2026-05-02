--- @module ion7.llm.pool
--- @author  ion7 / Ion7 Project Contributors
---
--- Multi-session inference pool. Drives N concurrent conversations
--- through ONE shared `ion7.core.Context`, packing one token per
--- active session into a single `llama_decode` call per tick.
---
--- This is the value-add over a `for s in sessions do engine:chat(s)`
--- loop : every active session contributes one row to the same batch,
--- the GPU sees a wider matmul, throughput-per-clock goes up
--- considerably (the parallel example in llama.cpp lands roughly N×
--- aggregate tokens / s for small N, decaying as the K cache grows).
---
--- The pool owns its OWN `llama_batch` struct — it does not borrow
--- ion7-core's pre-allocated decode batch. Side-by-side use with
--- `Engine` against the same context is safe : the engine only
--- touches its batch through `ctx:decode`, which fills the batch
--- afresh on every call.
---
--- Per-slot streaming chunks include `tool_call_delta` /
--- `tool_call_done` events when the slot crosses a tool-call marker
--- in the content stream — same set of formats `chat.tool_stream`
--- recognises (OpenAI / Qwen / Mistral, plus `auto`).
---
--- Mid-generation eviction : when a slot's per-seq KV row is about
--- to overflow, the pool calls `cm:make_room(slot.session, ...)`
--- between ticks rather than stopping the slot. The decode loop
--- keeps running until `max_tokens` or a real stop condition fires.
---
--- Lifecycle :
---
---   local pool = llm.Pool.new(ctx, vocab, cm)
---   for _, s in ipairs(sessions) do
---       pool:add(s, { sampler = profiles.balanced(), max_tokens = 256,
---                     on_chunk = function(slot, chunk) ... end })
---   end
---   pool:run()                          -- blocks until every slot is done
---   for _, slot in ipairs(pool:slots()) do
---       print(slot.session:last_response().content)
---   end

local ffi = require "ffi"

local llama_batch   = require "ion7.core.ffi.llama.batch"
local llama_context = require "ion7.core.ffi.llama.context"

local Response    = require "ion7.llm.response"
local Thinking    = require "ion7.llm.chat.thinking"
local Stream      = require "ion7.llm.chat.stream"
local ToolStream  = require "ion7.llm.chat.tool_stream"
local Stop        = require "ion7.llm.stop"
local Parse       = require "ion7.llm.chat.parse"
local snapshot    = require "ion7.llm.kv.snapshot"
local profiles    = require "ion7.llm.sampler.profiles"
local log         = require "ion7.llm.util.log"

-- Hot-path locals.
local string_format   = string.format
local table_concat    = table.concat
local stream_content  = Stream.content
local stream_thinking = Stream.thinking
local stream_stop     = Stream.stop
local llama_decode    = llama_context.llama_decode

local Pool = {}
Pool.__index = Pool

--- @class ion7.llm.pool.Slot
--- @field session     ion7.llm.Session
--- @field sampler     ion7.core.Sampler
--- @field stop        ion7.llm.Stop
--- @field thinking    ion7.llm.chat.Thinking
--- @field tool_sm     ion7.llm.chat.tool_stream?
--- @field raw_parts   string[]
--- @field toks        integer[]
--- @field max_tokens  integer
--- @field on_chunk    function?    `(slot, chunk) -> nil`. `chunk` is a
---                                   typed table from `chat/stream`.
--- @field next_tok    integer?     Token to consume on next tick.
--- @field n_generated integer
--- @field stop_reason string?      Set when the slot terminates.

--- @class ion7.llm.Pool
--- @field _ctx       ion7.core.Context
--- @field _vocab     ion7.core.Vocab
--- @field _cm        ion7.llm.kv.ContextManager
--- @field _batch     cdata
--- @field _batch_cap integer
--- @field _batch_gc  cdata
--- @field _slots     ion7.llm.pool.Slot[]
--- @field _opts      table

--- Build a pool.
---
--- @param  ctx   ion7.core.Context
--- @param  vocab ion7.core.Vocab
--- @param  cm    ion7.llm.kv.ContextManager
--- @param  opts  table?
---   `batch_capacity`      (integer)  Slots accepted per tick. Defaults to
---                                    `ctx:n_seq_max()`.
---   `max_tokens`          (integer)  Per-slot generation cap (default 2048).
---   `mid_gen_safety`      (integer, default 4)  Headroom guard before n_ctx.
---   `mid_gen_evict_chunk` (integer, default 64) Tokens freed per evict round.
--- @return ion7.llm.Pool
function Pool.new(ctx, vocab, cm, opts)
    assert(ctx and vocab and cm,
        "[ion7.llm.pool] Pool.new(ctx, vocab, cm, opts) requires all three")
    opts = opts or {}

    local cap = opts.batch_capacity or ctx:n_seq_max()
    assert(cap > 0, "[ion7.llm.pool] batch_capacity must be > 0")

    -- Own batch — one row per concurrent session, one seq_id per row.
    local b = llama_batch.llama_batch_init(cap, 0, 1)
    for i = 0, cap - 1 do b.n_seq_id[i] = 1 end

    local batch_gc = ffi.gc(ffi.new("int8_t[1]"),
        function() llama_batch.llama_batch_free(b) end)

    return setmetatable({
        _ctx       = ctx,
        _vocab     = vocab,
        _cm        = cm,
        _batch     = b,
        _batch_cap = cap,
        _batch_gc  = batch_gc,
        _slots     = {},
        _opts      = {
            max_tokens          = opts.max_tokens          or 2048,
            mid_gen_safety      = opts.mid_gen_safety      or 4,
            mid_gen_evict_chunk = opts.mid_gen_evict_chunk or 64,
        },
    }, Pool)
end

-- ── Internal helpers ──────────────────────────────────────────────────────

-- Route a chat-thinking demuxed piece through the slot's tool stream
-- and forward typed events to the consumer's `on_chunk`. Falls back
-- to a plain `content` chunk when the slot has no tool stream
-- (`opts.tools_stream = false` at add() time).
local function dispatch_content(slot, text)
    if #text == 0 or not slot.on_chunk then return end
    local ts = slot.tool_sm
    if ts then
        local events = ts:feed(text)
        for i = 1, #events do slot.on_chunk(slot, events[i]) end
    else
        slot.on_chunk(slot, stream_content(text))
    end
end

local function emit_chunk(slot, piece)
    if not slot.on_chunk then return end
    local kind, text, suffix = slot.thinking:feed(piece)
    if kind == "content" then
        dispatch_content(slot, text)
    elseif kind == "thinking" then
        if #text > 0 then slot.on_chunk(slot, stream_thinking(text)) end
    elseif kind == "split" then
        if slot.thinking:in_think() then
            dispatch_content(slot, text)
            if #suffix > 0 then slot.on_chunk(slot, stream_thinking(suffix)) end
        else
            if #text > 0 then slot.on_chunk(slot, stream_thinking(text)) end
            dispatch_content(slot, suffix)
        end
    end
end

-- Build the per-call Response and persist it on the session. Idempotent
-- — callable from `:run` and `:tick` without double-finalising.
function Pool:_finalise(slot)
    if slot._response then return slot._response end

    local raw = table_concat(slot.raw_parts)
    if slot.stop_reason == "stop_string" then
        local probe = Stop.new({ extra = slot.stop_strings })
        local toks  = slot.toks
        local vocab = self._vocab
        for i = 1, #toks do
            local p = vocab:piece(toks[i], true)
            local matched = probe:feed(p)
            if matched then raw = probe:text_before(matched) ; break end
        end
    end

    local parsed = Parse.split(self._vocab, raw, {
        enable_thinking = slot.thinking_opt,
        tool_format     = slot.tool_format,
    })
    local final_stop = parsed.has_tools and "tool_use" or slot.stop_reason

    -- Refresh per-seq snapshot so the next chat against this session
    -- can take the fast path without re-encoding.
    if slot.session.seq_id ~= nil then
        local blob = snapshot.save(self._ctx, slot.session.seq_id)
        slot.session:_save_seq_snapshot(blob, slot.session.n_past)
    end

    local resp = Response.new({
        content     = parsed.content,
        thinking    = parsed.thinking,
        tool_calls  = parsed.tool_calls,
        tokens      = slot.toks,
        stop_reason = final_stop,
        perf        = self._ctx:perf(),
    })
    slot.session._last_response = resp
    slot._response = resp

    -- Fire the stop chunk AFTER the Response is wired onto the
    -- session, so an `on_chunk(slot, { kind = "stop" })` handler can
    -- read `slot.session:last_response()` to grab the final perf /
    -- tokens / parsed content.
    if slot.on_chunk then
        slot.on_chunk(slot, stream_stop(final_stop))
    end
    return resp
end

-- ── Slot management ──────────────────────────────────────────────────────

--- Register a session into the pool. Prefills it through the context
--- manager, samples the first token, and queues it for the next tick.
---
--- @param  session ion7.llm.Session
--- @param  opts    table?
---   `sampler`      (ion7.core.Sampler?)  Per-slot chain. Default profile.
---   `max_tokens`   (integer?)            Per-slot generation cap.
---   `stop_strings` (string[]?)           Extra stop markers.
---   `on_chunk`     (function?)           `(slot, typed_chunk)` callback.
---   `thinking`     (boolean?)            Forwarded to post-parse.
---   `think_budget` (integer?)            Caps `<think>` block size.
---   `tool_format`  (string?)             `"auto" | "openai" | "qwen" | "mistral"`.
---   `tools_stream` (boolean, default true)  Wire a `tool_stream` SM into
---                                            the streaming callback so the
---                                            consumer gets `tool_call_delta`
---                                            chunks live. Pass `false` to
---                                            shrink memory when streaming
---                                            tool calls is not needed.
--- @return ion7.llm.pool.Slot
function Pool:add(session, opts)
    opts = opts or {}

    self._cm:prepare(session)

    local sampler = opts.sampler or profiles.balanced()

    -- The very first sample reads idx -1 of whatever the cm just left
    -- behind. cm's slow path always ends with a 1+ token decode that
    -- sets last-position logits ; the snapshot fast path leaves stale
    -- logits but is only taken when the session is unmutated since the
    -- previous chat — in which case n_past was already advanced past
    -- the last sample.
    local first_tok = nil
    if session.n_past > 0 then
        first_tok = sampler:sample(self._ctx:ptr(), -1)
    end

    local want_tools_stream = opts.tools_stream
    if want_tools_stream == nil then want_tools_stream = true end

    local slot = {
        session       = session,
        sampler       = sampler,
        stop          = Stop.new({ extra = opts.stop_strings }),
        thinking      = Thinking.new(),
        tool_sm       = want_tools_stream
                          and ToolStream.new({ format = opts.tool_format or "auto" })
                          or nil,
        raw_parts     = {},
        toks          = {},
        max_tokens    = opts.max_tokens or self._opts.max_tokens,
        stop_strings  = opts.stop_strings,
        on_chunk      = opts.on_chunk,
        next_tok      = first_tok,
        n_generated   = 0,
        stop_reason   = nil,
        thinking_opt  = opts.thinking,
        think_budget  = opts.think_budget,
        tool_format   = opts.tool_format,
    }
    self._slots[#self._slots + 1] = slot
    return slot
end

--- Snapshot of the slot list. Returns the live array — do not mutate.
--- @return ion7.llm.pool.Slot[]
function Pool:slots() return self._slots end

--- Number of slots that have not yet hit a stop condition.
--- @return integer
function Pool:n_active()
    local n = 0
    local slots = self._slots
    for i = 1, #slots do
        local s = slots[i]
        if s.stop_reason == nil and s.next_tok ~= nil then n = n + 1 end
    end
    return n
end

-- ── Tick / run loop ──────────────────────────────────────────────────────

--- Execute one parallel decode step. Returns true when at least one
--- slot was processed, false when every slot has already terminated.
--- @return boolean
function Pool:tick()
    local slots = self._slots
    local active = {}
    for i = 1, #slots do
        local s = slots[i]
        if s.stop_reason == nil and s.next_tok ~= nil then
            active[#active + 1] = s
        end
    end
    if #active == 0 then return false end

    -- Drain : consume each slot's pre-sampled next_tok and apply the
    -- halt checks. Survivors land in `to_decode`.
    local cm        = self._cm
    local vocab     = self._vocab
    local ctx       = self._ctx
    local opts      = self._opts
    local mid_safe  = opts.mid_gen_safety
    local mid_chunk = opts.mid_gen_evict_chunk
    local n_ctx_seq = ctx:n_ctx_seq()

    local to_decode = {}
    for i = 1, #active do
        local s   = active[i]
        local tok = s.next_tok
        s.toks[#s.toks + 1] = tok

        if vocab:is_eog(tok) then
            s.stop_reason = "stop"
        else
            local piece = vocab:piece(tok, true)
            s.raw_parts[#s.raw_parts + 1] = piece
            emit_chunk(s, piece)

            if s.think_budget and s.thinking:in_think()
                    and s.thinking:active_token_count() >= s.think_budget then
                s.thinking:force_close()
            end

            local matched = s.stop:feed(piece)
            if matched then
                s.stop_reason = "stop_string"
            elseif s.n_generated + 1 >= s.max_tokens then
                s.stop_reason = "length"
            else
                -- Mid-generation eviction guard. When the next decode
                -- would breach the wall, ask the cm to free
                -- `mid_chunk` cells. Slots that cannot evict (recurrent
                -- models / no movable tokens) terminate gracefully.
                if s.session.n_past + 1 + mid_safe > n_ctx_seq then
                    local freed = cm:make_room(s.session, mid_chunk)
                    if freed == 0 then
                        s.stop_reason = "length"
                    else
                        to_decode[#to_decode + 1] = s
                    end
                else
                    to_decode[#to_decode + 1] = s
                end
            end
        end
    end

    if #to_decode > self._batch_cap then
        error(string_format(
            "[ion7.llm.pool] %d active slots exceed batch capacity %d",
            #to_decode, self._batch_cap), 2)
    end

    if #to_decode == 0 then return true end

    -- Pack : one row per surviving slot.
    local b = self._batch
    for i = 1, #to_decode do
        local s   = to_decode[i]
        local idx = i - 1
        b.token[idx]      = s.next_tok
        b.pos[idx]        = s.session.n_past
        b.seq_id[idx][0]  = s.session.seq_id
        b.n_seq_id[idx]   = 1
        b.logits[idx]     = 1
    end
    b.n_tokens = #to_decode

    local rc = llama_decode(ctx:ptr(), b)
    if rc ~= 0 then
        log.warn(string_format(
            "[ion7.llm.pool] llama_decode returned %d, terminating active slots", rc))
        for i = 1, #to_decode do to_decode[i].stop_reason = "error" end
        return true
    end

    -- Advance + sample next.
    local ctx_ptr = ctx:ptr()
    for i = 1, #to_decode do
        local s = to_decode[i]
        s.session.n_past = s.session.n_past + 1
        s.n_generated    = s.n_generated   + 1
        s.next_tok       = s.sampler:sample(ctx_ptr, i - 1)
    end

    return true
end

--- Drive every slot to completion. Calls `tick` in a loop, then runs
--- `:_finalise` on each slot so `slot.session:last_response()` is
--- ready to read.
--- @return ion7.llm.pool.Slot[]  same array `:slots()` returns
function Pool:run()
    while self:tick() do end
    local slots = self._slots
    for i = 1, #slots do self:_finalise(slots[i]) end
    return slots
end

--- Drop every slot. Sessions are NOT released — the caller decides
--- whether to keep them around for the next round of chat.
function Pool:reset()
    self._slots = {}
end

--- Free the pool's batch immediately. Idempotent.
function Pool:free()
    if self._batch_gc then
        ffi.gc(self._batch_gc, nil)
        self._batch_gc = nil
        if self._batch then
            llama_batch.llama_batch_free(self._batch)
            self._batch = nil
        end
    end
end

return Pool
