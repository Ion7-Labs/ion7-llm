--- @module ion7.llm.kv
--- @author  ion7 / Ion7 Project Contributors
---
--- Top-level KV cache orchestrator. Wraps the four building blocks
--- under this directory (`snapshot`, `slots`, `prefix`, `eviction`)
--- behind a single `ContextManager` API the engine and pool talk to.
---
--- Responsibilities, in execution order :
---
---   1. Slot allocation. On first `prepare(session)`, claim a free
---      `seq_id` from the pool ; release it on `release(session)`.
---   2. Prefix copy. When the prefix manager owns a real slot, the
---      session's row gets the prefix `kv_seq_cp`'d in before any
---      user message is decoded.
---   3. Snapshot fast-path. A clean (`!_dirty`) session restores its
---      `_seq_snapshot` instead of replaying the whole history.
---   4. Slow-path prefill. Render the chat template, tokenise,
---      decode the suffix that is not yet in cache.
---   5. Overflow eviction. Before decoding the suffix, evict enough
---      old tokens to leave `headroom` free for the upcoming
---      generation.
---   6. Snapshot on success. Save `seq_snapshot(seq_id)` back onto the
---      session for the next call's fast path.
---
--- Hooks :
---
---   - `before_encode(messages, session) -> messages?`
---       Called just before tokenisation. Return a replacement array
---       to inject RAG context, condense the history, etc.
---   - `on_evict(dropped_messages, session) -> messages?`
---       Called when eviction drops messages. Return one or more
---       replacement messages (typically a summarisation) to keep
---       the model informed of the dropped content.
---   - `sink_fn(session) -> integer`
---       Called every eviction round to compute a dynamic
---       attention-sink length. Defaults to the constant `n_sink`
---       option set at construction. Use to implement Y-Token-style
---       per-message sinks.
---
--- Mid-generation eviction :
---   The engine and pool call `cm:make_room(session, n_needed)`
---   between decode rounds when the per-seq context is about to
---   overflow. This is identical to the prepare-time overflow
---   handler, just without re-rendering the template afterwards :
---   we shift the surviving rows left and the next decode lands at
---   `session.n_past`.
---
--- Context configuration : multi-session use requires the underlying
--- `ion7.core.Context` to be created with `kv_unified = true` AND
--- `n_seq_max > 1`. Both `kv_seq_cp` (used by `Prefix:copy_to` and
--- `ContextManager:fork`) and `seq_snapshot` rely on the unified KV
--- layout. A non-unified context is fine for the single-session path
--- where the prefix degrades to text-only mode.

local snapshot = require "ion7.llm.kv.snapshot"
local Slots    = require "ion7.llm.kv.slots"
local Prefix   = require "ion7.llm.kv.prefix"
local eviction = require "ion7.llm.kv.eviction"
local RadixCache = require "ion7.llm.kv.radix"

-- Hot-path locals. Caching builtins as upvalues lets LuaJIT compile
-- to direct references instead of metatable lookups every call.
local table_insert = table.insert
local ipairs       = ipairs

local ContextManager = {}
ContextManager.__index = ContextManager

--- @class ion7.llm.kv.ContextManager
--- @field _ctx        ion7.core.Context
--- @field _vocab      ion7.core.Vocab
--- @field _slots      ion7.llm.kv.Slots
--- @field _prefix     ion7.llm.kv.Prefix
--- @field _hooks      table
--- @field _opts       table

--- Build a manager.
---
--- @param  ctx   ion7.core.Context
--- @param  vocab ion7.core.Vocab
--- @param  opts  table?
---   `headroom`     (integer, default 256)  Tokens reserved for the next
---     generation step. Eviction kicks in early enough to leave at
---     least this many cells free in the per-sequence window.
---   `n_sink`       (integer, default 4)    Static attention-sink length.
---   `eviction`     (string,  default `"message"`)  `"message"` | `"fifo"`.
---   `no_think`     (boolean, default false) When true, the chat template
---     is asked with `enable_thinking = 0`. Models that ignore the
---     flag (no thinking variants) just see the regular template.
---   `radix`        (boolean | table, default false) Enable the
---     RadixAttention-style prefix cache. Pass `true` for the default
---     parameters or a table with overrides : `chunk_size`,
---     `max_blobs`, `min_match`. The cache indexes per-seq snapshots
---     by their post-prefix token suffix, so two sessions whose
---     conversation begins identically share the encode work.
--- @return ion7.llm.kv.ContextManager
function ContextManager.new(ctx, vocab, opts)
    opts = opts or {}
    local prefix = Prefix.new(ctx, vocab)
    local slots  = Slots.new(ctx, {
        reserved = prefix:has_slot() and prefix:seq_id() or nil,
    })

    -- Radix cache. Disabled by default ; passing `radix = true` uses
    -- the default parameters, passing a table merges into them.
    local radix
    if opts.radix then
        local rcfg = type(opts.radix) == "table" and opts.radix or {}
        radix = RadixCache.new(rcfg)
    end

    return setmetatable({
        _ctx    = ctx,
        _vocab  = vocab,
        _slots  = slots,
        _prefix = prefix,
        _radix  = radix,
        _hooks  = {},
        _opts   = {
            headroom         = opts.headroom or 256,
            n_sink           = (opts.n_sink ~= nil) and opts.n_sink or 4,
            eviction         = opts.eviction or "message",
            no_think         = opts.no_think or false,
        },
        _stats  = {
            n_evictions      = 0,
            n_tokens_evicted = 0,
        },
    }, ContextManager)
end

-- ── Hooks ─────────────────────────────────────────────────────────────────

--- Install a hook. See module docstring for the list of names.
--- @param  name string
--- @param  fn   function
function ContextManager:set_hook(name, fn)
    assert(type(name) == "string" and type(fn) == "function",
        "[ion7.llm.kv] set_hook(name, fn) expects a string and a function")
    self._hooks[name] = fn
end

function ContextManager:clear_hook(name) self._hooks[name] = nil end

-- ── Prefix API (forwarded) ────────────────────────────────────────────────

--- Set or replace the system prompt. Does the right thing for both
--- multi-seq (encode into the reserved prefix slot once, copy to
--- every session) and single-seq (remember text only) modes.
---
--- A change in system text invalidates the radix cache : every
--- stored blob was taken under the previous system, restoring it
--- now would mix two different priors.
--- @param  text string
--- @return integer
function ContextManager:set_system(text)
    local prev = self._prefix:text()
    local n    = self._prefix:set(text)
    if self._radix and prev ~= text then
        self._radix:clear()
    end
    return n
end

function ContextManager:has_prefix() return self._prefix:n_tokens() > 0 end
function ContextManager:prefix_n_tokens() return self._prefix:n_tokens() end
function ContextManager:prefix_text() return self._prefix:text() end

-- ── Internal helpers ──────────────────────────────────────────────────────

local function template_thinking_arg(no_think)
    -- The chat template's `enable_thinking` parameter accepts -1
    -- (model default), 0 (off) or 1 (on). We expose two states only.
    return no_think and 0 or -1
end

function ContextManager:_render(messages, add_ass)
    local et = template_thinking_arg(add_ass and self._opts.no_think)
    return self._vocab:apply_template(messages, add_ass, et)
end

-- Compute the active sink count : if the consumer set a `sink_fn`
-- hook, call it with the session and trust its return value ; else
-- fall back to the constant `n_sink` option.
function ContextManager:_compute_sinks(session)
    local sink_fn = self._hooks.sink_fn
    if sink_fn then
        local n = sink_fn(session)
        if type(n) == "number" and n >= 0 then return n end
    end
    return self._opts.n_sink
end

-- Build per-message KV end positions starting at `base_pos`. Used by
-- the message-aligned eviction strategy. We render the prefix
-- `messages[1..i]` for every i and take the running token count.
-- That respects templates that enforce strict role alternation
-- (Ministral, Llama-3-Instruct, ...) where rendering a lone
-- `assistant` message in isolation is not a valid template input.
-- The cost is O(n²) tokenisations per prepare() — fine for typical
-- chat-length histories ; for very long sessions the FIFO eviction
-- strategy is a faster alternative.
function ContextManager:_compute_msg_positions(messages, base_pos)
    local vocab = self._vocab
    local apply = vocab.apply_template
    local n_msgs = #messages
    local ends, prev_n = {}, 0
    -- Build the prefix list once and grow it ; cheaper than rebuilding
    -- a fresh sub-array per iteration.
    local sub = {}
    for i = 1, n_msgs do
        sub[i] = messages[i]
        local ok, prompt = pcall(apply, vocab, sub, false, -1)
        if not ok then
            local clen = #(messages[i].content or "")
            ends[i] = base_pos + prev_n + clen
            prev_n  = prev_n + clen
        else
            local _, n = vocab:tokenize(prompt, false, true)
            ends[i] = base_pos + n
            prev_n  = n
        end
    end
    return ends
end

function ContextManager:_acquire_slot(session)
    local seq_id = self._slots:acquire()
    assert(seq_id ~= nil,
        "[ion7.llm.kv] no free seq slot — increase n_seq_max or release sessions")
    session.seq_id = seq_id
    return seq_id
end

-- Run an overflow round and bookkeeping. Returns the eviction report.
-- Pulled out so `prepare` and `make_room` share the exact same logic.
function ContextManager:_evict(session, need, prefix_n)
    local rep = eviction.handle_overflow(self._ctx, session.seq_id, session, {
        overflow = need,
        prefix_n = prefix_n,
        n_sink   = self:_compute_sinks(session),
        strategy = self._opts.eviction,
    })
    if rep.freed > 0 then
        self._stats.n_evictions      = self._stats.n_evictions      + 1
        self._stats.n_tokens_evicted = self._stats.n_tokens_evicted + rep.freed
    end
    return rep
end

-- ── Lifecycle ─────────────────────────────────────────────────────────────

--- Make `ctx` ready for the next decode of `session`. After this
--- call the session's row contains exactly the encoded prefix +
--- conversation history up to (but not including) the next reply.
---
--- @param  session ion7.llm.Session
--- @return integer  n_past after prepare (also written onto session)
function ContextManager:prepare(session)
    local ctx, vocab, prefix = self._ctx, self._vocab, self._prefix

    -- 1. Allocate a slot if the session does not have one yet.
    if session.seq_id == nil then
        self:_acquire_slot(session)
    end
    local seq_id = session.seq_id

    -- 2. Fast path : clean snapshot with the same shape as last call.
    if session:_has_clean_snapshot() then
        snapshot.restore(ctx, session:_seq_snapshot_blob(), seq_id)
        return session.n_past
    end

    -- 3. Slow path : prefill from prefix + replay history. Wipe the
    --    seq first so a stale prefill from a previous shape cannot
    --    leak into the new decode.
    ctx:kv_seq_rm(seq_id, 0, -1)
    session.n_past       = 0
    session._msg_kv_ends = {}

    -- Pull the prefix in. In multi-seq mode this is a fast kv_seq_cp ;
    -- in single-seq mode the prefix manager has no slot and we will
    -- include the system message in the rendered prompt instead.
    local prefix_n = prefix:copy_to(seq_id)
    if prefix_n > 0 then session.n_past = prefix_n end

    -- 4. Build the message list. When the prefix is in cache we drop
    --    the system message from `all_messages` — the prefix already
    --    encodes it, re-decoding would double the system tokens.
    local msgs = session:all_messages()
    if prefix_n > 0 and msgs[1] and msgs[1].role == "system" then
        local trimmed = {}
        for i = 2, #msgs do trimmed[#trimmed + 1] = msgs[i] end
        msgs = trimmed
    end

    -- An empty conversation (just system, no user turn yet) needs no
    -- decode — we still snapshot so the next call hits the fast path.
    if #msgs == 0 then
        session:_save_seq_snapshot(snapshot.save(ctx, seq_id), session.n_past)
        return session.n_past
    end

    -- 5. before_encode hook : RAG injection / context rewriting.
    local beh = self._hooks.before_encode
    if beh then msgs = beh(msgs, session) or msgs end

    local prompt   = self:_render(msgs, true)
    local toks, n  = vocab:tokenize(prompt, false, true)
    local n_ctx    = ctx:n_ctx_seq()
    local headroom = self._opts.headroom

    -- 5b. Radix prefix-cache lookup. When a previous session prefilled
    --     a row that started with the same suffix tokens, we restore
    --     its blob (which already carries the prefix + matched suffix)
    --     and skip decoding the matched tokens. The remaining suffix
    --     gets decoded as usual below.
    local radix_skip = 0
    if self._radix then
        local matched, blob = self._radix:find_longest_prefix(toks, n)
        if blob then
            -- The blob represents (prefix + toks[1..matched]) on whatever
            -- seq_id it was saved under ; seq_restore re-targets it to
            -- our current seq_id without disturbing other sessions.
            snapshot.restore(ctx, blob, seq_id)
            session.n_past = prefix_n + matched
            radix_skip     = matched
        end
    end

    -- 6. Overflow handling. We need session.n_past + n + headroom <= n_ctx.
    if session.n_past + n + headroom > n_ctx then
        local need = (session.n_past + n + headroom) - n_ctx
        local rep  = self:_evict(session, need, prefix_n)

        local hook = self._hooks.on_evict
        if hook and #rep.dropped > 0 then
            local replacement = hook(rep.dropped, session)
            if replacement and #replacement > 0 then
                for i = 1, #replacement do
                    table_insert(session.messages, i, replacement[i])
                end
                prompt   = self:_render(session:all_messages(), true)
                toks, n  = vocab:tokenize(prompt, false, true)
            end
        end
    end

    -- 7. Last-resort trim : the message list still does not fit.
    local available = n_ctx - headroom - session.n_past
    if n > available then
        local trimmed = {}
        for i = 1, #msgs do trimmed[i] = msgs[i] end
        while #trimmed > 1 and n > available do
            table.remove(trimmed, 1)
            prompt  = self:_render(trimmed, true)
            toks, n = vocab:tokenize(prompt, false, true)
        end
        msgs = trimmed
    end

    -- Safety clamp — even after trimming, template overhead might
    -- still push us past the available cells. Decode whatever fits.
    if n > available then n = available end
    if n <= 0 then
        session:_save_seq_snapshot(snapshot.save(ctx, seq_id), session.n_past)
        return session.n_past
    end

    -- 8. Decode the suffix that the radix cache could not warm-start
    --    for us. When `radix_skip` covered the whole prompt, we still
    --    need a 1-token "marker" decode so the next sample reads
    --    fresh logits — but typical prompts always grow by at least
    --    one token (the assistant header) so this rarely matters.
    if radix_skip < n then
        local toks_to_decode = toks
        local n_to_decode    = n - radix_skip
        local pos_offset     = session.n_past

        -- When we skipped some tokens, slice the input array. The
        -- decode helper accepts a start offset via the table layout —
        -- we build a fresh shifted table once. Cost is O(n_to_decode),
        -- amortised against the avoided forward pass which is way
        -- bigger.
        if radix_skip > 0 then
            local shifted = {}
            for i = 1, n_to_decode do shifted[i] = toks[radix_skip + i] end
            toks_to_decode = shifted
        end

        ctx:decode(toks_to_decode, n_to_decode, seq_id, pos_offset)
        session.n_past = session.n_past + n_to_decode
    end

    session._msg_kv_ends = self:_compute_msg_positions(msgs, prefix_n)

    -- 9. Snapshot on success. Save a fresh blob and pin it in the
    --    radix cache so future sessions starting with this same
    --    suffix can warm-start from it.
    local blob = snapshot.save(ctx, seq_id)
    session:_save_seq_snapshot(blob, session.n_past)
    if self._radix then
        self._radix:insert(toks, n, blob)
    end

    return session.n_past
end

--- Reclaim at least `n_needed` cells in `session`'s KV row by
--- evicting old tokens. Used by the engine and pool to push back
--- the n_ctx wall mid-generation without re-rendering the prompt.
---
--- The session's `n_past` is updated in place ; the caller must
--- keep its sampler state coherent across the call (penalty history
--- is window-relative, so a small post-eviction off-by-one is
--- tolerable in practice — a deliberate eviction means the model is
--- already at the edge of its useful context anyway).
---
--- @param  session  ion7.llm.Session
--- @param  n_needed integer  Minimum cells to free.
--- @return integer Tokens actually freed (may exceed `n_needed`,
---                 may be 0 if the model does not support shifting).
function ContextManager:make_room(session, n_needed)
    if session.seq_id == nil or n_needed <= 0 then return 0 end
    local prefix_n = self._prefix:n_tokens()
    local rep = self:_evict(session, n_needed, prefix_n)
    -- The `on_evict` hook is intentionally NOT called here ; mid-gen
    -- the message list does not get re-rendered, so plugging a
    -- summary back in would not flow through the model.
    return rep.freed
end

--- Drop the session's KV row + return its slot to the pool. Call
--- when a conversation is permanently closed.
--- @param  session ion7.llm.Session
function ContextManager:release(session)
    if session.seq_id == nil then return end
    self._slots:release(session.seq_id)
    session.seq_id        = nil
    session._seq_snapshot = nil
    session.n_past        = 0
    session._msg_kv_ends  = {}
end

--- Fork a session : duplicate its KV row onto a fresh slot. The
--- child's seq_id is assigned here ; on first prepare the child will
--- snapshot from the copy, no prefix re-encoding needed.
--- @param  session ion7.llm.Session
--- @return ion7.llm.Session
function ContextManager:fork(session)
    local child = session:fork()
    if session.seq_id == nil then return child end
    self:_acquire_slot(child)
    self._ctx:kv_seq_cp(session.seq_id, child.seq_id, 0, -1)
    child._seq_snapshot = snapshot.save(self._ctx, child.seq_id)
    return child
end

-- ── Stats ────────────────────────────────────────────────────────────────

--- Snapshot of the manager's runtime stats.
--- @return table
function ContextManager:stats()
    local rstats = self._radix and self._radix:stats() or nil
    return {
        slots_total      = self._slots:capacity(),
        slots_free       = self._slots:n_free(),
        prefix_tokens    = self._prefix:n_tokens(),
        prefix_cached    = self._prefix:n_tokens() > 0,
        n_sink           = self._opts.n_sink,
        eviction         = self._opts.eviction,
        n_evictions      = self._stats.n_evictions,
        n_tokens_evicted = self._stats.n_tokens_evicted,
        radix            = rstats,
    }
end

--- Force-clear the radix cache. Useful between unrelated workloads
--- when blobs from prior runs would no longer be useful.
function ContextManager:clear_radix()
    if self._radix then self._radix:clear() end
end

return ContextManager
