--- @module ion7.llm.kv.eviction
--- @author  ion7 / Ion7 Project Contributors
---
--- KV cache eviction strategies for sessions whose history overflows
--- the per-sequence context window.
---
--- Two eviction shapes are exposed :
---
---   - `message` — drop whole oldest messages from the head of the
---     conversation. Walks `session._msg_kv_ends` (a per-message KV
---     end-position table), removes the rows for messages `1..k`,
---     then `kv_seq_shift`s the remainder left so positions stay
---     contiguous. Never cuts a message in half ; the model keeps
---     seeing well-formed turn boundaries.
---
---   - `fifo` — drop the requested number of oldest tokens, no matter
---     where they fall in a message. Half the cost of `message` (no
---     boundary search), at the cost of arbitrary mid-sentence cuts.
---     Right choice for very long single-turn streams.
---
--- Both honour two pinned regions :
---
---   - the prefix span — `[0, prefix_n)` — never touched, so the
---     system prompt always stays in cache.
---   - the attention-sink span — `[prefix_n, prefix_n + n_sink)` —
---     never touched, so the very first user-side tokens stay in
---     cache (StreamingLLM, ICLR 2024 ; better long-context recall
---     than naïve FIFO).
---
--- Eviction silently degrades when the model does not support
--- position shifting (`ctx:kv_can_shift() == false`, e.g. recurrent
--- / SSM models) — the handler returns 0-freed and the caller falls
--- back to `prefix-restore + replay`.

local M = {}

--- Try to free at least `overflow` tokens from `seq_id`. The function
--- mutates the KV cache and the session's bookkeeping (`n_past`,
--- `_msg_kv_ends`) ; it returns a structured report of what happened.
---
--- @param  ctx     ion7.core.Context
--- @param  seq_id  integer
--- @param  session ion7.llm.Session
--- @param  opts    table
---   `overflow`  (integer)              Tokens to free at minimum.
---   `prefix_n`  (integer, default 0)   Pinned-prefix length.
---   `n_sink`    (integer, default 4)   Attention-sink length.
---   `strategy`  (string, default `"message"`) `"message"` or `"fifo"`.
--- @return table
---   `freed`   (integer) Tokens removed from KV (0 when unable).
---   `dropped` (table[]) Messages removed (only when strategy=message).
function M.handle_overflow(ctx, seq_id, session, opts)
    local overflow  = opts.overflow or 0
    local prefix_n  = opts.prefix_n or 0
    local n_sink    = opts.n_sink   or 0
    local strategy  = opts.strategy or "message"
    local report    = { freed = 0, dropped = {} }

    if overflow <= 0 then return report end
    if not ctx:kv_can_shift() then return report end

    local keep_from = prefix_n + n_sink
    local movable   = session.n_past - keep_from
    if movable <= 0 then return report end

    if strategy == "message" and #session._msg_kv_ends > 0 then
        return M._evict_messages(ctx, seq_id, session, overflow, keep_from)
    end
    return M._evict_fifo(ctx, seq_id, session, overflow, keep_from, movable)
end

-- ── Message-aligned eviction ─────────────────────────────────────────────
--
-- Walks the per-message end-position list, accumulating message
-- boundaries until enough tokens are queued for removal. Then issues
-- one `kv_seq_rm` for the dropped span and one `kv_seq_shift` to
-- shift the survivors left. Returns the dropped message indices via
-- the report so the caller can plug an `on_evict` summarisation hook.
function M._evict_messages(ctx, seq_id, session, overflow, keep_from)
    local report   = { freed = 0, dropped = {} }
    local ends     = session._msg_kv_ends
    local freed    = 0
    local last_pos = keep_from
    local count    = 0

    for i = 1, #ends do
        local msg_end = ends[i]
        if msg_end <= keep_from then
            count    = i
            last_pos = msg_end
        else
            local start = (i == 1) and keep_from
                          or math.max(ends[i - 1], keep_from)
            freed    = freed + (msg_end - start)
            count    = i
            last_pos = msg_end
            if freed >= overflow then break end
        end
    end

    if freed == 0 then return report end

    -- Collect the message rows we are removing for the on_evict hook.
    local dropped = {}
    for i = 1, count do
        if session.messages[i] then
            dropped[#dropped + 1] = session.messages[i]
        end
    end

    -- Remove the evicted span, then shift the survivors left so KV
    -- positions stay contiguous.
    ctx:kv_seq_rm   (seq_id, keep_from, last_pos)
    ctx:kv_seq_shift(seq_id, -freed,    last_pos, -1)
    session.n_past = session.n_past - freed

    -- Rebuild _msg_kv_ends for the survivors.
    local new_ends = {}
    for i = count + 1, #ends do
        new_ends[#new_ends + 1] = ends[i] - freed
    end
    session._msg_kv_ends = new_ends

    -- Drop the evicted messages from the session's history.
    local new_msgs = {}
    for i = count + 1, #session.messages do
        new_msgs[#new_msgs + 1] = session.messages[i]
    end
    session.messages = new_msgs

    report.freed   = freed
    report.dropped = dropped
    return report
end

-- ── FIFO eviction ────────────────────────────────────────────────────────
--
-- Removes a contiguous head slice. Cheaper than the message walk
-- (no per-message bookkeeping needed) but produces mid-message cuts
-- — fine when the consumer values throughput over conversational
-- coherence (long-form streaming, dictation, log condensation).
function M._evict_fifo(ctx, seq_id, session, overflow, keep_from, movable)
    -- Free at least `overflow` ; rounds up to the nearest sensible
    -- chunk so we don't have to evict again two tokens later.
    local shift = math.min(math.max(overflow, math.floor(movable / 2)), movable)
    if shift <= 0 then return { freed = 0, dropped = {} } end

    ctx:kv_seq_rm   (seq_id, keep_from, keep_from + shift)
    ctx:kv_seq_shift(seq_id, -shift,    keep_from + shift, -1)
    session.n_past = session.n_past - shift

    -- Update _msg_kv_ends : drop entries that fell in the evicted
    -- window, shift the rest down. Survivors below `keep_from` mean
    -- the eviction crossed back into the pinned region — the
    -- corresponding messages are effectively gone too.
    local new_ends = {}
    for _, e in ipairs(session._msg_kv_ends) do
        local adj = e - shift
        if adj > keep_from then new_ends[#new_ends + 1] = adj end
    end
    session._msg_kv_ends = new_ends

    return { freed = shift, dropped = {} }
end

return M
