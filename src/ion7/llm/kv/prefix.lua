--- @module ion7.llm.kv.prefix
--- @author  ion7 / Ion7 Project Contributors
---
--- System-prompt prefix cache.
---
--- Two operating modes, picked automatically from the context's
--- `n_seq_max` :
---
---   - `n_seq_max >= 2` : seq 0 is reserved as the "prefix slot". The
---     system prompt is decoded into seq 0 once ; every new session
---     pulls a copy with `kv_seq_cp(0, target_seq, 0, n_prefix)` —
---     a memory-bandwidth operation, no model forward pass needed.
---     Re-prefilling 30 sessions from a long system prompt drops from
---     `30 × O(n_prefix × layers)` to one decode plus 30 fast copies.
---
---   - `n_seq_max == 1` : there is no spare row to reserve. The
---     prefix lives at the start of the single session's row ; we
---     just remember the prompt text + token count, and let the
---     session's own re-prefill take care of replaying it. The
---     prefix-aware path becomes a no-op.
---
--- The class is stateless about which sessions copied the prefix —
--- that bookkeeping lives in `kv/init`.

local Prefix = {}
Prefix.__index = Prefix

local PREFIX_SEQ = 0    -- pinned slot when reservation is possible

--- @class ion7.llm.kv.Prefix
--- @field _ctx      ion7.core.Context
--- @field _vocab    ion7.core.Vocab
--- @field _seq      integer?     Reserved seq_id, or nil when unavailable.
--- @field _text     string?      Cached prompt text (for `:set` short-circuit).
--- @field _n        integer      Token count of the encoded prompt.

--- Build a prefix manager.
---
--- @param  ctx   ion7.core.Context
--- @param  vocab ion7.core.Vocab
--- @return ion7.llm.kv.Prefix
function Prefix.new(ctx, vocab)
    local n_seq = tonumber(ctx:n_seq_max()) or 1
    return setmetatable({
        _ctx   = ctx,
        _vocab = vocab,
        _seq   = (n_seq >= 2) and PREFIX_SEQ or nil,
        _text  = nil,
        _n     = 0,
    }, Prefix)
end

--- True when this manager owns a real prefix slot (i.e. the context
--- supports `n_seq_max >= 2`). When false, `:copy_to` is a no-op and
--- callers should account for the prompt at session-prefill time.
--- @return boolean
function Prefix:has_slot()
    return self._seq ~= nil
end

--- The seq_id that holds the encoded prefix, or nil when the manager
--- runs in degraded mode (n_seq_max == 1).
--- @return integer?
function Prefix:seq_id()
    return self._seq
end

--- Encode `system_text` into the prefix slot. No-op when called with
--- the same text twice (the cache is idempotent on identical input).
---
--- Has no effect when the manager is in degraded mode — the prompt is
--- only remembered ; the session will re-encode it itself.
---
--- @param  system_text string
--- @return integer  Token count after encoding (0 in degraded mode).
function Prefix:set(system_text)
    assert(type(system_text) == "string",
        "[ion7.llm.kv.prefix] system_text must be a string")
    if self._text == system_text then return self._n end

    if not self._seq then
        -- Degraded mode : just remember the text. The session's
        -- prefill path will see `prefix:text()` non-nil and prepend
        -- the system message itself.
        self._text = system_text
        self._n    = 0
        return 0
    end

    local vocab = self._vocab
    local toks, n = vocab:tokenize(
        vocab:apply_template({ { role = "system", content = system_text } },
                             false, -1),
        false, true)

    -- Wipe the prefix slot, decode fresh.
    self._ctx:kv_seq_rm(self._seq, 0, -1)
    self._ctx:decode(toks, n, self._seq, 0)

    self._text = system_text
    self._n    = n
    return n
end

--- Copy the prefix into `dst_seq_id`. The destination's pre-existing
--- KV (if any) is wiped first so a stale prefix from a prior session
--- cannot leak. Returns the number of tokens copied (0 in degraded
--- mode).
---
--- @param  dst_seq_id integer
--- @return integer
function Prefix:copy_to(dst_seq_id)
    if not self._seq or self._n == 0 then return 0 end
    assert(dst_seq_id ~= self._seq,
        "[ion7.llm.kv.prefix] cannot copy prefix onto itself")
    self._ctx:kv_seq_rm(dst_seq_id, 0, -1)
    self._ctx:kv_seq_cp(self._seq, dst_seq_id, 0, self._n)
    return self._n
end

--- Token count of the encoded prefix. Always 0 in degraded mode.
--- @return integer
function Prefix:n_tokens()
    return self._n
end

--- Cached prompt text, or nil when never set.
--- @return string?
function Prefix:text()
    return self._text
end

--- Drop the cached prefix. Wipes the prefix slot's KV and zeroes the
--- token count ; subsequent `:set` calls will re-encode.
function Prefix:clear()
    if self._seq then
        self._ctx:kv_seq_rm(self._seq, 0, -1)
    end
    self._text = nil
    self._n    = 0
end

return Prefix
