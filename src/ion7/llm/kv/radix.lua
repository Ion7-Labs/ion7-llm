--- @module ion7.llm.kv.radix
--- @author  ion7 / Ion7 Project Contributors
---
--- RadixAttention-style prefix cache. Indexes per-seq KV snapshots
--- by their full token sequence so a session that runs the SAME
--- prompt as a recent sibling can warm-start from the sibling's
--- blob instead of re-encoding it.
---
--- Inspired by SGLang's RadixAttention (Zheng et al., 2024). This
--- implementation is the SIMPLE variant : one blob per terminal node,
--- no block-level KV sharing. That covers the high-value cases :
---
---   - **Exact-match re-run** : agent loops that re-issue the same
---     prompt, A/B testing with the same head, regeneration after a
---     reject. The cached blob comes back to life in the time it
---     takes to memcpy ~tens of MB.
---   - **Multi-turn conversation continuation** : a follow-up turn
---     whose new tail matches the cached state at the boundary.
---
--- What it does NOT do :
---
---   - **Partial-match warm-start.** A blob captured at position N
---     describes the KV after decoding tokens[1..N]. Restoring it for
---     a prompt that diverges at position M < N would put tokens at
---     positions M+1..N in the KV that the new prompt has not seen —
---     incorrect generation. A correct partial-match cache requires
---     block-level KV sharing (vLLM PagedAttention). That is a v3
---     project ; until then we report no-match for non-exact prefixes
---     to stay correct.
---
--- Storage shape :
---
---   - Trie partitions any token sequence into edges. The node at the
---     end of an edge optionally pins ONE KV blob.
---   - Nodes are cheap (a Lua table with a few fields). KV blobs are
---     NOT — each one is `n_layer · n_kv_heads · head_dim · 2 ·
---     n_tokens · sizeof(kv_type)` bytes. The cache enforces a
---     `max_blobs` LRU cap and evicts the least-recently-used blob
---     when a new insert would breach it.
---
--- Use cases :
---   - System-prompt sharing across thousands of sessions (`kv.Prefix`
---     already handles the system-only case ; the radix extends it
---     to longer common stems).
---   - Agent loops re-issuing identical sub-prompts.
---   - Forked sessions returning to a parent's state.

local M = {}

local Cache = {}
Cache.__index = Cache

-- ── Trie node ────────────────────────────────────────────────────────────

-- @class ion7.llm.kv.radix.Node
-- @field tokens   integer[]   Edge labels (chunk of token IDs leading
--                              to THIS node from its parent).
-- @field children table       `{ [first_tok_of_chunk] = child_node, ... }`
-- @field parent   table?      Backref for `:gc()` pruning.
-- @field n_tokens integer     Cumulative tokens from the root to this node.
-- @field blob     string?     Snapshot blob pinned at this prefix, or nil.
-- @field last_used integer    Monotonic counter for LRU.
local function new_node(tokens, parent, n_tokens)
    return {
        tokens    = tokens,
        children  = {},
        parent    = parent,
        n_tokens  = n_tokens,
        blob      = nil,
        last_used = 0,
    }
end

-- ── Cache constructor ────────────────────────────────────────────────────

--- Build an empty radix cache.
---
--- @param  opts table?
---   `chunk_size` (integer, default 32) Tokens per trie edge.
---   `max_blobs`  (integer, default 64) LRU cap on stored snapshots.
---   `min_match`  (integer, default 64) Minimum prefix length below
---       which `find_longest_prefix` reports "no useful match" ; the
---       caller skips the warm-start and falls back to the slow path.
---       Avoids the case where restoring 8 tokens costs more than
---       just decoding them.
--- @return ion7.llm.kv.radix.Cache
function M.new(opts)
    opts = opts or {}
    return setmetatable({
        _root       = new_node({}, nil, 0),
        _chunk_size = opts.chunk_size or 32,
        _max_blobs  = opts.max_blobs  or 64,
        _min_match  = opts.min_match  or 64,
        _n_blobs    = 0,
        _clock      = 0,
        _stats      = {
            hits         = 0,
            misses       = 0,
            blobs_evicted = 0,
        },
    }, Cache)
end

-- ── Internal helpers ─────────────────────────────────────────────────────

-- Compare two integer arrays up to `n` ; returns the count of equal
-- leading entries.
local function shared_prefix_len(a, b, n)
    for i = 1, n do
        if a[i] ~= b[i] then return i - 1 end
    end
    return n
end

-- Bump the LRU clock and stamp a node.
local function touch(self, node)
    self._clock = self._clock + 1
    node.last_used = self._clock
end

-- Walk the trie until we run out of matching tokens. Returns the
-- deepest matched node + the number of input tokens consumed.
local function descend(self, tokens, n_tokens)
    local node = self._root
    local pos  = 0
    while pos < n_tokens do
        local first = tokens[pos + 1]
        local child = node.children[first]
        if not child then break end

        -- Shared prefix between input[pos+1..] and child.tokens.
        local edge = child.tokens
        local edge_len = #edge
        local input_left = n_tokens - pos
        local probe = edge_len < input_left and edge_len or input_left
        local matched = 0
        for i = 1, probe do
            if tokens[pos + i] ~= edge[i] then break end
            matched = i
        end

        if matched < edge_len then
            -- Partial edge match — we cannot descend into this child
            -- without splitting the edge. For lookup, we report the
            -- partial match by returning the parent node and the
            -- partial token count. Splits only happen on insert.
            return node, pos, child, matched
        end

        node = child
        pos  = pos + matched
    end
    return node, pos, nil, 0
end

-- Split an edge : turn (parent → child via edge of length E) into
-- (parent → mid via edge[1..k] → child via edge[k+1..E]). The
-- caller passes the parent pointer + the matched length k. Returns
-- the new mid node.
local function split_edge(parent, child, k)
    local edge = child.tokens
    local edge_len = #edge

    -- Extract the two halves.
    local head, tail = {}, {}
    for i = 1, k       do head[i]       = edge[i]       end
    for i = k + 1, edge_len do tail[i - k] = edge[i] end

    -- Replace `child` with a new mid node holding `head`.
    local mid = new_node(head, parent, child.n_tokens - (edge_len - k))
    parent.children[head[1]] = mid

    -- Reparent `child` under `mid` with the tail label.
    child.tokens   = tail
    child.parent   = mid
    mid.children[tail[1]] = child

    return mid
end

-- Evict the least-recently-used blob until we are below `max_blobs`.
-- Cheap walk — number of blobs is bounded and small (typically ≤ 64).
local function evict_lru(self)
    while self._n_blobs >= self._max_blobs do
        local victim, victim_node = math.huge, nil
        local function walk(node)
            if node.blob and node.last_used < victim then
                victim, victim_node = node.last_used, node
            end
            for _, c in pairs(node.children) do walk(c) end
        end
        walk(self._root)
        if not victim_node then return end
        victim_node.blob = nil
        victim_node.last_used = 0
        self._n_blobs = self._n_blobs - 1
        self._stats.blobs_evicted = self._stats.blobs_evicted + 1
    end
end

-- ── Public API ───────────────────────────────────────────────────────────

--- Find a cached prefix of `tokens` that is SAFE to restore.
---
--- The match must be exact at the cumulative position the blob was
--- saved at — restoring a longer-than-prompt blob would put unseen
--- tokens in the KV ; restoring a partial would require block-level
--- sharing we do not (yet) implement. So this function only reports
--- a hit when the input tokens at positions `1..blob_pos` exactly
--- match the path the blob was inserted on AND `blob_pos` is the
--- entire input length we have.
---
--- @param  tokens   table     1-based array of token IDs.
--- @param  n_tokens integer
--- @return integer, string?   `(n_matched, blob)` or `(0, nil)`.
function Cache:find_longest_prefix(tokens, n_tokens)
    n_tokens = n_tokens or #tokens
    if n_tokens == 0 then return 0, nil end

    -- Walk down. We only accept a result when:
    --   1. the descent consumed every input token (full path match),
    --   2. the landing node has a blob,
    --   3. the matched length passes `min_match`.
    -- This keeps us correct without block-level KV sharing.
    local node, pos = descend(self, tokens, n_tokens)

    if pos == n_tokens and node.blob and pos >= self._min_match then
        self._stats.hits = self._stats.hits + 1
        touch(self, node)
        return pos, node.blob
    end

    self._stats.misses = self._stats.misses + 1
    return 0, nil
end

--- Insert (or refresh) a KV blob at the prefix `tokens[1..n_tokens]`.
---
--- Splits trie edges as needed so the blob lands at exactly the
--- requested boundary. If the resulting node already had a blob, it
--- is overwritten (same prefix, fresher state).
---
--- @param  tokens   table
--- @param  n_tokens integer
--- @param  blob     string  Result of `kv.snapshot.save(ctx, seq_id)`.
function Cache:insert(tokens, n_tokens, blob)
    n_tokens = n_tokens or #tokens
    if n_tokens == 0 or not blob then return end

    -- Walk down, splitting edges and creating children as needed.
    local node = self._root
    local pos  = 0
    while pos < n_tokens do
        local first = tokens[pos + 1]
        local child = node.children[first]
        if not child then
            -- Append a fresh edge with the entire remainder. Single
            -- node, single edge — most common case for first inserts.
            local left = n_tokens - pos
            local edge = {}
            for i = 1, left do edge[i] = tokens[pos + i] end
            local fresh = new_node(edge, node, n_tokens)
            node.children[first] = fresh
            node = fresh
            pos  = n_tokens
            break
        end

        local edge = child.tokens
        local edge_len = #edge
        local input_left = n_tokens - pos
        local probe = edge_len < input_left and edge_len or input_left
        local matched = 0
        for i = 1, probe do
            if tokens[pos + i] ~= edge[i] then break end
            matched = i
        end

        if matched == edge_len then
            -- Full edge match : descend.
            node = child
            pos  = pos + edge_len
        elseif matched == input_left then
            -- Input fully consumed, but the edge is longer. Split so
            -- our boundary lands at `pos + matched`.
            node = split_edge(node, child, matched)
            pos  = n_tokens
        else
            -- Partial match in both directions. Split, then create a
            -- new sibling child for the diverging tail.
            local mid = split_edge(node, child, matched)
            local left = n_tokens - (pos + matched)
            local fresh_edge = {}
            for i = 1, left do fresh_edge[i] = tokens[pos + matched + i] end
            local fresh = new_node(fresh_edge, mid, n_tokens)
            mid.children[fresh_edge[1]] = fresh
            node = fresh
            pos  = n_tokens
        end
    end

    -- LRU bookkeeping. Increment counter ONLY when a fresh blob is
    -- pinned (overwrite of the same node does not bump _n_blobs).
    evict_lru(self)
    if not node.blob then
        self._n_blobs = self._n_blobs + 1
    end
    node.blob = blob
    touch(self, node)
end

--- Drop every blob and every node ; reset the cache to empty.
function Cache:clear()
    self._root = new_node({}, nil, 0)
    self._n_blobs = 0
    self._clock   = 0
end

--- Garbage-collect : drop any subtree that contains no blobs. Useful
--- after heavy LRU eviction churn to keep the trie compact.
function Cache:gc()
    local function has_blob(node)
        if node.blob then return true end
        for _, c in pairs(node.children) do
            if has_blob(c) then return true end
        end
        return false
    end
    local function prune(node)
        for k, c in pairs(node.children) do
            if has_blob(c) then prune(c)
            else node.children[k] = nil end
        end
    end
    prune(self._root)
end

--- @return table { hits, misses, blobs_evicted, n_blobs, max_blobs }
function Cache:stats()
    return {
        hits          = self._stats.hits,
        misses        = self._stats.misses,
        blobs_evicted = self._stats.blobs_evicted,
        n_blobs       = self._n_blobs,
        max_blobs     = self._max_blobs,
    }
end

return M
