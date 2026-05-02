--- @module ion7.llm.kv.slots
--- @author  ion7 / Ion7 Project Contributors
---
--- Sequence-id allocator. Each user `Session` borrows one row (`seq_id`)
--- of the shared KV cache from this pool ; on release, the row is
--- wiped (`kv_seq_rm seq 0 -1`) and goes back into rotation.
---
--- A `reserved` slot can be excluded from rotation, which is how the
--- kv layer pins seq 0 for the system-prompt prefix when the context
--- supports more than one sequence.

local Slots = {}
Slots.__index = Slots

--- @class ion7.llm.kv.Slots
--- @field _ctx      ion7.core.Context
--- @field _free     boolean[]   `_free[i]` is true when seq `i` is available.
--- @field _reserved table       `_reserved[i] == true` for slots the allocator must skip.
--- @field _capacity integer     Total seqs the context supports.

--- Build a slot allocator over `[0, n_seq_max - 1]` of `ctx`.
---
--- @param  ctx ion7.core.Context
--- @param  opts table?
---   `reserved` (integer[] | integer?)  Slots the allocator must skip
---     when handing out seq_ids. Pass a single int or an array. Use
---     this to pin seq 0 for the prefix cache.
--- @return ion7.llm.kv.Slots
function Slots.new(ctx, opts)
    opts = opts or {}
    local capacity = tonumber(ctx:n_seq_max()) or 1

    local reserved = {}
    if type(opts.reserved) == "number" then
        reserved[opts.reserved] = true
    elseif type(opts.reserved) == "table" then
        for _, s in ipairs(opts.reserved) do reserved[s] = true end
    end

    local free = {}
    for i = 0, capacity - 1 do
        free[i] = not reserved[i]
    end

    return setmetatable({
        _ctx      = ctx,
        _free     = free,
        _reserved = reserved,
        _capacity = capacity,
    }, Slots)
end

--- Claim a free slot. Returns nil when every non-reserved seq is
--- already in use ; the caller decides whether to fail, evict, or
--- queue.
--- @return integer?
function Slots:acquire()
    for seq, free in pairs(self._free) do
        if free then
            self._free[seq] = false
            return seq
        end
    end
    return nil
end

--- Return a slot to the pool. The KV row is wiped before the slot
--- becomes available again so a stale prefill from the previous
--- tenant cannot leak into the new conversation.
--- @param  seq_id integer
function Slots:release(seq_id)
    if seq_id == nil then return end
    if self._reserved[seq_id] then return end
    self._ctx:kv_seq_rm(seq_id, 0, -1)
    self._free[seq_id] = true
end

--- Number of slots currently free (excludes reserved).
--- @return integer
function Slots:n_free()
    local n = 0
    for _, free in pairs(self._free) do
        if free then n = n + 1 end
    end
    return n
end

--- Total non-reserved capacity (the upper bound on
--- `Pool` concurrency).
--- @return integer
function Slots:capacity()
    return self._capacity - self:_n_reserved()
end

function Slots:_n_reserved()
    local n = 0
    for _ in pairs(self._reserved) do n = n + 1 end
    return n
end

--- True when `seq_id` is in the reserved set (e.g. the prefix slot).
--- @param  seq_id integer
--- @return boolean
function Slots:is_reserved(seq_id)
    return self._reserved[seq_id] == true
end

return Slots
