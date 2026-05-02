--- @module ion7.llm.kv.snapshot
--- @author  ion7 / Ion7 Project Contributors
---
--- Per-sequence snapshot helpers. Thin wrappers around ion7-core's
--- `Context:seq_snapshot` / `seq_restore` family, factored out so the
--- rest of the kv layer never touches the FFI surface directly.
---
--- All four operations work on ONE sequence at a time — restoring a
--- per-seq blob never touches another session's KV row, which is why
--- this is the foundation of correct multi-session behaviour.

local M = {}

--- Capture the KV state of `seq_id` into a Lua string.
---
--- @param  ctx    ion7.core.Context
--- @param  seq_id integer
--- @return string?  blob, or nil when the sequence is empty
function M.save(ctx, seq_id)
    return ctx:seq_snapshot(seq_id)
end

--- Restore the blob into `dst_seq_id`. The destination's previous
--- contents (if any) are wiped — call `kv_seq_rm` first if you need
--- to merge instead of replace.
---
--- @param  ctx        ion7.core.Context
--- @param  blob       string
--- @param  dst_seq_id integer
--- @return integer  bytes consumed (0 on failure)
function M.restore(ctx, blob, dst_seq_id)
    return ctx:seq_restore(blob, dst_seq_id)
end

--- Byte size the next snapshot of `seq_id` would produce. Useful for
--- pre-allocating buffers in a producer / consumer pipeline.
---
--- @param  ctx    ion7.core.Context
--- @param  seq_id integer
--- @return integer
function M.size(ctx, seq_id)
    return ctx:seq_state_size(seq_id)
end

--- File-backed save / load — drives `seq_save_state` /
--- `seq_load_state` so a session's KV can persist across process
--- restarts without going through a Lua-string blob.
---
--- Each helper returns the underlying call's success boolean.
---
--- @param  ctx    ion7.core.Context
--- @param  path   string
--- @param  seq_id integer
--- @return boolean
function M.save_file(ctx, path, seq_id)
    return ctx:seq_save_state(path, seq_id)
end

--- @param  ctx        ion7.core.Context
--- @param  path       string
--- @param  dst_seq_id integer
--- @return boolean
function M.load_file(ctx, path, dst_seq_id)
    return ctx:seq_load_state(path, dst_seq_id)
end

return M
