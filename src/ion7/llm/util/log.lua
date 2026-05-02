--- @module ion7.llm.util.log
--- @author  ion7 / Ion7 Project Contributors
---
--- A tiny logging wrapper. Two modes :
---
---   - silent (default) — drops everything, costs an empty function call.
---   - active            — emits `[ion7.llm] level : message` to stderr.
---
--- The level filter mirrors `ion7.core.log` (0 = silent, 1 = error,
--- 2 = warn, 3 = info, 4 = debug). When `ion7.core.log` is loadable we
--- read its current level once at module load and re-export the four
--- helpers so callers do not have to know which back-end is wired in.
---
--- Usage :
---
---   local log = require "ion7.llm.util.log"
---   log.set_level(2)
---   log.warn("KV cache 80%% full ; consider session:reset()")

local M = { level = 0, _stream = io.stderr }

local LABELS = { [1] = "ERROR", [2] = "WARN", [3] = "INFO", [4] = "DEBUG" }

local function emit(level_int, msg)
    if level_int > M.level then return end
    M._stream:write(string.format("[ion7.llm] %s : %s\n",
        LABELS[level_int] or "?", msg))
    M._stream:flush()
end

--- Set the verbosity threshold. 0 silences every helper.
--- @param  n integer 0–4
function M.set_level(n)
    M.level = (type(n) == "number" and n >= 0 and n <= 4) and n or 0
end

--- Redirect output. Pass `io.stdout`, an open file handle, or any
--- table with a `write` and `flush` method.
--- @param  stream table
function M.set_stream(stream)
    if stream and type(stream.write) == "function" then
        M._stream = stream
    end
end

function M.error(msg) emit(1, msg) end
function M.warn (msg) emit(2, msg) end
function M.info (msg) emit(3, msg) end
function M.debug(msg) emit(4, msg) end

-- Mirror ion7.core.log's level when that module is reachable. Lets a
-- single `ion7.init({ log_level = 3 })` call propagate to both layers.
do
    local ok, core_log = pcall(require, "ion7.core.util.log")
    if ok and core_log and type(core_log.snapshot) == "function" then
        local snap = core_log.snapshot()
        if snap and type(snap.level) == "number" then
            M.level = snap.level
        end
    end
end

return M
