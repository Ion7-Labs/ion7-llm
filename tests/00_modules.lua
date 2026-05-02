#!/usr/bin/env luajit
--- @module tests.00_modules
--- @author  ion7 / Ion7 Project Contributors
---
--- Verify that every Lua module under `src/ion7/llm/` loads cleanly,
--- WITHOUT depending on a built `libllama.so` or the `ion7_bridge.so`.
---
--- Same idea as ion7-core's `00_modules.lua` : we monkey-patch
--- `ffi.load` and pre-register a bridge stub in `package.preload`,
--- then require every file under the source tree. Catches require
--- typos, syntax errors, missing FFI symbols a happy-path smoke test
--- might never hit.
---
--- Exit status :
---   0 — every module loaded.
---   1 — at least one require failed.

local T = require "tests.framework"
require "tests.helpers"

T.suite("Module load — every file under src/ion7/llm/ requires cleanly")

-- ── Stub the FFI surface ────────────────────────────────────────────────

local ffi = require "ffi"

local _stub = setmetatable({}, {
    __index = function() return function() end end,
    __call  = function() return nil end,
})

ffi.load = function() return _stub end

-- ion7-core's own ffi/bridge is loaded transitively by chat/parse,
-- sampler/schema, etc. Pre-register the stub so requires there resolve
-- without actually opening the .so.
package.preload["ion7.core.ffi.bridge"] = function() return _stub end

-- ── Walk the src tree ────────────────────────────────────────────────────

local function find_lua(dir)
    local out = {}
    local p = io.popen("find " .. dir .. " -type f -name '*.lua' -not -path '*/.old/*'")
    if not p then return out end
    for line in p:lines() do out[#out + 1] = line end
    p:close()
    return out
end

local files = find_lua("./src/ion7/llm")
table.sort(files)

T.test("at least one module discovered (sanity check)", function()
    T.gt(#files, 0, "find returned no .lua files — wrong CWD?")
end)

-- ── Try to require each file ────────────────────────────────────────────

for _, file in ipairs(files) do
    local mod = file
        :gsub("^%./src/", "")
        :gsub("%.lua$",   "")
        :gsub("/",        ".")
        :gsub("%.init$",  "")

    T.test("require '" .. mod .. "'", function()
        local ok, err = pcall(require, mod)
        if not ok then error(tostring(err):sub(1, 400), 0) end
    end)
end

-- ── Façade walk ─────────────────────────────────────────────────────────
--
-- The lazy façade in init.lua exposes both classes (Engine, Pool,
-- Session, Response, Embed, Stop) and sub-namespaces (chat, sampler,
-- tools, kv, util). Touch every advertised entry to verify the lazy
-- requires resolve.

T.suite("Façade — lazy class registry")

local llm = require "ion7.llm"

for _, name in ipairs({ "Engine", "Pool", "Session", "Response", "Embed", "Stop" }) do
    T.test("llm." .. name .. " resolves", function()
        T.is_type(llm[name], "table", "expected a class table for " .. name)
    end)
end

T.suite("Façade — sub-namespaces")

T.test("llm.chat exposes Template/Thinking/parse/stream", function()
    T.is_type(llm.chat.Template, "table")
    T.is_type(llm.chat.Thinking, "table")
    T.is_type(llm.chat.parse,    "table")
    T.is_type(llm.chat.stream,   "table")
end)

T.test("llm.sampler exposes profiles/budget (schema lives in ion7-grammar)", function()
    T.is_type(llm.sampler.profiles, "table")
    T.is_type(llm.sampler.budget,   "table")
    T.eq(llm.sampler.schema, nil,
        "schema-aware samplers belong in ion7-grammar")
end)

T.test("llm.tools exposes Tool/ToolSet/loop", function()
    T.is_type(llm.tools.Tool,    "table")
    T.is_type(llm.tools.ToolSet, "table")
    T.is_type(llm.tools.loop,    "function")
end)

T.test("llm.kv exposes ContextManager + helpers + callable shortcut", function()
    T.is_type(llm.kv.ContextManager, "table")
    T.is_type(llm.kv.new,            "function")
    T.is_type(llm.kv.snapshot,       "table")
    T.is_type(llm.kv.eviction,       "table")
end)

T.test("llm.util exposes messages/partial_json/log", function()
    T.is_type(llm.util.messages,     "table")
    T.is_type(llm.util.partial_json, "table")
    T.is_type(llm.util.log,          "table")
end)

T.test("llm.VERSION is a non-empty string", function()
    T.is_type(llm.VERSION, "string")
    T.gt(#llm.VERSION, 0)
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
