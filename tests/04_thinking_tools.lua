#!/usr/bin/env luajit
--- @module tests.04_thinking_tools
--- @author  ion7 / Ion7 Project Contributors
---
--- Pure-Lua coverage for the two state machines that wrap the engine's
--- per-token loop without touching the model :
---
---   - `chat.thinking` demultiplexes a streaming sequence into content
---     vs `<think>...</think>` reasoning. Tested below by feeding it
---     synthetic pieces — exact replicas of the splits BPE tokenisers
---     produce — and asserting the kind / text / suffix triples it
---     yields.
---
---   - `tools.spec` (Tool + ToolSet) describes the external actions the
---     model can request, plus the dispatch contract `tools.loop` uses
---     to invoke them. We exercise the validation, the JSON renderer,
---     and the safety contract `:dispatch` provides around handler
---     errors (a Lua throw must surface as `{ error = "..." }`, never
---     poison the engine loop).
---
--- Both modules are pure Lua so this file runs without a model.

local T = require "tests.framework"
require "tests.helpers"

local Thinking = require "ion7.llm.chat.thinking"
local Spec     = require "ion7.llm.tools.spec"
local Tool     = Spec.Tool
local ToolSet  = Spec.ToolSet

-- ════════════════════════════════════════════════════════════════════════
-- chat.thinking — `<think>...</think>` demultiplexing
-- ════════════════════════════════════════════════════════════════════════

T.suite("Thinking — initial state")

T.test("a fresh state is content-side and empty", function()
    local th = Thinking.new()
    T.eq(th:in_think(),            false)
    T.eq(th:active_token_count(),  0)
    T.eq(th:thinking(),            nil, "no thinking text yet")
end)

T.suite("Thinking — single-piece transitions (tag arrives whole)")

T.test("plain content stays on the content channel", function()
    local th = Thinking.new()
    local kind, text = th:feed("hello world")
    T.eq(kind, "content")
    T.eq(text, "hello world")
    T.eq(th:in_think(), false)
end)

T.test("a piece that contains the open tag splits into prefix + body", function()
    local th = Thinking.new()
    local kind, prefix, suffix = th:feed("hi <think>thought 1")
    T.eq(kind,   "split")
    T.eq(prefix, "hi ",         "content before <think> goes to the user")
    T.eq(suffix, "thought 1",   "after-tag body goes to the thinking channel")
    T.eq(th:in_think(),         true)
end)

T.test("a piece fully inside the block goes to thinking", function()
    local th = Thinking.new()
    th:feed("<think>")
    local kind, text = th:feed("more reasoning")
    T.eq(kind, "thinking")
    T.eq(text, "more reasoning")
end)

T.test("a piece carrying the close tag splits at the boundary", function()
    local th = Thinking.new()
    th:feed("<think>aha")
    local kind, before, after = th:feed("</think>back to user")
    T.eq(kind,   "split")
    T.eq(before, "",            "no thinking content remains in this piece")
    T.eq(after,  "back to user")
    T.eq(th:in_think(),         false)
    T.contains(th:thinking(),   "aha")
end)

T.suite("Thinking — multi-piece transitions (tag split across yields)")

T.test("the open tag split as '<th' + 'ink>' fires correctly", function()
    local th = Thinking.new()
    local k1, t1 = th:feed("hi <th")
    -- The piece ends with a partial tag — it must NOT leak as content,
    -- and the in-think state must NOT fire prematurely.
    T.eq(k1, "content")
    T.eq(th:in_think(), false)
    -- The next piece completes the tag.
    local k2, prefix, suffix = th:feed("ink>thinking starts")
    T.eq(k2,             "split")
    T.eq(suffix,         "thinking starts",
         "post-tag body should arrive on the thinking channel")
    T.eq(th:in_think(),  true)
end)

T.test("the close tag split as '</thi' + 'nk>' fires correctly", function()
    local th = Thinking.new()
    th:feed("<think>thinking ")
    local k1, t1 = th:feed("</thi")
    T.eq(k1, "thinking",
         "partial close-tag bytes must still arrive on the thinking channel")
    local k2, before, after = th:feed("nk>back")
    T.eq(k2,             "split")
    T.eq(after,          "back")
    T.eq(th:in_think(),  false)
end)

T.suite("Thinking — accumulator + budget helpers")

T.test("thinking() reports closed + active text", function()
    local th = Thinking.new()
    th:feed("<think>round1</think>...")
    th:feed("<think>round2 ")
    -- After two rounds : "round1" closed, "round2 " still active.
    local txt = th:thinking()
    T.contains(txt, "round1")
    T.contains(txt, "round2")
end)

T.test("active_token_count tracks pieces inside the active block", function()
    local th = Thinking.new()
    th:feed("<think>")
    T.eq(th:active_token_count(), 1, "the open-tag piece counts as one")
    th:feed("more")
    T.eq(th:active_token_count(), 2)
    th:feed("</think>")
    T.eq(th:active_token_count(), 0, "counter resets on close")
end)

T.test("force_close drops the active block and archives its body", function()
    local th = Thinking.new()
    th:feed("<think>incomplete reasoning")
    T.eq(th:in_think(), true)
    local body = th:force_close()
    T.contains(body,           "incomplete")
    T.eq(th:in_think(),        false)
    T.eq(th:active_token_count(), 0)
    T.contains(th:thinking(),  "incomplete",
        "force-closed body must remain in the archive")
end)

T.test("reset clears every channel and returns the accumulated thinking", function()
    local th = Thinking.new()
    th:feed("<think>foo</think>")
    local out = th:reset()
    T.contains(out, "foo")
    T.eq(th:thinking(),           nil)
    T.eq(th:active_token_count(), 0)
end)

-- ════════════════════════════════════════════════════════════════════════
-- tools.spec — Tool + ToolSet
-- ════════════════════════════════════════════════════════════════════════

T.suite("Tool — construction + validation")

T.test("Tool.new requires a non-empty name", function()
    T.err(function() Tool.new() end,                "table")
    T.err(function() Tool.new({}) end,              "name")
    T.err(function() Tool.new({ name = "" }) end,   "name")
end)

T.test("Tool.new fills sensible defaults for missing fields", function()
    local t = Tool.new({ name = "noop" })
    T.eq(t.name,        "noop")
    T.eq(t.description, "")
    T.is_type(t.schema, "table")
    T.eq(t.schema.type, "object")
end)

T.test("to_template_entry produces the OpenAI/modern function shape", function()
    local t = Tool.new({
        name        = "search",
        description = "Web search",
        schema      = {
            type       = "object",
            properties = { q = { type = "string" } },
            required   = { "q" },
        },
    })
    local entry = t:to_template_entry()
    T.eq(entry.type,                       "function")
    T.eq(entry["function"].name,           "search")
    T.eq(entry["function"].description,    "Web search")
    T.eq(entry["function"].parameters.type, "object")
end)

T.test("dispatch raises when no handler is attached", function()
    local t = Tool.new({ name = "noop" })
    T.err(function() t:dispatch({}) end, "no handler")
end)

T.test("dispatch wraps a Lua throw into a structured error result", function()
    local t = Tool.new({
        name    = "boom",
        handler = function() error("kaboom") end,
    })
    local r = t:dispatch({})
    T.is_type(r,    "table")
    T.is_type(r.error, "string")
    T.contains(r.error, "kaboom")
end)

T.test("dispatch returns the handler value on success", function()
    local t = Tool.new({
        name    = "double",
        handler = function(args) return args.n * 2 end,
    })
    T.eq(t:dispatch({ n = 21 }), 42)
end)

T.suite("ToolSet — registry semantics")

T.test("ToolSet.new() starts empty", function()
    T.eq(ToolSet.new():count(), 0)
end)

T.test("ToolSet.new(list) bulk-loads the initial population", function()
    local set = ToolSet.new({
        Tool.new({ name = "a" }),
        Tool.new({ name = "b" }),
    })
    T.eq(set:count(),        2)
    T.eq(set:find("a").name, "a")
    T.eq(set:find("b").name, "b")
    T.eq(set:find("c"),      nil)
end)

T.test("add raises on a non-Tool", function()
    local set = ToolSet.new()
    T.err(function() set:add({ name = "fake" }) end, "Tool instance")
end)

T.test("add rejects duplicate names", function()
    local set = ToolSet.new()
    set:add(Tool.new({ name = "x" }))
    T.err(function() set:add(Tool.new({ name = "x" })) end, "duplicate")
end)

T.test("iter walks tools in registration order", function()
    local set = ToolSet.new({
        Tool.new({ name = "first"  }),
        Tool.new({ name = "second" }),
        Tool.new({ name = "third"  }),
    })
    local names = {}
    for t in set:iter() do names[#names + 1] = t.name end
    T.eq(names[1], "first")
    T.eq(names[2], "second")
    T.eq(names[3], "third")
end)

T.test("to_json emits a JSON-encoded array of function entries", function()
    local set = ToolSet.new({
        Tool.new({
            name        = "search",
            description = "Web search",
            schema      = { type = "object", properties = {} },
        }),
    })
    local s = set:to_json()
    T.is_type(s, "string")
    T.contains(s, '"type":"function"')
    T.contains(s, '"name":"search"')
    T.contains(s, '"description":"Web search"')
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
