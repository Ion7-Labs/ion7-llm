#!/usr/bin/env luajit
--- @module tests.15_tools_loop
--- @author  ion7 / Ion7 Project Contributors
---
--- Tool-call loop : `tools.spec` (Tool / ToolSet) +
--- `tools.loop` (interleaved-thinking-aware dispatcher).
---
--- The pure-Lua tool spec is fully covered by `04_thinking_tools.lua`.
--- The here-and-now extras :
---
---   - `Engine:chat({ tools = ... })` on a non-tool-emitting model :
---     verify the call still completes with a plain `Response` and
---     does NOT crash the engine. The tools description is rendered
---     into the system prompt via the one-shot before_encode hook.
---
---   - `tools.loop.run` with a stub engine that returns a tool_call
---     once and a content turn after : we verify the dispatcher
---     calls the tool's handler, appends the tool_result message back
---     to the session, and re-issues the chat call.
---
--- A 3 B model rarely emits the JSON-tool-call format on its own, so
--- the dispatcher exercise uses a fake `engine` table that stubs the
--- two relevant methods. That avoids flake without losing coverage of
--- the dispatch invariants the loop relies on.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm, ctx, vocab, cm, engine = H.pipeline(model, {
    n_ctx     = 4096,
    n_seq_max = 4,
    headroom  = 256,
})

cm:set_system("You are a concise assistant.")

local Tool    = llm.tools.Tool
local ToolSet = llm.tools.ToolSet
local Response = llm.Response

-- ════════════════════════════════════════════════════════════════════════
-- engine:chat({ tools = ... }) on a real (non-tool) model
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:chat — tools description injection survives a real call")

T.test("a 3 B model with tools defined still returns a normal Response", function()
    local set = ToolSet.new({
        Tool.new({
            name        = "get_weather",
            description = "Return the current weather for a city.",
            schema      = {
                type       = "object",
                properties = { city = { type = "string" } },
                required   = { "city" },
            },
            handler     = function(args) return { city = args.city, temp = 21 } end,
        }),
    })

    local s = llm.Session.new()
    s:add_user("Reply with: hello.")
    local r = engine:chat(s, {
        max_tokens = 16,
        tools      = set,
    })
    T.is_type(r,           "table")
    T.is_type(r.content,   "string")
    T.gt(r.n_tokens,       0)

    cm:release(s)
end)

-- ════════════════════════════════════════════════════════════════════════
-- tools.loop.run — dispatcher contract (stub engine)
-- ════════════════════════════════════════════════════════════════════════

T.suite("tools.loop — dispatcher contract using a stub engine")

-- Build a fake engine whose chat() method emits a tool call on the
-- first turn and a content-only turn on the second. That isolates
-- the dispatcher's behaviour from the model's reliability.
local function make_stub_engine(turns)
    local i = 0
    return {
        chat = function(_, sess, opts)
            i = i + 1
            local turn = turns[i]
            -- Persist the assistant turn ourselves if it carries a content
            -- part — the dispatcher only handles tool turns.
            return turn
        end,
    }
end

T.test("a tool call gets dispatched and the result feeds the next turn", function()
    local invoked = false
    local set = ToolSet.new({
        Tool.new({
            name        = "tick",
            description = "no-op",
            schema      = { type = "object", properties = {} },
            handler     = function(args)
                invoked = true
                return { value = (args and args.x) or 42 }
            end,
        }),
    })

    -- Two-turn script :
    --   1) assistant says "let me check" + emits tool_call(tick, x=7),
    --   2) assistant gives a content-only turn.
    local turns = {
        Response.new({
            content    = "let me check",
            tool_calls = {
                { id = "0", name = "tick", arguments = { x = 7 } },
            },
            stop_reason = "tool_use",
        }),
        Response.new({
            content     = "all done.",
            stop_reason = "stop",
        }),
    }

    local stub = make_stub_engine(turns)
    local sess = llm.Session.new()
    sess:add_user("kick off the tools loop")

    local final, n_turns = llm.tools.loop(stub, sess, {
        tools     = set,
        max_turns = 4,
    })

    T.eq(invoked,           true,  "the tool handler must be invoked")
    T.eq(n_turns,           2,     "loop should run two turns (tool-then-content)")
    T.eq(final.content,     "all done.")
    T.eq(final.stop_reason, "stop")
    -- The loop persists the tool turn AND the tool result in the session.
    -- After turn 1 : add_assistant + add_tool_result ; turn 2 emits a
    -- content-only response which the loop returns WITHOUT appending.
    -- Sequence : user, assistant(tool_calls), tool, [no add for final].
    T.eq(#sess.messages,    3)
    T.eq(sess.messages[1].role, "user")
    T.eq(sess.messages[2].role, "assistant")
    T.eq(sess.messages[2].tool_calls and #sess.messages[2].tool_calls, 1)
    T.eq(sess.messages[3].role, "tool")
end)

T.test("on_call hook runs after every dispatch", function()
    local set = ToolSet.new({
        Tool.new({
            name    = "noop",
            schema  = { type = "object", properties = {} },
            handler = function() return { ok = true } end,
        }),
    })
    local hits = {}
    local turns = {
        Response.new({
            content    = "x",
            tool_calls = { { id = "0", name = "noop", arguments = {} } },
            stop_reason = "tool_use",
        }),
        Response.new({ content = "done", stop_reason = "stop" }),
    }
    local stub = make_stub_engine(turns)

    llm.tools.loop(stub, llm.Session.new(), {
        tools   = set,
        on_call = function(call, result)
            hits[#hits + 1] = { call = call, result = result }
        end,
    })
    T.eq(#hits, 1)
    T.eq(hits[1].call.name,   "noop")
    T.eq(hits[1].result.ok,   true)
end)

T.test("an unknown tool name yields a structured error result", function()
    local set = ToolSet.new() -- no tools registered
    local turns = {
        Response.new({
            content    = "x",
            tool_calls = { { id = "0", name = "ghost", arguments = {} } },
            stop_reason = "tool_use",
        }),
        Response.new({ content = "done", stop_reason = "stop" }),
    }
    local stub = make_stub_engine(turns)
    local sess = llm.Session.new()
    llm.tools.loop(stub, sess, { tools = set })

    -- The third message is the tool-result entry produced by the loop.
    local result = sess.messages[#sess.messages]
    T.eq(result.role,               "tool")
    -- Content was JSON-encoded by add_tool_result.
    T.contains(result.content,      "unknown tool")
end)

-- ════════════════════════════════════════════════════════════════════════
-- max_turns guard
-- ════════════════════════════════════════════════════════════════════════

T.suite("tools.loop — max_turns guard")

T.test("a tool-spamming model is bounded by max_turns", function()
    local set = ToolSet.new({
        Tool.new({
            name = "spam", schema = { type = "object", properties = {} },
            handler = function() return {} end,
        }),
    })
    -- Always return a tool call ; this should hit the cap.
    local turns = {}
    for i = 1, 20 do
        turns[i] = Response.new({
            content     = "x",
            tool_calls  = { { id = tostring(i), name = "spam", arguments = {} } },
            stop_reason = "tool_use",
        })
    end
    -- After max_turns, the loop falls through to a final chat() call —
    -- which still consumes one more entry from our script.
    turns[#turns + 1] = Response.new({ content = "stop", stop_reason = "stop" })

    local stub = make_stub_engine(turns)
    local _, n = llm.tools.loop(stub, llm.Session.new(), {
        tools = set, max_turns = 3,
    })
    T.eq(n, 3, "max_turns = 3 must cap the loop at 3 dispatch rounds")
end)

ctx:free()
local ok = T.summary()
os.exit(ok and 0 or 1)
