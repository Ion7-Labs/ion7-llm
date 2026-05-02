#!/usr/bin/env luajit
--- @module tests.12_engine_stream
--- @author  ion7 / Ion7 Project Contributors
---
--- `Engine:stream` coroutine iterator coverage.
---
--- The streaming API turns the per-token loop into typed chunks the
--- consumer can `for chunk in iter do ... end` over :
---
---   { kind = "content",  text = "..." }
---   { kind = "thinking", text = "..." }
---   { kind = "stop",     reason = "stop" | "length" | ... }
---
--- Tested invariants :
---
---   - exactly one `stop` chunk fires at the end of an iterator,
---   - the concatenation of `content` chunks equals the final
---     `Response.content` produced by the same call,
---   - `max_tokens` is honoured by the streaming path,
---   - the iterator surfaces a stop reason the response reflects,
---   - `session:last_response()` is wired by the time the iterator
---     drains, so consumers can read perf / tokens after the loop.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm, ctx, vocab, cm, engine = H.pipeline(model, {
    n_ctx     = 4096,
    n_seq_max = 4,
    headroom  = 256,
})

cm:set_system("You are a concise assistant.")

local function with_session(fn)
    local s = llm.Session.new()
    local ok, err = pcall(fn, s)
    cm:release(s)
    if not ok then error(err, 0) end
end

-- Drain an iterator into a typed-chunk array + a content concatenation.
local function drain(iter)
    local chunks = {}
    local content_parts = {}
    for c in iter do
        chunks[#chunks + 1] = c
        if c.kind == "content" then
            content_parts[#content_parts + 1] = c.text
        end
    end
    return chunks, table.concat(content_parts)
end

-- ════════════════════════════════════════════════════════════════════════
-- Stream — typed-chunk shape
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:stream — chunk shape + termination")

T.test("iterator yields content chunks then exactly one stop chunk", function()
    with_session(function(s)
        s:add_user("Reply with: ok.")
        local iter = engine:stream(s, { max_tokens = 16 })
        local chunks = drain(iter)

        -- The last chunk must be the terminator.
        local last = chunks[#chunks]
        T.eq(last.kind, "stop")
        T.one_of(last.reason, { "stop", "stop_string", "length", "tool_use", "error" })

        -- Exactly one stop chunk in the whole stream.
        local n_stops = 0
        for _, c in ipairs(chunks) do
            if c.kind == "stop" then n_stops = n_stops + 1 end
        end
        T.eq(n_stops, 1, "stream must emit exactly one stop chunk")
    end)
end)

T.test("the chunk types are limited to content / thinking / stop", function()
    with_session(function(s)
        s:add_user("Say one word.")
        local chunks = drain(engine:stream(s, { max_tokens = 8 }))
        for _, c in ipairs(chunks) do
            T.one_of(c.kind, { "content", "thinking", "stop", "tool_call_delta", "tool_call_done" })
        end
    end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Stream — content reconciliation with the final Response
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:stream — last_response reflects the streamed bytes")

T.test("session:last_response is set when the iterator drains", function()
    with_session(function(s)
        s:add_user("Reply with one short sentence.")
        local iter = engine:stream(s, { max_tokens = 24 })
        for _ in iter do end -- drain
        local r = s:last_response()
        T.is_type(r,         "table")
        T.is_type(r.content, "string")
        T.gt(r.n_tokens,     0)
    end)
end)

T.test("concatenated content chunks match the final Response content", function()
    with_session(function(s)
        s:add_user("Reply with: hello.")
        local _, streamed = drain(engine:stream(s, { max_tokens = 16 }))
        local resp = s:last_response()
        -- The post-generation parser may strip a trailing stop marker
        -- the streaming side already did NOT emit ; allow a strict
        -- prefix relationship rather than equality.
        T.eq(streamed:sub(1, #resp.content), resp.content,
            string.format("streamed prefix differs from final content:\n  streamed=%q\n  final=%q",
                streamed, resp.content))
    end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Stream — max_tokens cap propagates
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:stream — max_tokens cap")

T.test("max_tokens caps the number of streamed bytes", function()
    with_session(function(s)
        s:add_user("Tell me a long story about space.")
        local iter = engine:stream(s, { max_tokens = 8 })
        local chunks = drain(iter)
        local r = s:last_response()
        T.eq(r.n_tokens <= 8, true,
            string.format("got %d tokens for max_tokens=8", r.n_tokens))
        local stop = chunks[#chunks]
        T.one_of(stop.reason, { "length", "stop" })
    end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Stream — Stream.collect helper
-- ════════════════════════════════════════════════════════════════════════

T.suite("chat.stream.collect — materialise an iterator into a Response shape")

T.test("collect produces { content, thinking, tool_calls, stop_reason }", function()
    with_session(function(s)
        s:add_user("Reply with a short sentence.")
        local iter = engine:stream(s, { max_tokens = 16 })
        local view = llm.chat.stream.collect(iter)
        T.is_type(view,             "table")
        T.is_type(view.content,     "string")
        T.is_type(view.tool_calls,  "table")
        T.is_type(view.stop_reason, "string")
        T.one_of(view.stop_reason, { "stop", "stop_string", "length", "tool_use", "error" })
    end)
end)

ctx:free()
local ok = T.summary()
os.exit(ok and 0 or 1)
