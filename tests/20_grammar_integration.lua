#!/usr/bin/env luajit
--- @module tests.20_grammar_integration
--- @author  ion7 / Ion7 Project Contributors
---
--- Cross-module smoke test : ion7-core + ion7-llm + ion7-grammar boot
--- in the same Lua process and produce a grammar-constrained response.
---
--- Coverage :
---   - `Grammar.from_type` compiles to a valid GBNF string.
---   - The ion7-core SamplerBuilder accepts the GBNF and produces a
---     working sampler.
---   - `Engine:chat(s, { sampler = grammar_sampler })` honours the
---     constraint : the response is parseable JSON whose top-level
---     keys match the schema.
---   - Two concurrent sessions sharing the same grammar produce
---     independent valid outputs (per-seq isolation holds end-to-end).
---
--- We skip exact-content assertions — small models give probabilistic
--- answers — and focus on shape + JSON validity.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm, ctx, vocab, cm, engine = H.pipeline(model, {
    n_ctx     = 4096,
    n_seq_max = 4,
    headroom  = 256,
})

local Grammar = require "ion7.grammar"
local json    = require "ion7.vendor.json"

cm:set_system("You always reply with a single valid JSON object on one line, with no prose.")

-- Helper : build a grammar-constrained sampler from a GBNF string.
local function build_grammar_sampler(gbnf, root)
    return ion7.Sampler.chain()
        :grammar(gbnf, root or "root", vocab)
        :greedy()
        :build()
end

local function with_session(opts, fn)
    local s = llm.Session.new(opts)
    local ok, err = pcall(fn, s)
    cm:release(s)
    if not ok then error(err, 0) end
end

-- ════════════════════════════════════════════════════════════════════════
-- Grammar.from_type → GBNF → sampler → chat
-- ════════════════════════════════════════════════════════════════════════

T.suite("ion7-grammar + ion7-llm — type-driven grammar drives a chat turn")

T.test("response is parseable JSON matching the schema fields", function()
    -- Pin `additionalProperties = false` and `maxLength` so the model
    -- cannot drift into extra fields and blow the token budget.
    local g = Grammar.from_json_schema({
        type = "object",
        properties = {
            name = { type = "string",  maxLength = 24 },
            age  = { type = "integer" },
        },
        required = { "name", "age" },
        additionalProperties = false,
    })
    local sampler = build_grammar_sampler(g:to_gbnf())

    with_session(nil, function(s)
        s:add_user("Give me a fictional person : name and age.")
        local r = engine:chat(s, { max_tokens = 96, sampler = sampler })
        T.gt(#r.content, 0)
        local ok, parsed = pcall(json.decode, r.content)
        T.eq(ok, true, "engine output must be valid JSON; got: " .. r.content)
        T.eq(type(parsed), "table")
        T.eq(type(parsed.name), "string", "schema requires `name` (string)")
        T.eq(type(parsed.age),  "number", "schema requires `age` (integer)")
    end)

    sampler:reset()
end)

T.suite("ion7-grammar + ion7-llm — JSON Schema with enum + required")

T.test("response respects an enum constraint", function()
    local g = Grammar.from_json_schema({
        type = "object",
        properties = {
            status = { enum = { "ok", "error", "pending" } },
            code   = { type = "integer" },
        },
        required = { "status", "code" },
        additionalProperties = false,
    })
    local sampler = build_grammar_sampler(g:to_gbnf())

    with_session(nil, function(s)
        s:add_user("Return a status and a code for a successful request.")
        local r = engine:chat(s, { max_tokens = 48, sampler = sampler })
        local ok, parsed = pcall(json.decode, r.content)
        T.eq(ok, true, "JSON decode failed on: " .. r.content)
        T.eq(type(parsed.status), "string")
        local valid = parsed.status == "ok" or parsed.status == "error"
                   or parsed.status == "pending"
        T.eq(valid, true,
            "status must come from the enum; got: " .. tostring(parsed.status))
        T.eq(type(parsed.code), "number")
    end)

    sampler:reset()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Multi-session : two sessions on the same grammar do not collide
-- ════════════════════════════════════════════════════════════════════════

T.suite("ion7-grammar + ion7-llm — two sessions share a grammar safely")

T.test("two sessions on the same grammar each produce valid JSON", function()
    local g = Grammar.from_json_schema({
        type = "object",
        properties = {
            word = { type = "string", maxLength = 16 },
        },
        required = { "word" },
        additionalProperties = false,
    })
    local sampler_a = build_grammar_sampler(g:to_gbnf())
    local sampler_b = build_grammar_sampler(g:to_gbnf())

    local s1 = llm.Session.new()
    local s2 = llm.Session.new()

    s1:add_user("Reply with one short word.")
    s2:add_user("Reply with one different short word.")

    local r1 = engine:chat(s1, { max_tokens = 24, sampler = sampler_a })
    local r2 = engine:chat(s2, { max_tokens = 24, sampler = sampler_b })

    T.neq(s1.seq_id, s2.seq_id, "two sessions must own distinct seqs")

    local ok1, p1 = pcall(json.decode, r1.content)
    local ok2, p2 = pcall(json.decode, r2.content)
    T.eq(ok1, true, "session 1 JSON: " .. r1.content)
    T.eq(ok2, true, "session 2 JSON: " .. r2.content)
    T.eq(type(p1.word), "string")
    T.eq(type(p2.word), "string")

    cm:release(s1)
    cm:release(s2)
    sampler_a:reset()
    sampler_b:reset()
end)

ctx:free()
local ok = T.summary()
os.exit(ok and 0 or 1)
