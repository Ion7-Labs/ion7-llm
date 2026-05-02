#!/usr/bin/env luajit
--- @example examples.05_grammar
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 05 — Structured output via ion7-grammar
--- ════════════════════════════════════════════════════════════════════════
---
--- ion7-llm intentionally does NOT bake a JSON-Schema sampler in.
--- That is what ion7-grammar exists for — a full GBNF engine in pure
--- Lua with constructors for `from_type`, `from_json_schema`,
--- `from_regex`, `from_enum`, `from_tools`, plus composition
--- operators and DCCD (Draft-Conditioned Constrained Decoding).
---
--- The integration with ion7-llm is one line : build the sampler from
--- the grammar, hand it to `engine:chat(s, { sampler = ... })`. Every
--- sampled token is mask-rejected if it would push the output outside
--- the grammar, so the model CANNOT emit invalid JSON.
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/05_grammar.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7    = require "ion7.core"
local llm     = require "ion7.llm"
local Grammar = require "ion7.grammar"
local json    = require "ion7.vendor.json"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(MODEL, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0,
})
local ctx   = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local vocab = model:vocab()
local cm, engine = llm.pipeline(ctx, vocab, { headroom = 256 })

cm:set_system("You output JSON only. No prose, no Markdown fences.")

-- ── 1. Build the grammar via ion7-grammar ────────────────────────────────
--
-- `from_type` is the fastest path : write a Lua type annotation, get
-- back a `Grammar_obj` ready to compile to GBNF. For richer schemas
-- use `Grammar.from_json_schema(schema_table)`.
local g = Grammar.from_type({
    name      = { "string", maxLength = 40 },
    age       = { "integer", minimum = 0, maximum = 120 },
    languages = { "array", items = "string", maxItems = 3, minItems = 1 },
})

print("--- compiled GBNF (first 200 chars) ---")
local gbnf = g:to_gbnf()
print(gbnf:sub(1, 200) .. (#gbnf > 200 and "..." or ""))
print("---")

-- ── 2. Wrap the grammar into a sampler ───────────────────────────────────
local sampler = ion7.Sampler.chain()
    :grammar(gbnf, "root", vocab)
    :top_k(40)
    :top_p(0.95)
    :temperature(0.4)
    :dist()
    :build()

-- ── 3. Run the constrained generation ────────────────────────────────────
local r = engine:complete(
    "Describe Ada Lovelace as a JSON object matching the agreed schema.",
    {
        sampler    = sampler,
        max_tokens = 192,
    })

print("\nRaw JSON  :", r.content)

-- The output is GUARANTEED to parse against `g`. Decode and use.
local person = json.decode(r.content)
print(string.format("\nName      : %s", person.name))
print(string.format("Age       : %d", person.age))
print("Languages :")
for _, l in ipairs(person.languages) do print("  - " .. l) end

sampler:free()
ctx:free()
model:free()
ion7.shutdown()
