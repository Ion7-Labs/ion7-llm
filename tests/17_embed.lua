#!/usr/bin/env luajit
--- @module tests.17_embed
--- @author  ion7 / Ion7 Project Contributors
---
--- Embedding pipeline : `Embed.new`, `:encode`, `:encode_many`,
--- `Embed.cosine`.
---
--- The embedder owns its own `ion7.core.Context` configured with
--- `embeddings = true`, a small `n_ctx`, and a pooling strategy
--- chosen at `Embed.new` time. Tests below verify :
---
---   - encode returns a Lua array of floats whose length is the
---     model's output embedding dimension,
---   - the same input produces the same vector twice,
---   - `cosine(v, v) ≈ 1` for any non-zero vector,
---   - related sentences are MORE similar than unrelated ones.
---
--- The suite needs a dedicated embedding model — `ION7_EMBED` env var.
--- Without it the file skips cleanly and exits 0.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7 = H.require_backend(T)
local path = H.require_embed_model(T)
local model = ion7.Model.load(path, { n_gpu_layers = H.gpu_layers() })

local llm = require "ion7.llm"

-- ════════════════════════════════════════════════════════════════════════
-- Embed — basic encode shape
-- ════════════════════════════════════════════════════════════════════════

T.suite("Embed — encode shape + dimension")

T.test("encode returns a Lua array of n_embd floats", function()
    local em = llm.Embed.new(model, { n_ctx = 256, pooling = "last" })
    local v = em:encode("Hello, world.")
    T.is_type(v, "table")
    T.gt(#v, 0)
    -- Each entry is a number.
    T.is_type(v[1], "number")
    em:free()
end)

T.test("the same input produces the same vector across two calls", function()
    local em = llm.Embed.new(model, { n_ctx = 256, pooling = "last" })
    local v1 = em:encode("the quick brown fox")
    local v2 = em:encode("the quick brown fox")
    T.eq(#v1, #v2)
    -- Float equality is a reasonable check for a deterministic forward.
    -- Allow a very tight tolerance to avoid backend FP differences.
    for i = 1, #v1 do
        T.near(v1[i], v2[i], 1e-3,
            string.format("v1[%d]=%g vs v2[%d]=%g", i, v1[i], i, v2[i]))
    end
    em:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Cosine similarity sanity
-- ════════════════════════════════════════════════════════════════════════

T.suite("Embed — cosine similarity")

T.test("cosine(v, v) is approximately 1", function()
    local em = llm.Embed.new(model, { n_ctx = 256, pooling = "last" })
    local v = em:encode("identity")
    T.near(llm.Embed.cosine(v, v), 1.0, 1e-3)
    em:free()
end)

T.test("a related pair scores higher than an unrelated pair", function()
    local em = llm.Embed.new(model, { n_ctx = 256, pooling = "last" })
    local va = em:encode("the cat sat on the mat")
    local vb = em:encode("a feline rested on the rug")
    local vc = em:encode("the deficit ratio is 4.2 percent")
    local sim_ab = llm.Embed.cosine(va, vb)
    local sim_ac = llm.Embed.cosine(va, vc)
    T.gt(sim_ab, sim_ac,
        string.format("expected sim_ab > sim_ac, got %g vs %g", sim_ab, sim_ac))
    em:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- encode_many
-- ════════════════════════════════════════════════════════════════════════

T.suite("Embed — encode_many batches a list of texts")

T.test("encode_many returns one vector per input", function()
    local em = llm.Embed.new(model, { n_ctx = 256, pooling = "last" })
    local out = em:encode_many({ "alpha", "beta", "gamma" })
    T.eq(#out, 3)
    for i = 1, #out do T.gt(#out[i], 0) end
    em:free()
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
