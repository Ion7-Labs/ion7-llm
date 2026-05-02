#!/usr/bin/env luajit
--- @example examples.09_embed
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 09 — Embeddings : encode + cosine similarity matrix
--- ════════════════════════════════════════════════════════════════════════
---
--- The `Embed` class wraps an embedding pipeline on a dedicated
--- context. Generation models and embedding models are NOT
--- interchangeable — the latter ship with `embeddings = true` in the
--- context params, a small `n_ctx`, and a pooling strategy
--- (mean / last / cls).
---
--- The example loads an embedding model, encodes a small set of
--- sentences, prints the cosine similarity matrix. Related sentences
--- score higher than unrelated ones.
---
---   ION7_EMBED=/path/to/embed.gguf luajit examples/09_embed.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local EMBED = os.getenv("ION7_EMBED") or
    error("Set ION7_EMBED=/path/to/embed.gguf", 0)

local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(EMBED, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0,
})

-- `pooling = "last"` is the recommended default for modern embedding
-- models (GTE, E5, Qwen-Embedding). SBERT-style models prefer "mean".
-- BERT classifiers use "cls".
local em = llm.Embed.new(model, { n_ctx = 512, pooling = "last" })

-- ── Encode a set of related + unrelated sentences ────────────────────────

local sentences = {
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "The deficit ratio was 4.2 percent.",
    "Inflation eased to a four-year low.",
    "Vector databases store dense representations.",
    "FAISS and Annoy index high-dimensional points.",
}

print("[encode] " .. #sentences .. " sentences")
local vectors = em:encode_many(sentences)
print(string.format("  embedding dimension : %d", #vectors[1]))

-- ── Pairwise cosine similarity ──────────────────────────────────────────

print("\nCosine similarity matrix :")
io.write("       ")
for i = 1, #sentences do io.write(string.format(" s%-2d ", i)) end
io.write("\n")
for i = 1, #sentences do
    io.write(string.format("s%-2d   ", i))
    for j = 1, #sentences do
        local sim = llm.Embed.cosine(vectors[i], vectors[j])
        io.write(string.format("%.2f ", sim))
    end
    io.write("\n")
end

print("\nLegend :")
for i, s in ipairs(sentences) do
    print(string.format("  s%d : %s", i, s))
end

-- ── Closest-neighbour query ─────────────────────────────────────────────

local query = "Cats are sleeping animals."
print("\nQuery : " .. query)
local q = em:encode(query)
local scored = {}
for i, v in ipairs(vectors) do
    scored[#scored + 1] = { idx = i, sim = llm.Embed.cosine(q, v) }
end
table.sort(scored, function(a, b) return a.sim > b.sim end)
print("Top-3 nearest sentences :")
for k = 1, math.min(3, #scored) do
    print(string.format("  %.3f -> %s", scored[k].sim, sentences[scored[k].idx]))
end

em:free()
model:free()
ion7.shutdown()
