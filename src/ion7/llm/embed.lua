--- @module ion7.llm.embed
--- @author  ion7 / Ion7 Project Contributors
---
--- Embedding pipeline. Lazy-creates a dedicated embedding `Context`
--- on first use ; once created, every `embed(text)` call decodes
--- through that context and reads back the pooled vector.
---
--- The embedding context is intentionally separate from the
--- generation context :
---
---   - it has `embeddings = true` (the generation context does not),
---   - it has a small `n_ctx` (512 is plenty for sentences),
---   - it uses a `pooling` strategy suited to embedding models
---     (`mean` for SBERT-style, `last` for GTE / E5 / Qwen-Embedding,
---     `cls` for BERT). The user picks at `Embed.new` time ; the
---     default of `last` is the best fit for modern embedders.
---
--- A consumer that wants a generation context AND an embedding
--- context on the same model holds both at once — they share the
--- model weights but each owns its KV cache.

local Embed = {}
Embed.__index = Embed

--- @class ion7.llm.Embed
--- @field _model  ion7.core.Model
--- @field _vocab  ion7.core.Vocab
--- @field _ctx    ion7.core.Context?     Lazily created at first call.
--- @field _opts   table

--- Build an embedder.
---
--- @param  model ion7.core.Model
--- @param  opts  table?
---   `n_ctx`     (integer, default 512)  Embedding context window.
---   `pooling`   (string,  default `"last"`) — `"none" | "mean" | "cls" | "last" | "rank"`.
---   `n_threads` (integer, default 4)
--- @return ion7.llm.Embed
function Embed.new(model, opts)
    opts = opts or {}
    return setmetatable({
        _model = model,
        _vocab = model:vocab(),
        _ctx   = nil,
        _opts  = {
            n_ctx     = opts.n_ctx     or 512,
            pooling   = opts.pooling   or "last",
            n_threads = opts.n_threads or 4,
        },
    }, Embed)
end

--- Lazy : build the embedding context on first use.
function Embed:_ensure_ctx()
    if self._ctx then return self._ctx end
    self._ctx = self._model:embedding_context({
        n_ctx     = self._opts.n_ctx,
        pooling   = self._opts.pooling,
        n_seq_max = 1,
        n_threads = self._opts.n_threads,
    })
    return self._ctx
end

--- Encode `text` into a Lua array of floats. The array length is
--- the model's output embedding dimension (typically 384 / 768 /
--- 1024 / 1536 / 4096 depending on the model).
---
--- @param  text string
--- @return number[]   1-based array of n_embd floats.
function Embed:encode(text)
    local ctx   = self:_ensure_ctx()
    local vocab = self._vocab
    local toks, n = vocab:tokenize(text, true, true)
    if n > ctx:n_ctx() then n = ctx:n_ctx() end
    ctx:kv_clear()
    ctx:decode(toks, n)
    local n_embd = self._model:n_embd_out() > 0
                   and self._model:n_embd_out() or self._model:n_embd()
    return ctx:embedding(0, n_embd)
end

--- Encode a list of texts. Each text gets its own forward pass —
--- no batched packing yet, but the embedding context is reused so
--- the per-call overhead is tokenisation + decode only.
---
--- @param  texts string[]
--- @return number[][]
function Embed:encode_many(texts)
    local out = {}
    for i, t in ipairs(texts) do out[i] = self:encode(t) end
    return out
end

--- Cosine similarity between two equal-length float arrays. Returns
--- a scalar in `[-1, 1]`.
---
--- @param  a number[]
--- @param  b number[]
--- @return number
function Embed.cosine(a, b)
    local dot, na, nb = 0, 0, 0
    for i = 1, #a do
        dot = dot + a[i] * b[i]
        na  = na  + a[i] * a[i]
        nb  = nb  + b[i] * b[i]
    end
    if na == 0 or nb == 0 then return 0 end
    return dot / (math.sqrt(na) * math.sqrt(nb))
end

--- Pooling strategy of the underlying context (after lazy creation),
--- as a symbolic string. Useful for diagnostics.
--- @return string
function Embed:pooling()
    if not self._ctx then return self._opts.pooling end
    return self._ctx:pooling_type()
end

--- Free the embedding context immediately. The model is left alone
--- (the caller still owns it). Calling `:encode` after `:free` will
--- lazy-rebuild a fresh context.
function Embed:free()
    if self._ctx then self._ctx:free() ; self._ctx = nil end
end

return Embed
