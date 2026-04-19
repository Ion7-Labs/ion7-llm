--- @module ion7.llm.generator
--- SPDX-License-Identifier: MIT
--- Generation loop: coroutine streaming, grammar constraints, KV rollback,
--- optional speculative decoding.
---
--- Sampler compatibility:
---   - Sampler.chain() (llama_sampler_chain): sample() auto-accepts internally.
---     Never call accept() manually - double-accept crashes grammar samplers.
---   - CSampler (ion7_csampler_t): sample() also auto-accepts in the Lua wrapper.
---     Both types are drop-in compatible in the generation loop.
---
--- Grammar caching:
---   Grammar samplers are expensive to build (GBNF parsing + automaton).
---   Generator caches the last grammar sampler by GBNF string and reuses it
---   across calls. The sampler is reset() before each use to clear grammar state.
---
--- Native think budget:
---   When _native_budget = true, ion7_reasoning_budget_init() is already in the
---   sampler chain and enforces the hard token limit inside <think> blocks.
---   The Lua token counter is skipped; think-block stripping still runs normally.
---
--- Speculative decoding:
---   When a Speculative engine is set (_spec != nil), the generation loop uses
---   batch decoding to draft-and-verify multiple tokens per step via n-gram
---   prediction or draft-model heads. Acceptance rate varies by content.
---   The think-block and stop-string logic applies uniformly to all emitted tokens.
---
--- Entropy-adaptive speculation (_entropy_threshold):
---   When set, ctx:entropy(0) is measured after each sample. If the model's
---   distribution is uncertain (entropy > threshold), drafting is skipped for
---   that step and normal single-token decoding runs instead.
---   Low entropy (model confident) → drafts are accepted → speedup.
---   High entropy (model uncertain) → drafts would be rejected → skip overhead.
---   Typical range: 0.5–2.0 nats. Set via Generator:set_entropy_threshold().
---
--- @author Ion7-Labs
--- @version 0.3.0

local Stop     = require "ion7.llm.stop"
local Response = require "ion7.llm.response"

local Generator = {}
Generator.__index = Generator

--- @param  ctx    Context
--- @param  vocab  Vocab
--- @param  cm     ContextManager
--- @param  opts   table?
---   opts.sampler        table?   Pre-built Sampler or CSampler.
---   opts.sampler_opts   table?   { temp, top_k, top_p, min_p, seed } for fallback chain.
---   opts.grammar        table?   Default Grammar_obj or GBNF string.
---   opts.max_tokens     number?  Default: 2048.
---   opts.stop           table?   Extra stop strings.
---   opts.think          bool?    Strip <think> blocks. Default: false.
---   opts.think_budget   number?  Max tokens inside a <think> block. Default: nil.
---   opts.native_budget  bool?    True when reasoning_budget sampler is in the chain.
---   opts.speculative        table?   Speculative instance (ion7.core.Speculative).
---   opts.entropy_threshold  number?  Skip drafting when entropy > this value.
---                                    nil = always speculate (default).
--- @return Generator
function Generator.new(ctx, vocab, cm, opts)
    assert(ctx,   "[ion7.llm.generator] ctx required")
    assert(vocab, "[ion7.llm.generator] vocab required")
    assert(cm,    "[ion7.llm.generator] ContextManager required")
    opts = opts or {}

    local ion7_core    = require "ion7.core"
    local sampler_opts = opts.sampler_opts or {}
    local sampler

    if opts.sampler then
        sampler = opts.sampler
    else
        sampler = ion7_core.Sampler.chain()
            :top_k(sampler_opts.top_k or 40)
            :top_p(sampler_opts.top_p or 0.95, 1)
            :min_p(sampler_opts.min_p or 0.05, 1)
            :temperature(sampler_opts.temp or sampler_opts.temperature or 0.8)
            :dist(sampler_opts.seed or 0xFFFFFFFF)
            :build(vocab)
    end

    return setmetatable({
        _ctx             = ctx,
        _vocab           = vocab,
        _cm              = cm,
        _sampler         = sampler,
        _sampler_opts    = sampler_opts,
        _default_grammar = opts.grammar,
        _stop            = Stop.new({ extra = opts.stop }),
        _max_tokens      = opts.max_tokens or 2048,
        _think           = opts.think or false,
        _think_budget    = opts.think_budget or nil,
        -- True when ion7_reasoning_budget_init() is already in the sampler chain.
        -- Skips the Lua token counter; think-block stripping still runs normally.
        _native_budget      = opts.native_budget or false,
        _spec               = opts.speculative or nil,
        -- Entropy threshold for adaptive speculation. nil = always try to draft.
        -- When set, ctx:entropy(0) is checked before each draft attempt.
        -- If entropy > threshold, the speculative path is bypassed for that step.
        _entropy_threshold  = opts.entropy_threshold or nil,
        _checkpoint         = nil,
        -- Grammar sampler cache: { gbnf = string, sampler = Sampler }
        -- Avoids rebuilding the GBNF automaton on every call.
        _grammar_cache   = nil,
    }, Generator)
end

-- ── Grammar sampler ───────────────────────────────────────────────────────────

-- Grammar samplers must be first in the chain (they mask logits before sampling).
-- Returns the cached sampler if the GBNF string is unchanged; builds a new one
-- otherwise. The sampler is reset() by the caller before use.
function Generator:_grammar_sampler(grammar)
    local ion7_core = require "ion7.core"
    local s = self._sampler_opts

    local gbnf
    if type(grammar) == "string" then
        gbnf = grammar
    elseif type(grammar) == "table" and type(grammar.to_gbnf) == "function" then
        gbnf = grammar:to_gbnf()
    else
        error("[ion7.llm.generator] grammar must be a Grammar_obj or GBNF string, got: "
              .. type(grammar))
    end

    -- Cache hit: same grammar, reuse sampler
    if self._grammar_cache and self._grammar_cache.gbnf == gbnf then
        return self._grammar_cache.sampler
    end

    -- Cache miss: build new grammar sampler chain
    local new_sampler = ion7_core.Sampler.chain()
        :grammar(gbnf, "root", self._vocab)
        :top_k(s.top_k or 40)
        :top_p(s.top_p or 0.95, 1)
        :min_p(s.min_p or 0.05, 1)
        :temperature(s.temp or s.temperature or 0.8)
        :dist(s.seed or 0xFFFFFFFF)
        :build(self._vocab)

    if self._grammar_cache then self._grammar_cache.sampler:free() end
    self._grammar_cache = { gbnf = gbnf, sampler = new_sampler }
    return new_sampler
end

-- ── KV checkpoint / rollback ──────────────────────────────────────────────────

--- Save a KV snapshot. Call rollback() to undo subsequent generation.
--- @return Generator  self
function Generator:checkpoint()
    local ctx = self._ctx
    self._checkpoint = { snap = ctx:snapshot(), n_past = ctx:n_past() }
    return self
end

--- Restore to the last checkpoint().
--- @return Generator  self
function Generator:rollback()
    assert(self._checkpoint,
        "[ion7.llm.generator] no checkpoint saved - call checkpoint() first")
    local ctx = self._ctx
    ctx:restore(self._checkpoint.snap)
    ctx:set_n_past(self._checkpoint.n_past)
    return self
end

-- ── Generation loop ───────────────────────────────────────────────────────────

--- Stream tokens as a coroutine iterator. Yields string pieces.
--- Response is attached to session._last_resp on completion.
---
--- @param  session  Session
--- @param  opts     table?
---   opts.max_tokens   number?       Override max tokens.
---   opts.sampler      table?        Override sampler (Sampler or CSampler).
---   opts.grammar      table|string? Override grammar.
---   opts.on_think     function?     Called with each piece inside <think> blocks.
---                                   Only fires when think=true. Receives (piece).
--- @return function  iterator
function Generator:stream(session, opts)
    opts = opts or {}
    local ctx     = self._ctx
    local vocab   = self._vocab
    local cm      = self._cm
    local stop    = self._stop
    local max_tok           = opts.max_tokens or self._max_tokens
    local spec              = self._spec
    local entropy_threshold = self._entropy_threshold
    local on_think          = opts.on_think  -- fn(piece) called for tokens inside <think>

    -- think_budget: Lua soft limit only when NOT using native reasoning_budget sampler
    local think_budget = (not self._native_budget)
                         and (opts.think_budget or self._think_budget)
                         or  nil

    -- Priority: opts.sampler > grammar > default sampler
    local sampler
    if opts.sampler then
        sampler = opts.sampler
    elseif opts.grammar or self._default_grammar then
        sampler = self:_grammar_sampler(opts.grammar or self._default_grammar)
    else
        sampler = self._sampler
    end

    return coroutine.wrap(function()
        cm:prepare(session)
        stop:reset()
        sampler:reset()
        ctx:perf_reset()

        -- ── Generation state ─────────────────────────────────────────────────
        local token_ids   = {}
        local text_parts  = {}
        local stop_reason = "length"

        -- Think-block state (tail buffers avoid O(n) concat per token)
        local in_think    = false
        local all_think   = {}
        local think_tail  = ""   -- last 7 chars for </think> detection
        local open_tail   = ""   -- last 6 chars for <think> detection
        local think_tok_n = 0    -- Lua soft budget counter

        -- Generated token history for speculative n-gram lookup
        local gen_ids = {}

        -- ── Per-token emit (think/stop/yield pipeline) ────────────────────────
        -- Returns true when a stop_string matched (caller must break).
        -- Closes over all state variables above.
        local function emit(tok)
            local piece = vocab:piece(tok)

            if self._think then
                if not in_think then
                    local check = open_tail .. piece
                    open_tail = check:sub(-6)
                    if check:find("<think>", 1, true) then
                        in_think    = true
                        think_tail  = ""
                        think_tok_n = 0
                        text_parts[#text_parts + 1] = piece
                        return false   -- suppress yield, skip stop check
                    end
                else
                    think_tok_n = think_tok_n + 1
                    all_think[#all_think + 1] = piece
                    if on_think then on_think(piece) end

                    local check = think_tail .. piece
                    think_tail  = check:sub(-7)

                    if check:find("</think>", 1, true) then
                        in_think    = false
                        think_tail  = ""
                        think_tok_n = 0
                    elseif think_budget and think_tok_n >= think_budget then
                        -- Lua soft budget exceeded
                        in_think    = false
                        think_tail  = ""
                        think_tok_n = 0
                        text_parts[#text_parts + 1] = piece
                        coroutine.yield(piece)
                        return stop:feed(piece)
                    end
                    text_parts[#text_parts + 1] = piece
                    return false   -- still inside think, suppress yield
                end
            end

            text_parts[#text_parts + 1] = piece
            coroutine.yield(piece)
            return stop:feed(piece)
        end

        -- ── Speculative init ──────────────────────────────────────────────────
        if spec then spec:begin(gen_ids) end

        -- ── Main generation loop ──────────────────────────────────────────────
        local tok_n = 0  -- total tokens committed (for max_tok budget)

        while tok_n < max_tok do
            if ctx:n_past() >= ctx:n_ctx_seq() - 1 then
                stop_reason = "length"
                break
            end

            local tok = sampler:sample(ctx:ptr(), -1)

            if vocab:is_eog(tok) then
                stop_reason = "stop"
                break
            end

            -- Entropy-adaptive: skip speculative drafting when the model is
            -- uncertain (high entropy). High entropy → drafts would be rejected
            -- anyway, so avoid the overhead of batch decoding.
            local do_spec = spec ~= nil
            if do_spec and entropy_threshold then
                local ent = ctx:entropy(0)
                if ent > entropy_threshold then
                    do_spec = false
                end
            end

            if do_spec then
                -- ── SPECULATIVE PATH ─────────────────────────────────────────
                local n_past_before = ctx:n_past()
                local drafts        = spec:draft(gen_ids, tok)
                local n_draft       = #drafts

                if n_draft > 0 then
                    -- Batch decode: [tok, d1, d2, ..., dk]
                    local batch = { tok }
                    for _, d in ipairs(drafts) do batch[#batch + 1] = d end
                    ctx:decode_multi(batch, 0)

                    -- Commit tok (already verified by the outer sampler:sample)
                    tok_n = tok_n + 1
                    token_ids[#token_ids + 1] = tok
                    gen_ids[#gen_ids + 1]     = tok
                    if emit(tok) then stop_reason = "stop_string"; break end

                    -- Verify draft tokens
                    local n_acc      = 0
                    local early_stop = false

                    for i, d_tok in ipairs(drafts) do
                        if tok_n >= max_tok then break end

                        -- Sample at batch position i (logits for position after d[i-1])
                        -- This auto-accepts the chosen token in sampler history.
                        local v = sampler:sample(ctx:ptr(), i)

                        if v == d_tok and not vocab:is_eog(v) then
                            -- Draft accepted: free token
                            n_acc = n_acc + 1
                            tok_n = tok_n + 1
                            token_ids[#token_ids + 1] = v
                            gen_ids[#gen_ids + 1]     = v
                            if emit(v) then
                                stop_reason = "stop_string"
                                early_stop  = true
                                -- Rollback any remaining undecoded drafts
                                local tail = n_draft - n_acc
                                if tail > 0 then
                                    ctx:kv_seq_rm(0,
                                        n_past_before + n_acc + 1,
                                        n_past_before + n_draft + 1)
                                    ctx:set_n_past(n_past_before + n_acc + 1)
                                end
                                break
                            end
                        else
                            -- Mismatch: v is the correction token.
                            -- Remove the wrongly-decoded excess draft positions.
                            local excess = n_draft - n_acc
                            if excess > 0 then
                                ctx:kv_seq_rm(0,
                                    n_past_before + n_acc + 1,
                                    n_past_before + n_draft + 1)
                                ctx:set_n_past(n_past_before + n_acc + 1)
                            end

                            if vocab:is_eog(v) then
                                stop_reason = "stop"
                                early_stop  = true
                            else
                                tok_n = tok_n + 1
                                token_ids[#token_ids + 1] = v
                                gen_ids[#gen_ids + 1]     = v
                                -- Decode correction to set logits for next iteration.
                                ctx:decode_single(v, 0)
                                if emit(v) then
                                    stop_reason = "stop_string"
                                    early_stop  = true
                                end
                            end
                            break  -- end verification loop regardless
                        end
                    end

                    spec:accept(n_acc)
                    if early_stop then break end
                    -- All drafts accepted: logits[-1] = after last draft token.
                    -- Next sampler:sample(ctx, -1) reads those logits correctly. ✓
                else
                    -- No speculation available: standard single-token step.
                    ctx:decode_single(tok, 0)
                    tok_n = tok_n + 1
                    token_ids[#token_ids + 1] = tok
                    gen_ids[#gen_ids + 1]     = tok
                    spec:accept(0)
                    if emit(tok) then stop_reason = "stop_string"; break end
                end
            else
                -- ── STANDARD PATH (no speculation, or skipped by entropy gate) ─
                ctx:decode_single(tok, 0)
                tok_n = tok_n + 1
                token_ids[#token_ids + 1] = tok
                -- Keep spec engine's accept counter consistent when entropy gating
                -- caused us to skip drafting (spec != nil but do_spec = false).
                if spec then
                    gen_ids[#gen_ids + 1] = tok
                    spec:accept(0)
                end
                if emit(tok) then stop_reason = "stop_string"; break end
            end
        end

        -- ── Build full_text (strip think blocks if requested) ─────────────────
        local full_text
        if self._think then
            local raw = table.concat(text_parts)
            full_text = raw:gsub("<think>.-</think>%s*", "")
            if full_text:match("^%s*$") then full_text = raw end
        else
            full_text = table.concat(text_parts)
        end

        local perf_data = ctx:perf()
        local resp = Response.new(full_text, token_ids, stop_reason, {
            tok_per_s = perf_data.tokens_per_s,
            n_eval    = perf_data.n_eval,
            t_eval_ms = perf_data.t_eval_ms,
            n_p_eval  = perf_data.n_p_eval,
        }, (#all_think > 0) and table.concat(all_think) or nil)

        -- Grammar sampler is cached - do NOT free it here.

        local snap = ctx:snapshot()
        session:_save_snapshot(snap, ctx:n_past())
        session._last_resp = resp
    end)
end

--- Blocking chat: consume stream and return Response.
--- @param  session  Session
--- @param  opts     table?
--- @return Response
function Generator:chat(session, opts)
    opts = opts or {}
    for piece in self:stream(session, opts) do
        if opts.on_token then opts.on_token(piece) end
    end
    return session._last_resp
end

--- Wrap a raw prompt string in a Session and chat.
--- @param  prompt  string
--- @param  opts    table?
--- @return Response
function Generator:complete(prompt, opts)
    local Session = require "ion7.llm.session"
    local session = Session.new()
    session:add("user", prompt)
    return self:chat(session, opts)
end

--- Attach or replace the speculative engine.
--- Pass nil to disable speculation.
--- @param  spec  table?  Speculative instance (ion7.core.Speculative).
--- @return Generator  self
function Generator:set_speculative(spec)
    self._spec = spec
    return self
end

--- Set entropy threshold for adaptive speculation.
--- When ctx:entropy(0) exceeds this value after sampling, the speculative draft
--- step is bypassed for that token — normal single-token decoding runs instead.
---
--- Rationale: high entropy = model uncertain = draft tokens unlikely to match =
--- batch-decode overhead wasted. Skipping speculation at high-entropy positions
--- preserves throughput gains where the model is actually confident.
---
--- Typical range: 0.5 nats (aggressive, only draft when very confident)
---                2.0 nats (conservative, draft unless almost random).
--- Pass nil to disable adaptive gating (always speculate when spec engine is set).
---
--- @param  threshold  number?
--- @return Generator  self
function Generator:set_entropy_threshold(threshold)
    self._entropy_threshold = threshold
    return self
end

--- Free cached grammar sampler explicitly (optional - GC handles it otherwise).
function Generator:close()
    if self._grammar_cache then
        self._grammar_cache.sampler:free()
        self._grammar_cache = nil
    end
end

return Generator
