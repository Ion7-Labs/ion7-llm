--- @module ion7.llm.generator
--- SPDX-License-Identifier: AGPL-3.0-or-later
--- Generation loop: coroutine streaming, grammar constraints, KV rollback.
---
--- llama_sampler_sample() on a chain calls llama_sampler_accept() internally.
--- Never call sampler:accept() explicitly in the loop - double-accept crashes
--- the grammar sampler.
---
--- @author Ion7-Labs
--- @version 0.1.0

local Stop     = require "ion7.llm.stop"
local Response = require "ion7.llm.response"

local Generator = {}
Generator.__index = Generator

--- @param  ctx    Context
--- @param  vocab  Vocab
--- @param  cm     ContextManager
--- @param  opts   table?
---   opts.sampler       cdata?   Pre-built Sampler.
---   opts.sampler_opts  table?   { temp, top_k, top_p, min_p, seed }
---   opts.grammar       table?   Default Grammar_obj or GBNF string.
---   opts.max_tokens    number?  Default: 2048.
---   opts.stop          table?   Extra stop strings.
---   opts.think         bool?    Strip <think> blocks. Default: false.
---   opts.think_budget  number?  Max tokens inside a <think> block. Default: nil.
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
        _checkpoint      = nil,
    }, Generator)
end

-- ── Grammar sampler ───────────────────────────────────────────────────────────

-- Grammar samplers must be first in the chain (they mask logits before sampling).
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

    return ion7_core.Sampler.chain()
        :grammar(gbnf, "root", self._vocab)
        :top_k(s.top_k or 40)
        :top_p(s.top_p or 0.95, 1)
        :min_p(s.min_p or 0.05, 1)
        :temperature(s.temp or s.temperature or 0.8)
        :dist(s.seed or 0xFFFFFFFF)
        :build(self._vocab)
end

-- ── KV checkpoint / rollback ──────────────────────────────────────────────────

--- Save a KV snapshot. Call rollback() to undo subsequent generation.
--- @return Generator  self
function Generator:checkpoint()
    local ctx = self._ctx
    self._checkpoint = { snap = ctx:snapshot(), n_past = ctx._n_past }
    return self
end

--- Restore to the last checkpoint().
--- @return Generator  self
function Generator:rollback()
    assert(self._checkpoint,
        "[ion7.llm.generator] no checkpoint saved - call checkpoint() first")
    local ctx = self._ctx
    ctx:restore(self._checkpoint.snap)
    ctx._n_past = self._checkpoint.n_past
    return self
end

-- ── Generation loop ───────────────────────────────────────────────────────────

--- Stream tokens as a coroutine iterator. Yields string pieces.
--- Response is attached to session._last_resp on completion.
---
--- @param  session  Session
--- @param  opts     table?
---   opts.max_tokens   number?       Override max tokens.
---   opts.sampler      cdata?        Override sampler.
---   opts.grammar      table|string? Override grammar.
---   opts.think_budget number?       Override think_budget.
--- @return function  iterator
function Generator:stream(session, opts)
    opts = opts or {}
    local ctx          = self._ctx
    local vocab        = self._vocab
    local cm           = self._cm
    local stop         = self._stop
    local max_tok      = opts.max_tokens or self._max_tokens
    local think_budget = opts.think_budget or self._think_budget

    -- Priority: opts.sampler > grammar > default sampler
    local sampler
    local grammar_sampler_built = false
    if opts.sampler then
        sampler = opts.sampler
    elseif opts.grammar or self._default_grammar then
        sampler = self:_grammar_sampler(opts.grammar or self._default_grammar)
        grammar_sampler_built = true
    else
        sampler = self._sampler
    end

    return coroutine.wrap(function()
        cm:prepare(session)
        stop:reset()
        sampler:reset()
        ctx:perf_reset()

        local token_ids  = {}
        local text_parts = {}
        local stop_reason = "length"

        -- Think block state. Tail buffers avoid O(n) table.concat per token.
        local in_think    = false
        local all_think   = {}   -- accumulated think text
        local think_tail  = ""   -- last 7 chars for </think> detection
        local open_tail   = ""   -- last 6 chars for <think> detection
        local think_tok_n = 0

        for _ = 1, max_tok do
            if ctx._n_past >= ctx:n_ctx_seq() - 1 then
                stop_reason = "length"
                break
            end

            local tok = sampler:sample(ctx:ptr(), -1)

            if vocab:is_eog(tok) then
                stop_reason = "stop"
                break
            end

            ctx:decode_single(tok, 0)
            token_ids[#token_ids + 1] = tok

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
                        goto continue
                    end
                else
                    think_tok_n = think_tok_n + 1
                    all_think[#all_think + 1] = piece

                    local check = think_tail .. piece
                    think_tail = check:sub(-7)

                    if check:find("</think>", 1, true) then
                        in_think    = false
                        think_tail  = ""
                        think_tok_n = 0
                    elseif think_budget and think_tok_n >= think_budget then
                        in_think    = false
                        think_tail  = ""
                        think_tok_n = 0
                        text_parts[#text_parts + 1] = piece
                        coroutine.yield(piece)
                        goto stop_check
                    end
                    text_parts[#text_parts + 1] = piece
                    goto continue
                end
            end

            text_parts[#text_parts + 1] = piece
            coroutine.yield(piece)

            ::stop_check::
            local matched = stop:feed(piece)
            if matched then
                stop_reason = "stop_string"
                break
            end

            ::continue::
        end

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

        if grammar_sampler_built then sampler:free() end

        local snap = ctx:snapshot()
        session:_save_snapshot(snap, ctx._n_past)
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

return Generator
