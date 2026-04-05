--- @module ion7.llm.scheduler
--- SPDX-License-Identifier: AGPL-3.0-or-later
--- True parallel generation via llama.cpp multi-sequence batch API.
---
--- Each step: 1 token per active session packed into a single llama_decode() call.
--- All sessions run on the GPU in parallel - not interleaved coroutines.
---
--- @author Ion7-Labs
--- @version 0.1.0

local Stop     = require "ion7.llm.stop"
local Response = require "ion7.llm.response"

-- ── Job ───────────────────────────────────────────────────────────────────────

local Job = {}
Job.__index = Job

function Job.new(session, sampler, stop_det, opts)
    return setmetatable({
        session       = session,
        sampler       = sampler,
        stop          = stop_det,
        max_tokens    = opts.max_tokens or 2048,
        on_piece      = opts.on_piece,
        on_done       = opts.on_done,
        pending_token = nil,
        batch_idx     = -1,
        n_generated   = 0,
        token_ids     = {},
        text_parts    = {},
        stop_reason   = "length",
        done          = false,
    }, Job)
end

-- ── Scheduler ─────────────────────────────────────────────────────────────────

local Scheduler = {}
Scheduler.__index = Scheduler

--- @param  ctx    Context
--- @param  vocab  Vocab
--- @param  cm     ContextManager
--- @return Scheduler
function Scheduler.new(ctx, vocab, cm)
    local Loader = require "ion7.core.ffi.loader"
    local L = Loader.instance()
    return setmetatable({
        _ctx   = ctx,
        _vocab = vocab,
        _cm    = cm,
        _lib   = L.lib,
        _ffi   = L.ffi,
        _jobs  = {},
    }, Scheduler)
end

--- Queue a session for parallel generation.
--- @param  session  Session
--- @param  opts     table
---   opts.sampler     cdata     Required.
---   opts.max_tokens  number?   Default: 2048.
---   opts.stop        table?    Extra stop strings.
---   opts.on_piece    function? Called per decoded piece.
---   opts.on_done     function? Called with Response on completion.
--- @return Scheduler  self
function Scheduler:submit(session, opts)
    assert(opts and opts.sampler,
        "[ion7.llm.scheduler] opts.sampler required for each job")
    local job = Job.new(session, opts.sampler, Stop.new({ extra = opts.stop }), opts)
    self._jobs[#self._jobs + 1] = job
    return self
end

--- Run all jobs to completion. Blocks until every job is done.
--- @return Scheduler  self
function Scheduler:run()
    assert(#self._jobs > 0, "[ion7.llm.scheduler] no jobs submitted")
    if #self._jobs == 1 then
        io.stderr:write("[ion7.llm.scheduler] single job - prefer Generator:chat()\n")
    end

    local ctx, vocab, cm, lib = self._ctx, self._vocab, self._cm, self._lib

    -- Phase 1: sequential prefill (fast with prefix cache)
    for _, job in ipairs(self._jobs) do
        cm:prepare(job.session)
        job.stop:reset()
        job.sampler:reset()

        local first = job.sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(first) then
            job.done        = true
            job.stop_reason = "stop"
            if job.on_done then pcall(job.on_done, Response.new("", {}, "stop", {})) end
        else
            job.sampler:accept(first)
            job.pending_token = first
        end
    end

    -- Phase 2: batched generation - 1 token/session per llama_decode()
    while true do
        local active = {}
        for _, job in ipairs(self._jobs) do
            if not job.done and job.pending_token ~= nil then
                active[#active + 1] = job
            end
        end
        if #active == 0 then break end

        local n     = #active
        local batch = lib.llama_batch_init(n, 0, 1)

        for i, job in ipairs(active) do
            local idx = i - 1
            batch.token[idx]     = job.pending_token
            batch.pos[idx]       = job.session.n_past
            batch.n_seq_id[idx]  = 1
            batch.seq_id[idx][0] = job.session.seq_id or 0
            batch.logits[idx]    = 1
            job.batch_idx        = idx
        end
        batch.n_tokens = n

        local ret = lib.llama_decode(ctx:ptr(), batch)
        lib.llama_batch_free(batch)

        if ret ~= 0 then
            for _, job in ipairs(active) do
                job.done        = true
                job.stop_reason = "error"
                if job.on_done then
                    pcall(job.on_done, Response.new(
                        table.concat(job.text_parts), job.token_ids, "error", {}))
                end
            end
            break
        end

        for _, job in ipairs(active) do
            local piece = vocab:piece(job.pending_token)
            job.token_ids[#job.token_ids + 1]  = job.pending_token
            job.text_parts[#job.text_parts + 1] = piece
            job.session.n_past = job.session.n_past + 1
            job.n_generated    = job.n_generated  + 1

            if job.on_piece then pcall(job.on_piece, piece) end

            local matched = job.stop:feed(piece)
            if matched then
                job.done          = true
                job.stop_reason   = "stop_string"
                job.pending_token = nil
            elseif job.n_generated >= job.max_tokens then
                job.done          = true
                job.pending_token = nil
            else
                local next_tok = job.sampler:sample(ctx:ptr(), job.batch_idx)
                if vocab:is_eog(next_tok) then
                    job.done          = true
                    job.stop_reason   = "stop"
                    job.pending_token = nil
                else
                    job.sampler:accept(next_tok)
                    job.pending_token = next_tok
                end
            end

            if job.done and job.on_done then
                pcall(job.on_done, Response.new(
                    table.concat(job.text_parts), job.token_ids, job.stop_reason, {}))
            end
        end
    end

    self._jobs = {}
    return self
end

--- Number of submitted jobs not yet run.
--- @return number
function Scheduler:n_jobs()
    return #self._jobs
end

return Scheduler
