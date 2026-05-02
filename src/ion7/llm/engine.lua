--- @module ion7.llm.engine
--- @author  ion7 / Ion7 Project Contributors
---
--- Single-session inference engine. Wires the chat pipeline together :
---
---   1. `cm:prepare(session)` brings the per-seq KV row into a state
---      that ends exactly at the next decode position (snapshot fast
---      path or template re-render with eviction on overflow).
---   2. A sampler chain — caller-supplied or one of the
---      `sampler.profiles` presets — selects the next token from the
---      logits.
---   3. The tight per-token loop calls `ctx:decode(tok_cdata, 1,
---      seq_id, session.n_past)` for every step, advancing the
---      session's own `n_past`. The context's Lua-side `_n_past`
---      mirror is intentionally NOT touched — it would alias across
---      sessions sharing the same context.
---   4. Streaming demux through `chat.thinking` splits the flow into
---      `content` / `thinking` chunks ; `chat.tool_stream` then
---      routes content bytes that cross a tool-call marker into
---      typed `tool_call_delta` / `tool_call_done` chunks before the
---      rolling stop-string detector decides to halt.
---   5. `chat.parse.split` runs once at the end to produce the
---      authoritative `{ content, thinking, tool_calls }` Response.
---
--- Mid-generation eviction : when the per-seq KV cache approaches
--- the context wall, the loop calls `cm:make_room(session, ...)`
--- BEFORE the next decode rather than letting llama.cpp raise. The
--- result is a sequence that keeps generating past `n_ctx` instead
--- of dying with a `length` cut.
---
--- Public surface :
---
---   - `engine:chat(session, opts) -> Response`        synchronous.
---   - `engine:stream(session, opts) -> iterator`      coroutine yielding
---                                                      typed chunks.
---   - `engine:complete(prompt, opts) -> Response`     ephemeral session.
---
--- Tools : when `opts.tools` is a `ToolSet`, the engine injects a JSON
--- description into the system message via a one-shot `before_encode`
--- hook on the kv manager and forces the slow path so the new prompt
--- is encoded fresh. The trade-off is documented at the call site.
---
--- Structured output : intentionally NOT in this module. Build a
--- grammar via `require "ion7.grammar"` and pass the resulting
--- sampler in `opts.sampler`. ion7-llm is the chat pipeline ;
--- ion7-grammar owns GBNF, regex, JSON Schema, type-driven schemas
--- and the matching sampler glue.

local ffi = require "ffi"

local core    = require "ion7.core"

local Response    = require "ion7.llm.response"
local Thinking    = require "ion7.llm.chat.thinking"
local Parse       = require "ion7.llm.chat.parse"
local Stream      = require "ion7.llm.chat.stream"
local ToolStream  = require "ion7.llm.chat.tool_stream"
local Stop        = require "ion7.llm.stop"
local snapshot    = require "ion7.llm.kv.snapshot"
local profiles    = require "ion7.llm.sampler.profiles"
local log         = require "ion7.llm.util.log"

-- Hot-path locals. Caching builtins as upvalues lets LuaJIT compile
-- to direct references instead of metatable lookups every call.
local coroutine_yield = coroutine.yield
local coroutine_wrap  = coroutine.wrap
local table_concat    = table.concat
local string_format   = string.format
local stream_content  = Stream.content
local stream_thinking = Stream.thinking
local stream_stop     = Stream.stop

local Engine = {}
Engine.__index = Engine

--- @class ion7.llm.Engine
--- @field _ctx              ion7.core.Context
--- @field _vocab            ion7.core.Vocab
--- @field _cm               ion7.llm.kv.ContextManager
--- @field _default_sampler  ion7.core.Sampler?
--- @field _opts             table
--- @field _tok_cdata        cdata  Pre-allocated `int32_t[1]` for per-token decode.

--- Build an engine.
---
--- @param  ctx   ion7.core.Context
--- @param  vocab ion7.core.Vocab
--- @param  cm    ion7.llm.kv.ContextManager
--- @param  opts  table?
---   `default_sampler`     (ion7.core.Sampler?)  Used when no per-call sampler.
---   `max_tokens`          (integer, default 2048)  Per-call generation cap.
---   `stop_strings`        (string[]?)  Extra stop strings to merge with the
---                                      default chat-end markers.
---   `mid_gen_safety`      (integer, default 4)  Headroom (in tokens) reserved
---                                      ahead of the n_ctx wall ; once the
---                                      next decode would breach it the loop
---                                      asks the cm to evict.
---   `mid_gen_evict_chunk` (integer, default 64) Tokens to free in one mid-gen
---                                      eviction round. Larger = fewer evict
---                                      calls but more abrupt memory churn.
--- @return ion7.llm.Engine
function Engine.new(ctx, vocab, cm, opts)
    assert(ctx and vocab and cm,
        "[ion7.llm.engine] Engine.new(ctx, vocab, cm, opts) requires all three")
    opts = opts or {}
    return setmetatable({
        _ctx             = ctx,
        _vocab           = vocab,
        _cm              = cm,
        _default_sampler = opts.default_sampler,
        _opts            = {
            max_tokens          = opts.max_tokens          or 2048,
            stop_strings        = opts.stop_strings,
            mid_gen_safety      = opts.mid_gen_safety      or 4,
            mid_gen_evict_chunk = opts.mid_gen_evict_chunk or 64,
        },
        _tok_cdata       = ffi.new("int32_t[1]"),
    }, Engine)
end

-- ── Internal helpers ──────────────────────────────────────────────────────

-- One-shot before_encode hook that prepends a tool-description block
-- to (or in front of) the system message so the chat template renders
-- the model's available tools. We work on a shallow copy so the
-- session's stored messages are not mutated.
local function make_tools_hook(toolset)
    local doc = "You have access to the following tools. Call them by emitting JSON tool_calls when appropriate.\n\nTools:\n"
                .. toolset:to_json()
    return function(msgs, _session)
        local out = {}
        if msgs[1] and msgs[1].role == "system" then
            out[1] = {
                role    = "system",
                content = msgs[1].content .. "\n\n" .. doc,
            }
            for i = 2, #msgs do out[#out + 1] = msgs[i] end
        else
            out[1] = { role = "system", content = doc }
            for i = 1, #msgs do out[#out + 1] = msgs[i] end
        end
        return out
    end
end

-- Engage the tools hook for a single call. Returns a teardown function
-- the caller invokes when the chat is over.
function Engine:_engage_tools(session, tools)
    if tools == nil then return function() end end
    local cm = self._cm
    local prev_hook = cm._hooks.before_encode
    cm:set_hook("before_encode", make_tools_hook(tools))
    -- The injected tool block changes the prompt shape ; any stale
    -- snapshot must be discarded so prepare() takes the slow path.
    session._dirty        = true
    session._seq_snapshot = nil
    return function()
        if prev_hook then cm:set_hook("before_encode", prev_hook)
        else cm:clear_hook("before_encode") end
        -- The snapshot just taken contains a tools-augmented prefill ;
        -- a follow-up chat with a different tool set (or none) would
        -- restore it incorrectly. Force the next prepare back onto
        -- the slow path.
        session._dirty        = true
        session._seq_snapshot = nil
    end
end

-- Resolve the sampler used by a call. `opts.sampler` wins ; otherwise
-- the engine's default ; otherwise the `balanced` profile.
function Engine:_resolve_sampler(opts)
    if opts.sampler then return opts.sampler, false end
    if self._default_sampler then return self._default_sampler, false end
    return profiles.balanced(), true
end

-- Sample, decode (when more tokens are wanted), demux, halt-detect.
-- Pulled out of `:chat` and `:stream` so both share the exact same
-- per-token logic — the only difference is what the caller does with
-- each piece (collect into a string vs yield a chunk).
function Engine:_run_loop(session, sampler, opts, on_piece)
    local ctx, vocab, cm = self._ctx, self._vocab, self._cm
    local seq_id   = session.seq_id
    local engine_opts = self._opts

    local max_tokens     = opts.max_tokens     or engine_opts.max_tokens
    local mid_gen_safety = opts.mid_gen_safety or engine_opts.mid_gen_safety
    local mid_gen_chunk  = opts.mid_gen_evict_chunk or engine_opts.mid_gen_evict_chunk
    local stop = Stop.new({
        extra = opts.stop_strings or engine_opts.stop_strings,
    })

    -- Cache hot-path methods as locals so the JIT trace can fold them.
    local sampler_sample = sampler.sample
    local vocab_is_eog   = vocab.is_eog
    local vocab_piece    = vocab.piece
    local ctx_decode     = ctx.decode
    local ctx_ptr        = ctx:ptr()

    local tok_buf       = self._tok_cdata
    local toks_emitted  = {}
    local raw_parts     = {}
    local n_generated   = 0
    local stop_reason   = "length"
    local think_budget  = opts.think_budget
    local thinking      = Thinking.new()

    -- n_ctx_seq is a live FFI accessor ; cache it once. The mid-gen
    -- eviction path may push back the wall ; we re-read after eviction.
    local n_ctx_seq = ctx:n_ctx_seq()

    while n_generated < max_tokens do
        local tok = sampler_sample(sampler, ctx_ptr, -1)
        toks_emitted[#toks_emitted + 1] = tok

        if vocab_is_eog(vocab, tok) then
            stop_reason = "stop"
            break
        end

        local piece = vocab_piece(vocab, tok, true)
        raw_parts[#raw_parts + 1] = piece

        if on_piece then on_piece(piece, thinking) end

        if think_budget and thinking:in_think()
                and thinking:active_token_count() >= think_budget then
            local body = thinking:force_close()
            if log.level >= 4 then
                log.debug(string_format(
                    "reasoning budget hit (%d tok), force-closing", #body))
            end
        end

        local matched = stop:feed(piece)
        if matched then
            stop_reason = "stop_string"
            break
        end

        -- Mid-generation eviction guard. When the next decode would
        -- breach the per-seq context window minus a small safety
        -- margin, ask the cm to free `mid_gen_chunk` cells.
        if session.n_past + 1 + mid_gen_safety > n_ctx_seq then
            local freed = cm:make_room(session, mid_gen_chunk)
            if freed == 0 then
                -- The model does not support kv_seq_shift (recurrent /
                -- SSM) or no movable tokens are left. Honest stop.
                stop_reason = "length"
                break
            end
            -- ctx:n_ctx_seq() does not change after eviction (we do
            -- not resize the context, only its content) — the loop
            -- check above naturally accepts the new headroom because
            -- session.n_past has shrunk.
        end

        tok_buf[0] = tok
        ctx_decode(ctx, tok_buf, 1, seq_id, session.n_past)
        session.n_past = session.n_past + 1
        n_generated    = n_generated   + 1
    end

    return {
        raw          = table_concat(raw_parts),
        tokens       = toks_emitted,
        stop_reason  = stop_reason,
        thinking_sm  = thinking,
    }
end

-- Refresh the per-seq snapshot to match the post-generation KV state.
-- The session is left clean so the next chat() can take the fast path
-- if the caller does not mutate the message list.
function Engine:_resnapshot(session)
    if session.seq_id == nil then return end
    local blob = snapshot.save(self._ctx, session.seq_id)
    session:_save_seq_snapshot(blob, session.n_past)
end

-- Build the final Response from a loop result. Runs the
-- post-generation parser once for an authoritative split.
function Engine:_finalise(loop_result, opts)
    local raw = loop_result.raw
    if loop_result.stop_reason == "stop_string" and #raw > 0 then
        -- The last piece may carry trailing text up to AND past the stop
        -- string ; clip the visible content at the match boundary.
        local stop  = Stop.new({ extra = opts.stop_strings or self._opts.stop_strings })
        local vocab = self._vocab
        local toks  = loop_result.tokens
        for i = 1, #toks do
            local p = vocab:piece(toks[i], true)
            local matched = stop:feed(p)
            if matched then
                raw = stop:text_before(matched)
                break
            end
        end
    end

    local parsed = Parse.split(self._vocab, raw, {
        enable_thinking = opts.thinking,
        tool_format     = opts.tool_format,
    })

    local final_stop = parsed.has_tools and "tool_use" or loop_result.stop_reason

    return Response.new({
        content     = parsed.content,
        thinking    = parsed.thinking,
        tool_calls  = parsed.tool_calls,
        tokens      = loop_result.tokens,
        stop_reason = final_stop,
        perf        = self._ctx:perf(),
    })
end

-- ── Public API ────────────────────────────────────────────────────────────

--- Synchronous chat. Decodes the session, samples until a stop
--- condition, returns a fully-parsed Response.
---
--- @param  session ion7.llm.Session
--- @param  opts    table?
---   `sampler`             (ion7.core.Sampler?)  Replaces engine default.
---                                                Build via `ion7.grammar` for
---                                                schema/regex/type constraints.
---   `tools`               (ion7.llm.tools.ToolSet?)  Inject tool descriptions.
---   `tool_format`         (string?)             `"auto" | "openai" | "qwen" | "mistral"`.
---   `thinking`            (boolean?)            Force template `enable_thinking`.
---   `think_budget`        (integer?)            Max tokens inside `<think>...</think>`.
---   `max_tokens`          (integer?)            Per-call generation cap.
---   `stop_strings`        (string[]?)           Extra stop markers.
---   `mid_gen_safety`      (integer?)            Override engine default.
---   `mid_gen_evict_chunk` (integer?)            Override engine default.
--- @return ion7.llm.Response
function Engine:chat(session, opts)
    opts = opts or {}
    local teardown = self:_engage_tools(session, opts.tools)

    self._cm:prepare(session)

    local sampler, sampler_owned = self:_resolve_sampler(opts)
    local loop_result = self:_run_loop(session, sampler, opts, nil)

    if sampler_owned then sampler:reset() end

    self:_resnapshot(session)
    teardown()

    local resp = self:_finalise(loop_result, opts)
    session._last_response = resp
    return resp
end

--- Streaming chat. Returns an iterator that yields typed chunks :
---
---   { kind = "content",         text = "..." }
---   { kind = "thinking",        text = "..." }
---   { kind = "tool_call_delta", call_id, name, args_partial }
---   { kind = "tool_call_done",  call_id, call }
---   { kind = "stop",            reason = "stop" | "length" | "stop_string" | "tool_use" }
---
--- The iterator emits exactly one final `stop` chunk after the model
--- halts. Tool-call chunks fire AS SOON AS the open marker is detected
--- in the content stream, with `tool_call_delta` updates as the
--- arguments JSON accumulates and a `tool_call_done` once the close
--- marker (or balanced JSON braces) closes the call.
---
--- @param  session ion7.llm.Session
--- @param  opts    table?  Same as `:chat`.
--- @return function       Coroutine iterator yielding chunks.
function Engine:stream(session, opts)
    opts = opts or {}
    local engine = self
    return coroutine_wrap(function()
        local teardown = engine:_engage_tools(session, opts.tools)
        engine._cm:prepare(session)

        local sampler, sampler_owned = engine:_resolve_sampler(opts)

        -- Tool-stream state machine ; routes content bytes through it
        -- so the consumer gets typed tool_call chunks live.
        local tool_sm = ToolStream.new({ format = opts.tool_format or "auto" })

        -- Route a content chunk through the tool stream and yield
        -- whatever events it produces. Content bytes that don't cross
        -- a tool marker come back out as `content` chunks unchanged.
        local function emit_content(text)
            if #text == 0 then return end
            local events = tool_sm:feed(text)
            for i = 1, #events do
                coroutine_yield(events[i])
            end
        end

        local function on_piece(piece, thinking_sm)
            local kind, text, suffix = thinking_sm:feed(piece)
            if kind == "content" then
                emit_content(text)
            elseif kind == "thinking" then
                if #text > 0 then coroutine_yield(stream_thinking(text)) end
            elseif kind == "split" then
                -- The piece straddles a `<think>` boundary. The first
                -- half goes to one channel, the second half to the
                -- other ; which is which depends on whether we just
                -- ENTERED or LEFT the thinking block.
                if thinking_sm:in_think() then
                    emit_content(text)
                    if #suffix > 0 then coroutine_yield(stream_thinking(suffix)) end
                else
                    if #text > 0 then coroutine_yield(stream_thinking(text)) end
                    emit_content(suffix)
                end
            end
        end

        local loop_result = engine:_run_loop(session, sampler, opts, on_piece)
        if sampler_owned then sampler:reset() end
        engine:_resnapshot(session)
        teardown()

        local resp = engine:_finalise(loop_result, opts)
        session._last_response = resp

        coroutine_yield(stream_stop(resp.stop_reason))
    end)
end

--- One-shot completion : create an ephemeral session, chat, return
--- the Response. The session is discarded — no history is preserved.
---
--- @param  prompt string
--- @param  opts   table?  `system` (string?) plus any `:chat` option.
--- @return ion7.llm.Response
function Engine:complete(prompt, opts)
    opts = opts or {}
    local Session = require "ion7.llm.session"
    local s = Session.new({ system = opts.system })
    s:add_user(prompt)
    local resp = self:chat(s, opts)
    self._cm:release(s)
    return resp
end

-- ── Accessors ────────────────────────────────────────────────────────────

function Engine:context() return self._ctx end
function Engine:vocab()   return self._vocab end
function Engine:cm()      return self._cm end

return Engine
