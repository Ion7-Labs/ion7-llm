--- @module ion7.llm
--- @author  ion7 / Ion7 Project Contributors
---
--- Ion7 LLM — chat pipeline and multi-session inference orchestration
--- on top of `ion7.core`.
---
--- ion7-core gives you a model, a context, a vocab, a sampler.
--- ion7-llm wraps those into the missing layer between "raw next-token
--- decode" and "production chat application" :
---
---   - `Session`         conversation state with per-seq KV bookkeeping
---   - `kv.ContextManager` shared-context KV orchestrator (slots, prefix
---                          cache, snapshot fast path, eviction strategies)
---   - `Engine`          single-session inference (chat / stream / complete)
---   - `Pool`            multi-session inference (one batch per tick across
---                          N concurrent conversations)
---   - `chat.*`          chat-template + thinking + tool-call demux
---   - `sampler.*`       schema-constrained, reasoning-budgeted, profile presets
---   - `tools.*`         declarative Tool / ToolSet + interleaved-thinking loop
---   - `Embed`           dedicated embedding pipeline on a separate context
---
--- Out of scope (and intentionally so) :
---   HTTP / SSE / WebSocket transports, OpenAI / Anthropic / MCP server
---   endpoints, CLI binaries. ion7-llm is a library, not a server.
---
--- Lazy module loading mirrors `ion7.core` — accessing `ion7.llm.Engine`
--- triggers the require ; subsequent reads are direct table lookups.
---
---   local llm    = require "ion7.llm"
---   local core   = require "ion7.core"
---
---   core.init()
---   local model  = core.Model.load("qwen3-8b.gguf", { n_gpu_layers = 99 })
---   local ctx    = model:context({ n_ctx = 32768, n_seq_max = 1 })
---   local vocab  = model:vocab()
---
---   local cm     = llm.kv.new(ctx, vocab, { headroom = 256 })
---   local engine = llm.Engine.new(ctx, vocab, cm)
---
---   cm:set_system("You are a concise, helpful assistant.")
---
---   local s = llm.Session.new()
---   s:add_user("What is LuaJIT ?")
---   local r = engine:chat(s)
---   print(r.content)

local llm = {}

-- ── Class registry (lazy) ─────────────────────────────────────────────────

local _CLASSES = {
    -- Top-level engines / data.
    Engine     = "ion7.llm.engine",
    Pool       = "ion7.llm.pool",
    Session    = "ion7.llm.session",
    Response   = "ion7.llm.response",
    Embed      = "ion7.llm.embed",
    Stop       = "ion7.llm.stop",
}

-- ── Sub-namespaces (lazy) ─────────────────────────────────────────────────
--
-- Each sub-namespace materialises on first access into a table that
-- itself lazy-requires its own children. The factory returns a fresh
-- proxy table per sub-namespace ; we store it on the outer table so
-- subsequent reads hit the cache.

local function make_lazy(map)
    local t = {}
    return setmetatable(t, {
        __index = function(_, k)
            local p = map[k]
            if not p then return nil end
            local mod = require(p)
            rawset(t, k, mod)
            return mod
        end,
    })
end

local _NAMESPACES = {
    chat = function() return make_lazy({
        Template = "ion7.llm.chat.template",
        Thinking = "ion7.llm.chat.thinking",
        parse    = "ion7.llm.chat.parse",
        stream   = "ion7.llm.chat.stream",
    }) end,

    sampler = function() return make_lazy({
        profiles = "ion7.llm.sampler.profiles",
        budget   = "ion7.llm.sampler.budget",
    }) end,

    tools = function()
        local spec = require "ion7.llm.tools.spec"
        local loop = require "ion7.llm.tools.loop"
        return {
            Tool    = spec.Tool,
            ToolSet = spec.ToolSet,
            loop    = loop.run,
        }
    end,

    kv = function()
        local CM = require "ion7.llm.kv.init"
        -- Keep the class accessible AND expose a one-call constructor
        -- that mirrors the rest of the API : `llm.kv.new(ctx, vocab, opts)`.
        return setmetatable({
            ContextManager = CM,
            new            = CM.new,
            Slots          = require "ion7.llm.kv.slots",
            Prefix         = require "ion7.llm.kv.prefix",
            snapshot       = require "ion7.llm.kv.snapshot",
            eviction       = require "ion7.llm.kv.eviction",
            radix          = require "ion7.llm.kv.radix",
        }, { __call = function(_, ...) return CM.new(...) end })
    end,

    util = function() return make_lazy({
        messages     = "ion7.llm.util.messages",
        partial_json = "ion7.llm.util.partial_json",
        log          = "ion7.llm.util.log",
    }) end,
}

setmetatable(llm, {
    __index = function(t, k)
        local class_path = _CLASSES[k]
        if class_path then
            local mod = require(class_path)
            rawset(t, k, mod)
            return mod
        end
        local ns_factory = _NAMESPACES[k]
        if ns_factory then
            local ns = ns_factory()
            rawset(t, k, ns)
            return ns
        end
        return nil
    end,
})

-- ── Version + capability snapshot ────────────────────────────────────────

llm.VERSION = "0.2.0-beta1"

--- Capability snapshot of the ion7-llm runtime, complementing
--- `ion7.core.capabilities()`. Reflects what the chat pipeline can
--- offer given the linked libcommon bridge.
---
--- @return table {
---   version, has_chat_parse, has_schema_grammar,
---   has_reasoning_budget, has_template
--- }
function llm.capabilities()
    local core = require "ion7.core"
    local core_caps = core.capabilities()
    local has_bridge = core_caps.bridge_ver ~= nil

    return {
        version              = llm.VERSION,
        bridge_ver           = core_caps.bridge_ver,
        has_chat_parse       = has_bridge,
        has_schema_grammar   = has_bridge,
        has_reasoning_budget = has_bridge,
        has_template         = has_bridge,
    }
end

--- Convenience constructor : build a single-session chat pipeline in
--- one call. Returns the manager + engine pair.
---
---   local cm, engine = llm.pipeline(ctx, vocab, {
---       headroom = 256,
---       default_sampler = llm.sampler.profiles.balanced(),
---   })
---
--- @param  ctx   ion7.core.Context
--- @param  vocab ion7.core.Vocab
--- @param  opts  table?  Pass-through to both `kv.new` and `Engine.new`.
--- @return ion7.llm.kv.ContextManager
--- @return ion7.llm.Engine
function llm.pipeline(ctx, vocab, opts)
    opts = opts or {}
    local kv     = require "ion7.llm.kv.init"
    local Engine = require "ion7.llm.engine"
    local cm     = kv.new(ctx, vocab, opts)
    local engine = Engine.new(ctx, vocab, cm, opts)
    return cm, engine
end

return llm
