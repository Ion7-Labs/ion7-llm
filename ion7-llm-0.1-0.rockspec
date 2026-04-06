package = "ion7-llm"
version = "0.1-0"

source = {
    url = "git+https://github.com/Ion7-Labs/ion7-llm.git",
    tag = "v0.1.0",
}

description = {
    summary  = "LuaJIT LLM pipeline - chat, streaming, sessions, prefix cache, parallel generation",
    detailed = [[
        High-level generation API built on ion7-core.
        Session management, prefix cache, sliding window context,
        coroutine streaming, grammar constraints (ion7-grammar),
        multi-session Scheduler, reasoning model support.
    ]],
    homepage = "https://github.com/Ion7-Labs/ion7-llm",
    license  = "MIT-or-later",
}

dependencies = {
    "lua >= 5.1",
    "ion7-core >= 1.0",
    -- Optional: dkjson or cjson - required only for Session:save() / Session.load().
    -- Install one with: luarocks install dkjson
}

build = {
    type    = "builtin",
    modules = {
        ["ion7.llm"]                  = "src/ion7/llm/init.lua",
        ["ion7.llm.session"]          = "src/ion7/llm/session.lua",
        ["ion7.llm.generator"]        = "src/ion7/llm/generator.lua",
        ["ion7.llm.context_manager"]  = "src/ion7/llm/context_manager.lua",
        ["ion7.llm.scheduler"]        = "src/ion7/llm/scheduler.lua",
        ["ion7.llm.stop"]             = "src/ion7/llm/stop.lua",
        ["ion7.llm.response"]         = "src/ion7/llm/response.lua",
        ["ion7.llm.sampler_profiles"] = "src/ion7/llm/sampler_profiles.lua"
    },
}
