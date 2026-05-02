--- @module ion7.llm.tools.loop
--- @author  ion7 / Ion7 Project Contributors
---
--- Multi-turn tool-use loop : repeatedly chat with the model,
--- dispatch any tool_calls it emits, feed the results back, until
--- the model produces a content-only turn (no more tool_calls).
---
--- The loop preserves the model's `thinking` block on every turn
--- alongside the tool_calls — required by reasoning models that
--- alternate between planning and tool invocation within a single
--- assistant exchange ("interleaved thinking", late-2025 pattern
--- across Claude Opus Thinking, OpenAI o-series, DeepSeek-R1, Qwen3).
--- Stripping thinking on a tool turn would discard the model's plan
--- before the next reasoning step gets to read it back.
---
---   local final = tools.loop(engine, session, {
---       tools     = my_tool_set,
---       max_turns = 6,
---       on_call   = function(call, result) ... end,
---   })

local M = {}

local DEFAULT_MAX_TURNS = 8

--- Run the tool-use loop until the model returns a content-only
--- turn or `max_turns` is exceeded.
---
--- The dispatch path is :
---   - call `engine:chat(session, { tools = toolset })` ;
---   - if the response carries `tool_calls`, append the assistant
---     message (with thinking + tool_calls preserved) to the
---     session, then for each call resolve the matching tool from
---     the `ToolSet`, dispatch its `handler`, append the result as
---     a tool message ;
---   - loop, with the next chat turn re-rendering the augmented
---     history through the chat template.
---
--- @param  engine    ion7.llm.Engine
--- @param  session   ion7.llm.Session
--- @param  opts      table
---   `tools`      (ion7.llm.tools.ToolSet, required)
---   `max_turns`  (integer, default 8)   Hard cap on dispatch rounds.
---   `on_call`    (function?, optional)  Notified after each dispatch :
---                  `on_call(call, result)`.
---   `chat_opts`  (table?, optional)     Forwarded to `engine:chat`
---                                        on every iteration.
--- @return ion7.llm.Response  The final content-only response.
--- @return integer            Number of dispatch rounds executed.
function M.run(engine, session, opts)
    assert(opts and opts.tools,
        "[ion7.llm.tools.loop] opts.tools is required")
    local tools     = opts.tools
    local max_turns = opts.max_turns or DEFAULT_MAX_TURNS
    local chat_opts = opts.chat_opts or {}

    -- Make sure the engine knows about the tool descriptions on
    -- every turn — we copy here so a caller can mutate `chat_opts`
    -- between rounds without losing the tools.
    chat_opts.tools = tools

    local turn = 0
    while turn < max_turns do
        turn = turn + 1
        local resp = engine:chat(session, chat_opts)

        if not resp:has_tools() then
            return resp, turn
        end

        -- Persist the assistant's interleaved-thinking + tool_calls
        -- envelope before dispatching, so the next chat call's
        -- chat-template re-render contains the model's plan.
        session:add_assistant(resp.content, {
            thinking   = resp.thinking,
            tool_calls = resp.tool_calls,
        })

        for _, call in ipairs(resp.tool_calls) do
            local tool = tools:find(call.name)
            local result
            if not tool then
                result = { error = "unknown tool : " .. tostring(call.name) }
            elseif type(tool.handler) ~= "function" then
                result = { error = "tool '" .. call.name .. "' has no handler" }
            else
                result = tool:dispatch(call.arguments)
            end

            if opts.on_call then opts.on_call(call, result) end
            session:add_tool_result(call.id, result)
        end
    end

    -- Hit the turn cap : run one final chat to extract whatever the
    -- model has now, without forcing tool dispatch on the result.
    chat_opts.tools = nil
    local final = engine:chat(session, chat_opts)
    return final, turn
end

return M
