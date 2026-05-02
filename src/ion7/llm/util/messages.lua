--- @module ion7.llm.util.messages
--- @author  ion7 / Ion7 Project Contributors
---
--- Helpers for the `{ role, content, ... }` message shape that flows
--- between Session, ChatTemplate and the model.
---
--- The chat template (Jinja2) renders four canonical roles :
---   - `system`     pre-conversation instructions
---   - `user`       human turn
---   - `assistant`  model reply (may carry `thinking`, `tool_calls`)
---   - `tool`       result of a tool call, follows a `tool_calls` turn

local M = {}

local VALID_ROLES = {
    system    = true,
    user      = true,
    assistant = true,
    tool      = true,
}

--- True when `role` is one of the four canonical roles the chat
--- template knows how to render.
--- @param  role string
--- @return boolean
function M.is_valid_role(role)
    return VALID_ROLES[role] == true
end

--- Build a `user` message.
--- @param  text string
--- @return table { role = "user", content = text }
function M.user(text)
    return { role = "user", content = text or "" }
end

--- Build a `system` message.
--- @param  text string
--- @return table
function M.system(text)
    return { role = "system", content = text or "" }
end

--- Build an `assistant` message. Optional `thinking` and `tool_calls`
--- fields are passed through to the chat template — the template
--- decides how to render them (Qwen3 wraps them in `<think>`,
--- Claude-style chats serialise tool_calls to a tool_use block, etc.).
--- @param  text       string
--- @param  extras     table? { thinking?, tool_calls? }
--- @return table
function M.assistant(text, extras)
    local m = { role = "assistant", content = text or "" }
    if extras then
        if extras.thinking   then m.thinking   = extras.thinking   end
        if extras.tool_calls then m.tool_calls = extras.tool_calls end
    end
    return m
end

local json = require "ion7.vendor.json"

--- Build a `tool` message — the result of executing a tool call. The
--- `tool_call_id` ties the result back to the assistant's call so
--- the model can reason about which result corresponds to which call.
---
--- @param  tool_call_id string
--- @param  content      string|table  Stringified result, or a table —
---                                    tables go through ion7-core's
---                                    bundled JSON encoder.
--- @return table
function M.tool_result(tool_call_id, content)
    return {
        role         = "tool",
        tool_call_id = tool_call_id,
        content      = type(content) == "string" and content
                       or json.encode(content),
    }
end

--- Validate a message shape. Raises on a malformed entry so a stray
--- `nil` content or a typoed role surfaces at append time, not
--- silently at apply_template time.
--- @param  msg table
--- @raise   When `msg` is malformed.
function M.validate(msg)
    assert(type(msg) == "table",
        "[ion7.llm.util.messages] message must be a table")
    assert(VALID_ROLES[msg.role],
        "[ion7.llm.util.messages] invalid role : " .. tostring(msg.role))
    assert(type(msg.content) == "string",
        "[ion7.llm.util.messages] content must be a string")
    if msg.role == "tool" then
        assert(type(msg.tool_call_id) == "string" and #msg.tool_call_id > 0,
            "[ion7.llm.util.messages] tool message requires a non-empty tool_call_id")
    end
end

return M
