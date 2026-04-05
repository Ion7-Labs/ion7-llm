--- examples/01_with_grammar.lua
--- ion7-llm + ion7-grammar - Structured generation examples.
---
--- Shows the integration between ion7-llm (pipeline) and ion7-grammar
--- (grammar constraints). The model output is guaranteed to match the
--- grammar - enforced token-by-token, not by post-processing.
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/01_with_grammar.lua
---   ION7_GRAMMAR=../ion7-grammar    (optional, default: ../ion7-grammar)
---
--- @author Ion7-Labs

package.path =
    (os.getenv("ION7_GRAMMAR") or "../ion7-grammar") .. "/src/?.lua;" ..
    (os.getenv("ION7_GRAMMAR") or "../ion7-grammar") .. "/src/?/init.lua;" ..
    "./src/?.lua;./src/?/init.lua;" ..
    package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/01_with_grammar.lua\n")
    os.exit(1)
end

local llm     = require "ion7.llm"
local Grammar = require "ion7.grammar"

-- ── Setup ─────────────────────────────────────────────────────────────────────

llm.init({
    model  = MODEL,
    system = "You are a concise, structured assistant. Follow format instructions exactly.",
    think  = true,   -- strip <think> blocks for clean output
})
io.write("═══ ion7-llm + ion7-grammar integration ═══════════════════\n")
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── 1. llm.structured() - one-shot shorthand ──────────────────────────────────
io.write("── 1. llm.structured() ──────────────────────────────────────\n")

--- Sentiment classification - constrained to exactly three values.
local sentiment = Grammar.from_enum("root", { "positive", "negative", "neutral" })

local reviews = {
    "This product changed my life, absolutely fantastic!",
    "Arrived broken, terrible quality, waste of money.",
    "It works as described. Nothing special.",
}

for _, review in ipairs(reviews) do
    local resp = llm.structured(
        string.format("Review: '%s'\nSentiment:", review),
        sentiment,
        { max_tokens = 8 }
    )
    io.write(string.format("  %-55s -> %s\n", review:sub(1, 55), resp.text))
end

-- ── 2. JSON Schema extraction ─────────────────────────────────────────────────
io.write("\n── 2. JSON Schema extraction ────────────────────────────────\n")

local person_schema = Grammar.from_json_schema({
    type = "object",
    properties = {
        name  = { type = "string" },
        age   = { type = "integer" },
        city  = { type = "string" },
        role  = { enum = { "engineer", "designer", "manager", "researcher" } },
    },
    required = { "name", "age", "city" },
    additionalProperties = false,
})

local texts = {
    "Alice is 34 and works as an engineer in Paris.",
    "Bob, 28, is a UX designer based in Berlin.",
}

for _, text in ipairs(texts) do
    local resp = llm.structured(
        "Extract information from this text as JSON. Text: " .. text,
        person_schema,
        { max_tokens = 96 }
    )
    io.write("  text:   " .. text .. "\n")
    io.write("  output: " .. resp.text .. "\n\n")
end

-- ── 3. Type annotation (shortest path) ────────────────────────────────────────
io.write("── 3. Grammar.from_type() - shortest path ───────────────────\n")

local api_result = Grammar.from_type({
    success = "boolean",
    message = "string",
    code    = "integer",
})

local resp3 = llm.structured(
    "A database query failed due to a missing index. Return a JSON API response.",
    api_result,
    { max_tokens = 64 }
)
io.write("  output: " .. resp3.text .. "\n\n")

-- ── 4. opts.grammar in llm.chat() ─────────────────────────────────────────────
io.write("── 4. opts.grammar in llm.chat() ────────────────────────────\n")

--- The grammar option works with the full Session API too.
local method = Grammar.from_enum("root", { "GET", "POST", "PUT", "DELETE", "PATCH" })

local resp4 = llm.chat(
    "What HTTP method should I use to create a new user resource?",
    { grammar = method, max_tokens = 8 }
)
io.write("  HTTP method for create: " .. resp4.text .. "\n\n")

-- ── 5. Streaming with grammar ─────────────────────────────────────────────────
io.write("── 5. Streaming with grammar ────────────────────────────────\n")

local date_grammar = Grammar.from_regex("[0-9]{4}-[0-9]{2}-[0-9]{2}")

io.write("  Generating date: ")
for piece in llm.stream_structured(
    "What date is Christmas 2026? Reply with YYYY-MM-DD only.",
    date_grammar,
    { max_tokens = 12 }
) do
    io.write(piece)
    io.flush()
end
io.write("\n\n")

-- ── 6. Multi-turn with evolving grammar ───────────────────────────────────────
io.write("── 6. Multi-turn conversation + GrammarContext ───────────────\n")

--- Grammar evolves as the conversation reveals context.
local gc = Grammar.context()
gc:learn_enum("priority", { "low", "medium", "high", "critical" })

local Session = llm.Session
local session = Session.new({
    system = "You are a ticket triage system. Reply with priority levels only."
})

local tickets = {
    "Website is completely down, all users affected.",
    "Typo in the About page footer.",
    "Login page loads slowly for some users.",
}

for _, ticket in ipairs(tickets) do
    session:add("user", "Ticket: " .. ticket .. "\nPriority:")
    local g = gc:current()
    local resp = llm.gen():chat(session, { grammar = g, max_tokens = 8 })
    io.write(string.format("  %-55s -> %s\n", ticket:sub(1,55), resp.text))
    session:add("assistant", resp.text)
end

-- ── 7. Tool calling ───────────────────────────────────────────────────────────
io.write("\n── 7. Tool-call grammar ─────────────────────────────────────\n")

local tool_grammar = Grammar.from_tools({
    {
        name = "search",
        schema = { type = "object",
            properties = { query = { type = "string" } },
            required = { "query" } },
    },
    {
        name = "calculate",
        schema = { type = "object",
            properties = {
                expr      = { type = "string" },
                precision = { type = "integer" },
            },
            required = { "expr" } },
    },
    {
        name = "get-weather",
        schema = { type = "object",
            properties = { city = { type = "string" } },
            required   = { "city" } },
    },
})

local queries = {
    "What is 15% of 847?",
    "What's the weather like in Tokyo right now?",
    "Find recent papers about grammar-constrained LLM decoding.",
}

for _, q in ipairs(queries) do
    local resp = llm.structured(
        q .. "\nOutput a tool call JSON: {\"name\":\"...\",\"arguments\":{...}}",
        tool_grammar,
        { max_tokens = 80 }
    )
    io.write("  Q: " .. q .. "\n")
    io.write("  -> " .. resp.text .. "\n\n")
end

-- ── Cleanup ───────────────────────────────────────────────────────────────────
llm.shutdown()
io.write("═════════════════════════════════════════════════════════════\n")
io.write("All outputs guaranteed to match their grammars.\n")
