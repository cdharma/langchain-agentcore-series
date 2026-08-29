# v0.7 — AgentCore Gateway: the tool leaves the codebase.
#
# What changed vs v0.6: get_weather is no longer agent code. The two
# Open-Meteo calls became GATEWAY TARGETS — geocoding via an OpenAPI spec
# (specs/open-meteo-geocoding.json), forecast via a 15-line Lambda — and the
# gateway serves both to the agent as MCP tools:
#     agent -> gateway (MCP) -> [OpenAPI target | Lambda target] -> Open-Meteo
# What that buys us:
#   - tools become config: add a target + redeploy, no code change
#   - credentials injected AT THE EDGE by the gateway's credential provider
#     (ours is a placeholder — Open-Meteo is free — but this is where real
#     API keys live; the agent process never sees them)
#   - the whole catalog is policy-enforceable and searchable
#     (x_amz_bedrock_agentcore_search is on by default)
# v0.6's durable AgentCore Memory carries over unchanged.
#
# Locally (agentcore dev) neither the gateway URL nor the memory ID exist
# until after the first deploy — we run tool-less with in-process memory,
# or inject the real values as env vars for full production behavior locally.
import os
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model

LangchainInstrumentor().instrument()

app = BedrockAgentCoreApp()
log = app.logger

# Both injected by AgentCore at deploy time; absent during `agentcore dev`.
MEMORY_ID = os.getenv("MEMORY_WEATHERAGENTV7MEMORY_ID")
GATEWAY_URL = os.getenv("AGENTCORE_GATEWAY_WEATHERGATEWAY_URL")
REGION = os.getenv("AWS_REGION", "us-east-1")

if MEMORY_ID:
    checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)
    store = AgentCoreMemoryStore(memory_id=MEMORY_ID, region_name=REGION)  # keyword-only ctor
    log.info(f"AgentCore Memory active: {MEMORY_ID}")
else:
    checkpointer = InMemorySaver()  # local dev fallback (v0.3 behavior)
    store = None
    log.info("No memory ID env var — in-process memory only")

_llm = None

def get_or_create_model():
    global _llm
    if _llm is None:
        _llm = load_model()
    return _llm


DEFAULT_SYSTEM_PROMPT = """
You are a helpful weather assistant. Use tools when appropriate.
To answer weather questions: first geocode the city to coordinates, then
fetch the current weather for those coordinates.
When you use a tool, you may say a brief sentence first. If no tool can
express what the user asked for, say so instead of guessing. Do not include
internal or system XML tags in your response.
"""


# Note what ISN'T here anymore: get_weather(). This file contains zero
# weather code — the tools live behind the gateway now.
_tools = None

async def get_tools():
    """Discover MCP tools from the gateway (cached after first call).

    Pre-first-deploy local dev has no gateway URL yet -> zero tools; the
    agent will simply say it can't look things up. After the first deploy,
    run `agentcore dev` with AGENTCORE_GATEWAY_WEATHERGATEWAY_URL set to get
    the real gateway tools locally too.
    """
    global _tools
    if _tools is None:
        if GATEWAY_URL:
            client = MultiServerMCPClient(
                {"gateway": {"transport": "streamable_http", "url": GATEWAY_URL}}
            )
            _tools = await client.get_tools()
            log.info(f"gateway tools discovered: {[t.name for t in _tools]}")
        else:
            _tools = []
            log.info("no gateway URL — running tool-less (deploy once, then inject the URL)")
    return _tools


def _mem_text(value) -> str:
    """Memory search results vary in shape — dig out the text."""
    if isinstance(value, dict):
        content = value.get("content", value)
        if isinstance(content, dict):
            content = content.get("text", str(content))
        return str(content)
    return str(value)


def pre_model_hook(state, config: RunnableConfig, *, store: BaseStore):
    """Runs before every model call:
    1. hand the latest user message to AgentCore Memory (extraction is async)
    2. search extracted facts/preferences and inject them into context
    """
    actor_id = config["configurable"]["actor_id"]
    thread_id = config["configurable"]["thread_id"]
    messages = state.get("messages", [])
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    llm_messages = messages
    if last_human is not None:
        store.put((actor_id, thread_id), str(uuid.uuid4()), {"message": last_human})

        remembered = []
        for ns in (("users", actor_id, "facts"), ("users", actor_id, "preferences")):
            try:
                remembered += store.search(ns, query=str(last_human.content), limit=3)
            except Exception as e:
                log.warning(f"memory search failed for {ns}: {e}")
        if remembered:
            notes = "\n".join(f"- {_mem_text(item.value)}" for item in remembered)
            log.info(f"long-term memories injected:\n{notes}")
            llm_messages = [SystemMessage(f"Things you remember about this user from earlier conversations:\n{notes}"), *messages]

    return {"llm_input_messages": llm_messages}


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")

    graph = create_react_agent(
        get_or_create_model(),
        tools=await get_tools(),
        prompt=DEFAULT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        pre_model_hook=pre_model_hook if store else None,
    )

    prompt = payload.get("prompt", "What can you help me with?")
    session_id = getattr(context, "session_id", "default-session")
    actor_id = payload.get("userId", "default-user")  # long-term memory is per actor
    log.info(f"Agent input: {prompt} (session {session_id}, actor {actor_id})")

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": session_id, "actor_id": actor_id}},
    )

    output = result["messages"][-1].text  # text blocks only — .content can be a list
    log.info(f"Agent output: {output}")
    return {"result": output}


if __name__ == "__main__":
    app.run()
