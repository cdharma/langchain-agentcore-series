# v0.6 — durable memory: AgentCore Memory replaces InMemorySaver.
#
# What changed vs v0.5: memory finally survives. Two integrations from the
# official langgraph-checkpoint-aws package:
#   - AgentCoreMemorySaver  -> short-term: LangGraph checkpoints persisted to
#     the AgentCore Memory service (drop-in replacement for InMemorySaver)
#   - AgentCoreMemoryStore  -> long-term: raw messages are handed to the
#     service, which EXTRACTS facts + preferences in the background
#     (strategies: SEMANTIC -> /users/{actorId}/facts,
#                  USER_PREFERENCE -> /users/{actorId}/preferences,
#                  plus SUMMARIZATION and EPISODIC per session)
# The pre_model_hook saves each user message for extraction and injects any
# remembered facts/preferences back into the model's context — so a brand-new
# session can know things learned in an old one.
#
# Locally (agentcore dev) the MEMORY_* env var doesn't exist — we fall back to
# v0.3's InMemorySaver so the chat still has within-session memory.
import os
import uuid

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langchain.tools import tool
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model

LangchainInstrumentor().instrument()

app = BedrockAgentCoreApp()
log = app.logger

# Injected by AgentCore at deploy time (format: MEMORY_<UPPERCASENAME>_ID).
# Not set during `agentcore dev` — memory needs a deploy.
MEMORY_ID = os.getenv("MEMORY_WEATHERAGENTV6MEMORY_ID")
REGION = os.getenv("AWS_REGION", "us-east-1")

if MEMORY_ID:
    checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)
    store = AgentCoreMemoryStore(memory_id=MEMORY_ID, region_name=REGION)  # keyword-only ctor
    log.info(f"AgentCore Memory active: {MEMORY_ID}")
else:
    checkpointer = InMemorySaver()  # local dev fallback (v0.3 behavior)
    store = None
    log.info("No MEMORY_WEATHERAGENTV6MEMORY_ID — running with in-process memory only")

_llm = None

def get_or_create_model():
    global _llm
    if _llm is None:
        _llm = load_model()
    return _llm


DEFAULT_SYSTEM_PROMPT = """
You are a helpful weather assistant. Use tools when appropriate.
When you use a tool, you may say a brief sentence first. If no tool can
express what the user asked for, say so instead of guessing. Do not include
internal or system XML tags in your response.
"""


@tool
def get_weather(city: str) -> str:
    """Get the CURRENT, real weather for a city (live Open-Meteo data)."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=10,
    ).json()
    if not geo.get("results"):
        return f"Could not find a place called {city!r}."
    place = geo["results"][0]
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
        }, timeout=10,
    ).json()["current"]
    return (
        f"{place['name']}, {place.get('country', '')}: {wx['temperature_2m']}°C "
        f"(feels like {wx['apparent_temperature']}°C), humidity {wx['relative_humidity_2m']}%, "
        f"wind {wx['wind_speed_10m']} km/h"
    )


tools = [get_weather]


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
            try:  # namespace tuples map to the strategy paths: /users/{actorId}/facts|preferences
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
        tools=tools,
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
