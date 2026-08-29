# v0.5 — the series agent, deployed on Amazon Bedrock AgentCore Runtime.
#
# What changed vs v0.4: nothing about the agent — everything about where it
# runs. The REPL is gone; AgentCore Runtime calls @app.entrypoint over HTTP
# (POST /invocations) and gives every session its own isolated microVM.
# The checkpointer trick from v0.3 still works: thread_id = the runtime's
# session_id, so each conversation keeps its memory between invocations.
from collections import OrderedDict

import requests
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model

# auto-instrumentation: every model/tool call becomes an OTel span,
# visible in CloudWatch GenAI Observability once deployed
LangchainInstrumentor().instrument()

app = BedrockAgentCoreApp()
log = app.logger

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

# Module-level checkpointer preserves conversation history across invocations.
# InMemorySaver keeps every thread_id (= session_id) checkpoint in memory
# forever, so we bound it to 128 active threads with LRU eviction (the
# least-recently-used thread is deleted and its history reset) to keep a
# long-running process from growing without limit. For durable history, swap in
# a persistent checkpointer — or AgentCore Memory (that's v0.6).
_CHECKPOINT_LIMIT = 128
_checkpointer = InMemorySaver()
_thread_ids = OrderedDict()


def touch_thread(thread_id):
    if thread_id in _thread_ids:
        _thread_ids.move_to_end(thread_id)
        return
    while len(_thread_ids) >= _CHECKPOINT_LIMIT:
        evicted, _ = _thread_ids.popitem(last=False)
        _checkpointer.delete_thread(evicted)
    _thread_ids[thread_id] = True


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")

    # Same loop as every previous version — create_react_agent is what
    # langchain's create_agent builds on (checkpointer shared across invocations)
    graph = create_react_agent(
        get_or_create_model(),
        tools=tools,
        prompt=DEFAULT_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )

    # The runtime hands us a session_id per conversation — it becomes the
    # thread_id, exactly like /thread did in the v0.3 chat REPL.
    prompt = payload.get("prompt", "What can you help me with?")
    session_id = getattr(context, "session_id", "default-session")
    touch_thread(session_id)
    log.info(f"Agent input: {prompt}")

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": session_id}},
    )

    output = result["messages"][-1].text  # text blocks only — .content can be a list
    log.info(f"Agent output: {output}")
    return {"result": output}


if __name__ == "__main__":
    app.run()
