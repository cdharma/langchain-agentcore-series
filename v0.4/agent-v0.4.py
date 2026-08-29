# =============================================================================
# v0.4 — real tools: a live weather API + tools from an MCP server
#
# What changed vs v0.3:
#   1. get_weather() no longer returns a fake string — it calls Open-Meteo,
#      a free weather API that needs no API key. Crucially, the tool's
#      *interface* (name, docstring, type hints) is identical to the fake one,
#      so the model can't tell the difference. Only the results got real.
#   2. An MCP (Model Context Protocol) server contributes 14 extra tools we
#      never wrote — read_file, list_directory, search_files, ... MCP is a
#      standard plug: any MCP server's tools can be handed to any agent.
#   3. MCP tools are async, so the loop calls `await agent.ainvoke(...)`
#      instead of `agent.invoke(...)`. The agent loop itself is unchanged.
#
# Chat commands:
#   /tools           list every tool the agent has
#   /thread <name>   switch conversation (fresh memory, see v0.3)
#   /quit            exit
# =============================================================================
import asyncio
import os
from pathlib import Path

# --- .env loading (same trick as v0.2/v0.3) ---------------------------------
# Machine-specific config (AWS_PROFILE, LangSmith key/endpoint) lives in a
# gitignored .env next to this file. It MUST be loaded before the langchain
# imports below: langsmith caches its environment at import time, so setting
# LANGSMITH_ENDPOINT afterwards would be silently ignored.
env_file = Path(__file__).with_name(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            # setdefault → real environment variables still win over .env
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    os.environ.setdefault("LANGSMITH_TRACING", "true")

import requests
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

REPO_ROOT = Path(__file__).parent.parent


def get_weather(city: str) -> str:
    """Get the CURRENT, real weather for a city (live Open-Meteo data)."""
    # The model never sees this body — it only sees the function name, the
    # docstring above, and the JSON schema derived from the type hints.
    # Call 1: geocoding — turn "Mumbai" into latitude/longitude.
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=10,
    ).json()
    if not geo.get("results"):
        # Tool errors should be returned as text — the model reads this and
        # can recover (ask the user to rephrase) instead of crashing the run.
        return f"Could not find a place called {city!r}."
    place = geo["results"][0]
    # Call 2: current conditions at those coordinates.
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
        }, timeout=10,
    ).json()["current"]
    # Whatever this returns becomes the ToolMessage the model reads next turn —
    # so compress the API's raw JSON into one self-describing sentence instead
    # of dumping the payload (fewer tokens, and nothing for the model to misread).
    return (
        f"{place['name']}, {place.get('country', '')}: {wx['temperature_2m']}°C "
        f"(feels like {wx['apparent_temperature']}°C), humidity {wx['relative_humidity_2m']}%, "
        f"wind {wx['wind_speed_10m']} km/h"
    )

# --- MCP servers -------------------------------------------------------------
# Each entry describes how to reach one MCP server. "stdio" transport means:
# spawn this command as a subprocess and speak JSON-RPC over its stdin/stdout.
# The last arg scopes the filesystem server to THIS repo — it cannot see or
# touch anything outside it. (Note: it does include write tools!)
MCP_SERVERS = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(REPO_ROOT)],
    },
}

# region pinned: us.anthropic.* inference profiles only resolve from US
# regions, and a stray shell AWS_REGION would otherwise override the profile
model = ChatBedrockConverse(model="us.anthropic.claude-opus-5", region_name="us-east-1")


def tools_used(messages: list) -> list[str]:
    """Names of tools called since the last human message.

    The agent's state is just a message list; every tool execution leaves a
    ToolMessage behind. Walking backwards until the last human turn tells us
    exactly which tools this reply used — handy to display in the UI.
    """
    names = []
    for msg in reversed(messages):
        if msg.type == "human":
            break
        if msg.type == "tool":
            names.append(msg.name)
    return list(reversed(names))


async def main() -> None:
    console = Console()

    # One PERSISTENT session for the whole chat: the MCP server process starts
    # once here and every tool call reuses it. The adapter's default
    # (get_tools() with no session) is stateless — it spawns a fresh server
    # per tool call, which is slower and splatters startup banners mid-chat.
    client = MultiServerMCPClient(MCP_SERVERS)
    async with client.session("filesystem") as session:
        mcp_tools = await load_mcp_tools(session)

        # Our own tool and the MCP tools go into the same flat list — to the
        # model they are indistinguishable: each is name + description + schema.
        tools = [get_weather, *mcp_tools]
        agent = create_agent(model=model, tools=tools, checkpointer=InMemorySaver())
        await chat(console, agent, tools)


async def chat(console: Console, agent, tools) -> None:
    console.print(Panel(
        f"[bold]{len(tools)} tools[/]: get_weather (live API) + {len(tools) - 1} from the MCP filesystem server\n"
        "[bold]/tools[/] to list them · [bold]/thread <name>[/] to switch memory · [bold]/quit[/] to exit",
        title="agent v0.4 — real tools", border_style="yellow",
    ))
    thread = "demo"  # conversation key for the checkpointer (see v0.3)
    while True:
        try:
            user = console.input(f"[bold green]you[/] [cyan]\\[{thread}][/]> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue

        # Anything starting with "/" is handled locally — never sent to the
        # model, so typos like /thead cost nothing.
        if user.startswith("/"):
            if user in ("/quit", "/exit"):
                break
            if user == "/tools":
                for t in tools:
                    # plain functions have __name__/__doc__; MCP tools are
                    # StructuredTool objects with .name/.description
                    name = getattr(t, "name", None) or getattr(t, "__name__", "?")
                    desc = (getattr(t, "description", None) or getattr(t, "__doc__", None) or "").split("\n")[0]
                    console.print(f"  [bold]{name}[/] [dim]— {desc}[/]")
            elif user.startswith("/thread ") and len(user.split()) > 1:
                thread = user.split(maxsplit=1)[1]
                console.print(f"[magenta]switched to thread [bold]{thread!r}[/bold] — fresh memory if new[/]")
            else:
                console.print(f"[red]unknown command {user.split()[0]!r}[/] — try [bold]/tools[/], [bold]/thread <name>[/] or [bold]/quit[/]")
            continue

        # ainvoke, not invoke: MCP tools are async and must run on the event
        # loop. Everything else about the call is identical to v0.3.
        try:
            with console.status("[dim]thinking…[/]", spinner="dots"):
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user}]},
                    {"configurable": {"thread_id": thread}},
                )
        except Exception as e:  # transient errors (Bedrock throttling, network) must not kill the chat
            console.print(Panel(f"{type(e).__name__}: {e}\n\nyour conversations are safe — just try again", title="error", border_style="red"))
            continue

        used = tools_used(result["messages"])
        if used:
            console.print(f"[dim]tools used: {', '.join(used)}[/]")
        # the model answers in Markdown (**bold**, tables) — render it properly
        console.print(Panel(Markdown(result["messages"][-1].text), title="agent", border_style="bright_blue"))


if __name__ == "__main__":
    asyncio.run(main())
