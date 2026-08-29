# v0.3 — memory & multi-turn: same agent + a checkpointer.
# The checkpointer saves the message history per thread_id after every turn;
# the next invoke on the same thread resumes with the full conversation.
#   /thread <name>  switch conversation (fresh memory)
#   /quit           exit
import os
from pathlib import Path

# load v0.3/.env BEFORE importing langchain — langsmith caches env at import time
env_file = Path(__file__).with_name(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# region pinned: us.anthropic.* inference profiles only resolve from US regions
model = ChatBedrockConverse(model="us.anthropic.claude-opus-5", region_name="us-east-1")

agent = create_agent(
    model=model,
    tools=[get_weather],
    checkpointer=InMemorySaver(),  # <- the whole memory feature
)

if __name__ == "__main__":
    console = Console()
    thread = "demo"
    console.print(Panel(
        "chat with memory — [bold]/thread <name>[/] to switch conversations, [bold]/quit[/] to exit",
        title="agent v0.3", border_style="yellow",
    ))
    while True:
        try:
            user = console.input(f"[bold green]you[/] [cyan]\\[{thread}][/]> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.startswith("/"):
            if user in ("/quit", "/exit"):
                break
            if user.startswith("/thread ") and len(user.split()) > 1:
                thread = user.split(maxsplit=1)[1]
                console.print(f"[magenta]switched to thread [bold]{thread!r}[/bold] — fresh memory if new[/]")
            else:
                console.print(f"[red]unknown command {user.split()[0]!r}[/] — try [bold]/thread <name>[/] or [bold]/quit[/]")
            continue
        try:
            with console.status("[dim]thinking…[/]", spinner="dots"):
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user}]},
                    {"configurable": {"thread_id": thread}},
                )
        except Exception as e:  # transient Bedrock errors must not kill the chat (memory lives in-process)
            console.print(Panel(f"{type(e).__name__}: {e}\n\nyour conversations are safe — just try again", title="error", border_style="red"))
            continue
        # the model answers in Markdown (**bold**, tables) — render it properly
        console.print(Panel(Markdown(result["messages"][-1].text), title="agent", border_style="bright_blue"))
