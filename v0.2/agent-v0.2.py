# v0.2 — same agent as v0.1, plus 4 levels of logging:
#   1. step trace (always on): every message in the loop, pretty-printed
#   2. --debug : LangChain debug mode, full inputs/outputs as JSON
#   3. --wire  : botocore logs of the raw HTTPS calls to bedrock-runtime
#   4. LangSmith: no code needed — set LANGSMITH_TRACING=true and
#      LANGSMITH_API_KEY in the environment and runs appear in the web UI
import argparse
import os
from pathlib import Path

# load v0.2/.env BEFORE importing langchain — langsmith caches env at import time
# (stdlib parse; real env vars still take precedence)
env_file = Path(__file__).with_name(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    # having a .env here means "trace me" — key alone doesn't enable it
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse

parser = argparse.ArgumentParser(description="Weather agent with switchable logging")
parser.add_argument("--debug", action="store_true", help="LangChain debug mode (very verbose)")
parser.add_argument("--wire", action="store_true", help="log raw HTTP requests to Bedrock")
args = parser.parse_args()

if args.debug:
    from langchain_core.globals import set_debug
    set_debug(True)

if args.wire:
    import boto3
    boto3.set_stream_logger("botocore.endpoint")

if os.environ.get("LANGSMITH_TRACING") == "true":
    print("[LangSmith tracing enabled — see smith.langchain.com]")


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# region pinned: us.anthropic.* inference profiles only resolve from US regions,
# and shell env (AWS_REGION=ap-south-1 in ~/.zshrc) would otherwise override the profile
model = ChatBedrockConverse(model="us.anthropic.claude-opus-5", region_name="us-east-1")

agent = create_agent(model=model, tools=[get_weather])

if __name__ == "__main__":
    for step in agent.stream(
        {"messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
