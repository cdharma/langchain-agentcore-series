# From 18 Lines to Production — LangChain Agents on AWS

A tutorial series: one LangChain agent on Amazon Bedrock, grown from the simplest
possible loop into a production service on Amazon Bedrock AgentCore.

| Version | What it adds |
|---|---|
| [v0.1](v0.1/) | an agent in 18 lines — `create_agent` + one plain function |
| [v0.2](v0.2/) | observability — step trace, LangChain debug, raw HTTP, LangSmith |
| [v0.3](v0.3/) | memory — a checkpointer and `thread_id` per conversation |
| [v0.4](v0.4/) | real tools — a live weather API + an MCP server's tools |
| [v0.5](v0.5/) | deployed — AgentCore Runtime, one session per microVM |
| [v0.6](v0.6/) | durable memory — AgentCore Memory, facts recalled across sessions |
| [v0.7](v0.7/) | AgentCore Gateway — tools become infrastructure (OpenAPI + Lambda targets) |

## Run the laptop versions

```bash
pip install -r requirements.txt

python v0.1/agent-v0.1.py            # the agent
python v0.2/agent-v0.2.py --wire     # + raw HTTPS to Bedrock
python v0.3/agent-v0.3.py            # chat with memory
python v0.4/agent-v0.4.py            # + real API + MCP tools
```

Needs AWS credentials with Bedrock access to `us.anthropic.claude-opus-5`.
LangSmith tracing is optional — drop a `.env` next to the script with
`LANGSMITH_API_KEY` (and `LANGSMITH_ENDPOINT` if your account isn't on GCP-US).

## Run the deployed versions (v0.5+)

Each is a self-contained [AgentCore CLI](https://www.npmjs.com/package/@aws/agentcore)
project with its own stack. Set your account in `agentcore/aws-targets.json`
(see `aws-targets.example.json`), then:

```bash
cd v0.7/WeatherAgentV7
agentcore dev        # local dev server + chat UI
agentcore deploy -y  # ~4 min: IAM, runtime, memory, gateway
agentcore invoke "What's the weather in Mumbai?"
```

v0.7 also needs the forecast Lambda — see [lambda/forecast/README.md](v0.7/WeatherAgentV7/lambda/forecast/README.md).

## The demo hub

`serve.py` serves an overview page for the whole series plus a themed chat UI
that proxies to whatever `agentcore dev` is running:

```bash
AGENT_LABEL="WeatherAgentV7 · v0.7" python3 serve.py   # http://localhost:8321
```

The Excalidraw explainer boards used in the videos aren't part of this repo, so
the board panels on that page show a placeholder.
