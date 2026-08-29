from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# region pinned: us.anthropic.* inference profiles only resolve from US regions,
# and shell env (AWS_REGION=ap-south-1 in ~/.zshrc) would otherwise override the profile
model = ChatBedrockConverse(model="us.anthropic.claude-opus-5", region_name="us-east-1")

agent = create_agent(model=model, tools=[get_weather])

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]}
    )
    print(result["messages"][-1].text)
