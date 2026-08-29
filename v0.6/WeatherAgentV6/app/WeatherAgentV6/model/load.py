from langchain_aws import ChatBedrockConverse

# Same model as the whole series. Region pinned: us.anthropic.* inference
# profiles only resolve from US regions, and pinning keeps local
# `agentcore dev` immune to stray shell AWS_REGION values.
MODEL_ID = "us.anthropic.claude-opus-5"


def load_model() -> ChatBedrockConverse:
    """Get Bedrock model client using IAM credentials."""
    return ChatBedrockConverse(model=MODEL_ID, region_name="us-east-1")
