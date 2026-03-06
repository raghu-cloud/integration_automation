import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async def main():
    agent = AgentDefinition(
        description="test",
        prompt="Reply 'hello'",
        model="sonnet"
    )
    opts = ClaudeAgentOptions(
        agents={"test_agent": agent},
        allowed_tools=["Task"],
        model="haiku"
    )
    prompt = "Use the Task tool to call test_agent and tell it to say hello"
    async for msg in query(prompt=prompt, options=opts):
        if getattr(msg, 'subtype', None) == "result":
            print("RESULT USAGE:", getattr(msg, 'usage', None))
            print("RESULT TOTAL COST USD:", getattr(msg, 'total_cost_usd', None))

if __name__ == "__main__":
    asyncio.run(main())
