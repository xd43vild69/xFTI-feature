import asyncio
from fti_agent.config import load_config
from fti_agent.mcp_host import mcp_session

async def main():
    config = load_config()
    print(f"Config: model={config.model}, api_base={config.api_base}")
    print(f"Hub dir: {config.hub_dir}")
    
    try:
        async with mcp_session(config) as (session, tools):
            print(f"✓ MCP server connected")
            print(f"✓ {len(tools)} tools discovered:")
            for tool in sorted(tools, key=lambda t: t.name):
                print(f"  - {tool.name}")
        print(f"✓ All good!")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())