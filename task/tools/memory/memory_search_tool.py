import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "Searches long-term memories about the user using semantic similarity. "
            "Use this tool to recall previously stored information about the user such as preferences, "
            "personal details, goals, or context. The search uses meaning-based matching, so you can "
            "search with questions or keywords. Example queries: 'What programming languages does the user like?', "
            "'user's location', 'goals and plans'. Always search memories before answering questions about "
            "the user's preferences or past conversations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)

        # 2. Get `query` from arguments
        query = arguments["query"]

        # 3. Get `top_k` from arguments, default is 5
        top_k = arguments.get("top_k", 5)

        # 4. Call `memory_store` `search_memories`
        results: list[MemoryData] = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )

        # 5. If results are empty then set `final_result` as "No memories found.",
        #    otherwise iterate through results and collect content, category and topics (if present) in markdown format
        if not results:
            final_result = "No memories found."
        else:
            lines = []
            for i, memory in enumerate(results, 1):
                lines.append(f"**{i}. {memory.content}**")
                lines.append(f"   - Category: {memory.category}")
                if memory.topics:
                    lines.append(f"   - Topics: {', '.join(memory.topics)}")
            final_result = "\n".join(lines)

        # 6. Add result to stage as markdown text
        stage = tool_call_params.stage
        stage.append_content("## Memory Search\n")
        stage.append_content(f"**Query**: {query}\n\n")
        stage.append_content(f"**Results**:\n{final_result}\n")

        # 7. Return result
        return final_result
