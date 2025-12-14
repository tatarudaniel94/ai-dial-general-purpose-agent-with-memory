from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class DeleteMemoryTool(BaseTool):
    """
    Tool for deleting all long-term memories about the user.

    This permanently removes all stored memories from the system.
    Use with caution - this action cannot be undone.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "delete_all_memories"

    @property
    def description(self) -> str:
        return (
            "Permanently deletes ALL long-term memories stored about the user. "
            "This action cannot be undone. Use only when the user explicitly requests "
            "to clear or delete all their stored memories. Before executing, confirm with "
            "the user that they want to delete all memories as this will remove all previously "
            "saved preferences, personal information, goals, and context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {}
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        # 1. Call `memory_store` `delete_all_memories`
        result = await self.memory_store.delete_all_memories(
            api_key=tool_call_params.api_key
        )

        # 2. Add result to stage
        stage = tool_call_params.stage
        stage.append_content("## Memory Deletion\n")
        stage.append_content(f"**Result**: {result}\n")

        # 3. Return result
        return result