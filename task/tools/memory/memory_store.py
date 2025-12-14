import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        # 1. Set endpoint
        self.endpoint = endpoint
        # 2. Create SentenceTransformer as model, model name is `all-MiniLM-L6-v2`
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # 3. Create cache, dict of str and MemoryCollection
        self._cache: dict[str, MemoryCollection] = {}
        # 4. Set FAISS threads to 1 for debug mode compatibility
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        # 1. Get DIAL app home path
        app_home = await dial_client.my_appdata_home()
        # 2. Return string with path format: `files/{bucket_with_app_home}/__long-memories/data.json`
        return f"files/{(app_home / '__long-memories' / 'data.json').as_posix()}"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        # 1. Create AsyncDial client (api_version is 2025-01-01-preview)
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        # 2. Get memory file path
        memory_file_path = await self._get_memory_file_path(dial_client)

        # 3. Check cache
        if memory_file_path in self._cache:
            return self._cache[memory_file_path]

        # 4. Try to load from file, or create empty collection
        try:
            # Download file content
            response = await dial_client.files.download(memory_file_path)
            # Get content and decode with 'utf-8'
            content = response.get_content().decode('utf-8')
            # Load content with json
            data = json.loads(content)
            # Create MemoryCollection
            collection = MemoryCollection.model_validate(data)
        except Exception:
            # Create empty MemoryCollection
            collection = MemoryCollection(
                memories=[],
                updated_at=datetime.now(UTC)
            )

        # Put to cache
        self._cache[memory_file_path] = collection
        # 5. Return created MemoryCollection
        return collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        # 1. Create AsyncDial client
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        # 2. Get memory file path
        memory_file_path = await self._get_memory_file_path(dial_client)
        # 3. Update `updated_at` of memories (now)
        memories.updated_at = datetime.now(UTC)
        # 4. Converts memories to json string (no indentation to minimize size)
        json_data = memories.model_dump_json()
        # Upload file
        await dial_client.files.upload(url=memory_file_path, file=json_data.encode('utf-8'))
        # 5. Put to cache
        self._cache[memory_file_path] = memories

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        # 1. Load memories
        collection = await self._load_memories(api_key)
        # 2. Make encodings for content with embedding model
        embedding = self.model.encode([content])[0].tolist()
        # 3. Create Memory
        memory_data = MemoryData(
            id=int(datetime.now(UTC).timestamp()),
            content=content,
            importance=importance,
            category=category,
            topics=topics
        )
        memory = Memory(data=memory_data, embedding=embedding)
        # 4. Add to memories
        collection.memories.append(memory)
        # 5. Save memories
        await self._save_memories(api_key, collection)
        # 6. Return success information
        return f"Memory successfully stored: '{content}'"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        # 1. Load memories
        collection = await self._load_memories(api_key)
        # 2. If they are empty return empty array
        if not collection.memories:
            return []

        # 3. Check if deduplication is needed
        if self._needs_deduplication(collection):
            collection = await self._deduplicate_and_save(api_key, collection)

        # 4. Make vector search
        query_embedding = self.model.encode([query])[0]

        # Build embeddings matrix
        embeddings = np.array([m.embedding for m in collection.memories], dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        index.add(embeddings)

        # Search
        k = min(top_k, len(collection.memories))
        distances, indices = index.search(query_vec, k)

        # 5. Return top_k MemoryData based on vector search
        results = [collection.memories[idx].data for idx in indices[0]]
        return results

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        # Criteria: (length > 10 and >24 hours since last dedup) or (length > 10 and never deduped)
        if len(collection.memories) <= 10:
            return False

        if collection.last_deduplicated_at is None:
            return True

        hours_since_dedup = (datetime.now(UTC) - collection.last_deduplicated_at).total_seconds() / 3600
        return hours_since_dedup > self.DEDUP_INTERVAL_HOURS

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        # 1. Make fast deduplication
        deduplicated_memories = self._deduplicate_fast(collection.memories)
        collection.memories = deduplicated_memories
        # 2. Update last_deduplicated_at as now
        collection.last_deduplicated_at = datetime.now(UTC)
        # 3. Save deduplicated memories
        await self._save_memories(api_key, collection)
        # 4. Return deduplicated collection
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast O(n log n) deduplication using FAISS HNSW index with cosine similarity.

        Strategy:
        - Use HNSW (Hierarchical Navigable Small World) for O(log n) approximate nearest neighbor search
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        if len(memories) <= 1:
            return memories

        SIMILARITY_THRESHOLD = 0.75

        # Build embeddings matrix
        embeddings = np.array([m.embedding for m in memories], dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create HNSW index for O(n log n) complexity
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter (neighbors per layer)
        index.hnsw.efConstruction = 40  # Construction-time search depth
        index.hnsw.efSearch = 16  # Query-time search depth
        index.add(embeddings)

        # Search for k nearest neighbors for each memory
        # k=min(10, n) to find potential duplicates efficiently
        k = min(10, len(memories))
        distances, indices = index.search(embeddings, k)

        # Track which memories to keep (not marked as duplicates)
        to_remove = set()

        # Sort memories by importance (descending) - higher importance survives
        importance_order = sorted(range(len(memories)), key=lambda i: memories[i].data.importance, reverse=True)

        for i in importance_order:
            if i in to_remove:
                continue

            # Check neighbors of this memory
            for j_idx in range(1, k):  # Skip index 0 (self)
                neighbor_idx = indices[i][j_idx]
                similarity = distances[i][j_idx]

                # If similar enough and not already removed
                if similarity > SIMILARITY_THRESHOLD and neighbor_idx not in to_remove:
                    # Mark the neighbor as duplicate (current has higher or equal importance)
                    if memories[i].data.importance >= memories[neighbor_idx].data.importance:
                        to_remove.add(neighbor_idx)
                    else:
                        to_remove.add(i)
                        break

        # Return memories that weren't removed
        return [m for idx, m in enumerate(memories) if idx not in to_remove]

    async def delete_all_memories(self, api_key: str) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        # 1. Create AsyncDial client
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        # 2. Get memory file path
        memory_file_path = await self._get_memory_file_path(dial_client)
        # 3. Delete file
        try:
            await dial_client.files.delete(memory_file_path)
        except Exception:
            pass  # File might not exist
        # Clear cache
        if memory_file_path in self._cache:
            del self._cache[memory_file_path]
        # 4. Return info about successful memory deletion
        return "All memories have been successfully deleted."
