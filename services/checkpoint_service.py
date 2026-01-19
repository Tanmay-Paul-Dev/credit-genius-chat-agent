from typing import Any, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint


class MongoCheckpointSaver(BaseCheckpointSaver):
    def __init__(
        self, mongo_uri: str, db_name: str = "chatbot", collection: str = "checkpoints"
    ):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.collection = self.client[db_name][collection]

    async def get(self, thread_id: str) -> Optional[Checkpoint]:
        doc = await self.collection.find_one({"_id": thread_id})
        if not doc:
            return None
        return Checkpoint(
            thread_id=thread_id,
            checkpoint=doc["checkpoint"],
            metadata=doc.get("metadata", {}),
        )

    async def put(self, checkpoint: Checkpoint) -> None:
        await self.collection.update_one(
            {"_id": checkpoint.thread_id},
            {
                "$set": {
                    "checkpoint": checkpoint.checkpoint,
                    "metadata": checkpoint.metadata,
                }
            },
            upsert=True,
        )
