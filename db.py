import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

_client: Optional[AsyncIOMotorClient] = None
_sync_client: Optional[MongoClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    return _client


def get_collection(name: str):
    return get_client()["estimate_db"][name]


def get_sync_client() -> MongoClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(os.environ["MONGO_URI"])
    return _sync_client


def get_sync_collection(name: str):
    return get_sync_client()["estimate_db"][name]
