import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def delete_legacy_product():
    client = AsyncIOMotorClient("mongodb://127.0.0.1:27017")
    db = client.inventory_db
    collection = db.products

    # 1. Delete the specific legacy product
    target_name = "iPhone 11 Battery"
    result = await collection.delete_many({"name": target_name})
    
    print(f"Deleted {result.deleted_count} product(s) with name '{target_name}'")

    # 2. Verify remaining products
    print("\nRemaining products:")
    async for doc in collection.find({}):
        print(f"- {doc.get('name')} (Model: {doc.get('model', 'MISSING')})")

if __name__ == "__main__":
    asyncio.run(delete_legacy_product())
