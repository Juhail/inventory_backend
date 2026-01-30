from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import uvicorn

# --------------------
# App setup
# --------------------
app = FastAPI(
    title="Inventory Mobile App Backend",
    description="Backend API for Flutter Inventory App",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# MongoDB
# --------------------
import os

# --------------------
# MongoDB
# --------------------
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client.inventory_db
products_collection = db.products

# --------------------
# Models (data shape)
# --------------------
from typing import Optional

class Product(BaseModel):
    name: str
    brand: str
    category: str
    model: Optional[str] = None
    price: float
    stock: int

# --------------------
# Routes
# --------------------
@app.get("/")
async def health():
    return {"status": "ok", "message": "Inventory Backend is running"}

@app.post("/products")
async def add_product(product: Product):
    from datetime import datetime
    
    product_dict = product.dict()
    product_dict["createdAt"] = datetime.utcnow()
    
    result = await products_collection.insert_one(product_dict)
    
    product_dict["_id"] = str(result.inserted_id)
    
    return product_dict

@app.get("/products")
async def get_products(search: str = None, brand: str = None, category: str = None):
    query = {}
    
    if search:
        query["name"] = {"$regex": search, "$options": "i"}
    
    if brand:
        query["brand"] = brand
    
    if category:
        query["category"] = category
    
    products = []
    async for item in products_collection.find(query):
        item["_id"] = str(item["_id"])
        products.append(item)
    return products

@app.delete("/products/{id}")
async def delete_product(id: str):
    from bson import ObjectId
    
    try:
        obj_id = ObjectId(id)
    except:
        return {"error": "Invalid ID format"}, 400
        
    result = await products_collection.delete_one({"_id": obj_id})
    
    if result.deleted_count == 1:
        print(f"Deleted product with ID: {id}")
        return {"status": "success", "message": f"Product {id} deleted"}
    else:
        return {"error": "Product not found"}, 404

@app.put("/products/{id}")
async def update_product(id: str, product: Product):
    from bson import ObjectId
    
    try:
        obj_id = ObjectId(id)
    except:
        return {"error": "Invalid ID format"}, 400

    update_data = {
        "name": product.name,
        "brand": product.brand,
        "category": product.category,
        "price": product.price,
        "stock": product.stock
    }
    
    if product.model:
        update_data["model"] = product.model
    else:
        update_data["model"] = None

    result = await products_collection.update_one(
        {"_id": obj_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        return {"error": "Product not found"}, 404
        
    # Return the updated document
    updated_doc = await products_collection.find_one({"_id": obj_id})
    updated_doc["_id"] = str(updated_doc["_id"])
    return updated_doc
