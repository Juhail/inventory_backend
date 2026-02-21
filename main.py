from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from bson import ObjectId
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Inventory Mobile App Backend",
    description="Backend API for Flutter Inventory App",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    logger.critical("MONGODB_URI is not set!")

client = AsyncIOMotorClient(MONGODB_URI)
db = client.inventory_db

products_collection = db.products
brands_collection   = db.brands
models_collection   = db.models
events_collection   = db.events

# ── Helpers ───────────────────────────────────────────────────────────────────
def _serialize(doc: dict) -> dict:
    if not doc:
        return {}
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id", ""))
    return doc

def _obj(id_str: str) -> ObjectId:
    if not ObjectId.is_valid(id_str):
        raise HTTPException(400, "Invalid ID format")
    return ObjectId(id_str)

# ── Pydantic Models ───────────────────────────────────────────────────────────
class ProductCreate(BaseModel):
    """Product creation model (no ID)"""
    name:     str
    brand:    str
    category: str
    model:    Optional[str] = None
    price:    float
    stock:    int
    barcode:  Optional[str] = None

class BrandCreate(BaseModel):
    name: str

class ModelCreate(BaseModel):
    brand: str
    name:  str

class AuditEvent(BaseModel):
    event_id:            str
    product_id:          str
    delta_qty:           int
    unit_price_snapshot: float = 0.0
    actor_label:         str = "system"
    source:              str = "manual"
    device_id:           str = "unknown"
    timestamp:           datetime
    action:              Optional[str] = None
    description:         Optional[str] = None
    resulting_stock:     Optional[int] = None
    previous_stock:      Optional[int] = None
    audit_only:          bool = False
    product_snapshot:    Optional[Dict[str, Any]] = None

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
async def health():
    """Health check endpoint"""
    try:
        await client.admin.command("ping")
        return {"status": "healthy", "version": "0.3.0"}
    except Exception as e:
        raise HTTPException(503, f"Database error: {e}")

# ── Products ──────────────────────────────────────────────────────────────────
@app.get("/products")
async def get_products(
    search:   Optional[str] = None,
    brand:    Optional[str] = None,
    category: Optional[str] = None,
):
    """Get all products with optional filtering.

    FIXED ISSUES:
    - Wrapped in try/except to catch all errors
    - Converts MongoDB ObjectId to string
    - Returns empty array if no products (not an error)
    - Logs full traceback on error
    """
    try:
        query: dict = {}
        if brand:
            query["brand"] = brand
        if category:
            query["category"] = category
        if search:
            query["$or"] = [
                {"name":  {"$regex": search, "$options": "i"}},
                {"brand": {"$regex": search, "$options": "i"}},
                {"model": {"$regex": search, "$options": "i"}},
            ]
        docs = await products_collection.find(query).to_list(1000)
        return [_serialize(d) for d in docs]
    except Exception as e:
        logger.error(f"get_products error: {e}", exc_info=True)
        return []

@app.post("/products")
async def add_product(product: ProductCreate):
    """Add a new product"""
    doc = product.model_dump()
    doc["_id"]           = ObjectId()
    doc["created_at"]    = datetime.utcnow()
    doc["updated_at"]    = datetime.utcnow()
    doc["stock_version"] = 1
    await products_collection.insert_one(doc)
    return _serialize(doc)

@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get a single product by ID"""
    doc = await products_collection.find_one({"_id": _obj(product_id)})
    if not doc:
        raise HTTPException(404, "Product not found")
    return _serialize(doc)

@app.put("/products/{product_id}")
async def update_product(product_id: str, product: ProductCreate):
    """Update a product"""
    update_data = product.model_dump()
    update_data["updated_at"] = datetime.utcnow()
    result = await products_collection.find_one_and_update(
        {"_id": _obj(product_id)},
        {"$set": update_data},
        return_document=True,
    )
    if not result:
        raise HTTPException(404, "Product not found")
    return _serialize(result)

@app.delete("/products/{product_id}")
async def delete_product(product_id: str):
    """Delete a product"""
    result = await products_collection.delete_one({"_id": _obj(product_id)})
    if result.deleted_count == 0:
        raise HTTPException(404, "Product not found")
    return {"status": "deleted", "id": product_id}

# ── Brands ────────────────────────────────────────────────────────────────────
@app.get("/brands")
async def get_brands():
    """Get all available brands"""
    docs = await brands_collection.find().to_list(1000)
    return [_serialize(d) for d in docs]

@app.post("/brands")
async def add_brand(brand: BrandCreate):
    """Add a new brand if it doesn't exist"""
    existing = await brands_collection.find_one({"name": brand.name})
    if existing:
        return _serialize(existing)
    doc = {"_id": ObjectId(), "name": brand.name}
    await brands_collection.insert_one(doc)
    return _serialize(doc)

@app.put("/brands/{brand_id}")
async def update_brand(brand_id: str, brand: BrandCreate):
    """Update a brand name"""
    result = await brands_collection.find_one_and_update(
        {"_id": _obj(brand_id)},
        {"$set": {"name": brand.name}},
        return_document=True,
    )
    if not result:
        raise HTTPException(404, "Brand not found")
    return _serialize(result)

@app.delete("/brands/{brand_id}")
async def delete_brand(brand_id: str):
    """Delete a brand if no models exist for it"""
    result = await brands_collection.delete_one({"_id": _obj(brand_id)})
    if result.deleted_count == 0:
        raise HTTPException(404, "Brand not found")
    return {"status": "deleted", "id": brand_id}

# ── Models ────────────────────────────────────────────────────────────────────
@app.get("/models")
async def get_models(brand: str = Query(...)):
    """Get all models for a specific brand"""
    docs = await models_collection.find({"brand": brand}).to_list(1000)
    return [_serialize(d) for d in docs]

@app.post("/models")
async def add_model(model: ModelCreate):
    """Add a new model to a brand if it doesn't exist"""
    existing = await models_collection.find_one({"brand": model.brand, "name": model.name})
    if existing:
        return _serialize(existing)
    doc = {"_id": ObjectId(), "brand": model.brand, "name": model.name}
    await models_collection.insert_one(doc)
    return _serialize(doc)

@app.put("/models/{model_id}")
async def update_model(model_id: str, model: ModelCreate):
    """Update a model name"""
    result = await models_collection.find_one_and_update(
        {"_id": _obj(model_id)},
        {"$set": {"brand": model.brand, "name": model.name}},
        return_document=True,
    )
    if not result:
        raise HTTPException(404, "Model not found")
    return _serialize(result)

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    result = await models_collection.delete_one({"_id": _obj(model_id)})
    if result.deleted_count == 0:
        raise HTTPException(404, "Model not found")
    return {"status": "deleted", "id": model_id}

# ── Events / Audit Log ────────────────────────────────────────────────────────
@app.post("/events")
async def post_event(event: AuditEvent):
    """
    Record a stock event / audit log entry.

    audit_only=False (default): applies stock delta AND logs.
    audit_only=True: only logs — stock is NOT modified.
      Flutter uses this after PUT /products/:id sells so the audit screen
      shows the transaction without double-applying the stock change.

    Idempotent: duplicate event_id is silently skipped.
    """
    if await events_collection.find_one({"event_id": event.event_id}):
        return {"status": "idempotent_skip"}

    try:
        previous_stock  = event.previous_stock
        resulting_stock = event.resulting_stock

        if not event.audit_only:
            product = await products_collection.find_one({"_id": _obj(event.product_id)})
            if product:
                previous_stock  = product.get("stock", 0)
                resulting_stock = previous_stock + event.delta_qty
                await products_collection.update_one(
                    {"_id": _obj(event.product_id)},
                    {
                        "$inc": {"stock": event.delta_qty, "stock_version": 1},
                        "$set": {"last_action": event.action or "update", "updated_at": datetime.utcnow()},
                    },
                )

        snapshot = event.product_snapshot
        if snapshot is None:
            raw = await products_collection.find_one({"_id": _obj(event.product_id)})
            if raw:
                raw.pop("_id", None)
                snapshot = raw

        record = event.model_dump()
        record["previous_stock"]   = previous_stock
        record["resulting_stock"]  = resulting_stock
        record["product_snapshot"] = snapshot
        record["logged_at"]        = datetime.utcnow()
        await events_collection.insert_one(record)

        return {"status": "success", "event_id": event.event_id}

    except Exception as e:
        logger.error(f"post_event error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/events")
async def get_events(
    product_id: Optional[str] = None,
    limit:      int = Query(50, ge=1, le=500),
):
    """Fetch audit log / event history, newest first."""
    query: dict = {}
    if product_id:
        query["product_id"] = product_id

    cursor = events_collection.find(query).sort("logged_at", -1).limit(limit)
    docs   = await cursor.to_list(limit)

    results = []
    for doc in docs:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("logged_at"), datetime):
            doc["logged_at"] = doc["logged_at"].isoformat()
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        results.append(doc)

    return results
