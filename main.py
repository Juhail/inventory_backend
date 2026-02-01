from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional, List
from bson import ObjectId
import os
import logging
import traceback

# ============================================================================
# STEP 1: ENABLE FULL ERROR LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Inventory Mobile App Backend",
    description="Backend API for Flutter Inventory App",
    version="0.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    logger.error("=" * 80)
    logger.error("CRITICAL: MONGODB_URI environment variable is NOT set!")
    logger.error("Please set MONGODB_URI in Render dashboard → Environment tab")
    logger.error("=" * 80)
    # Use localhost as fallback (will fail on Render, but shows the issue clearly)
    MONGODB_URI = "mongodb://localhost:27017"
    logger.warning(f"Using fallback MongoDB URI: {MONGODB_URI}")
else:
    # Log only the host part (hide password)
    uri_parts = MONGODB_URI.split("@")
    if len(uri_parts) > 1:
        logger.info(f"MongoDB URI configured: ...@{uri_parts[1]}")
    else:
        logger.info("MongoDB URI configured (local)")

logger.info(f"Connecting to MongoDB...")
client = AsyncIOMotorClient(MONGODB_URI)
db = client.inventory_db
products_collection = db.products

# ============================================================================
# STEP 3: MONGODB OBJECTID SERIALIZATION FIX
# ============================================================================
class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic that serializes to string"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


# ============================================================================
# STEP 4: VERIFY PRODUCT SCHEMA - MAKE FIELDS OPTIONAL WHERE NEEDED
# ============================================================================
class Product(BaseModel):
    """Product model matching MongoDB schema"""
    id: Optional[str] = Field(None, alias="_id")
    name: str
    brand: str
    category: str
    model: Optional[str] = None  # Optional field
    price: float
    stock: int

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}


class ProductCreate(BaseModel):
    """Product creation model (no ID)"""
    name: str
    brand: str
    category: str
    model: Optional[str] = None
    price: float
    stock: int


# ============================================================================
# HELPER FUNCTION: CONVERT MONGODB DOC TO JSON-SAFE DICT
# ============================================================================
def product_helper(product) -> dict:
    """
    Convert MongoDB document to JSON-serializable dict.
    CRITICAL: Converts ObjectId to string to prevent 500 errors.
    """
    if product is None:
        return None
    
    return {
        "id": str(product["_id"]),  # Convert ObjectId to string
        "name": product.get("name", ""),
        "brand": product.get("brand", ""),
        "category": product.get("category", ""),
        "model": product.get("model"),  # Can be None
        "price": float(product.get("price", 0.0)),
        "stock": int(product.get("stock", 0))
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================
@app.get("/")
async def health():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        await client.admin.command('ping')
        return {
            "status": "healthy",
            "database": "connected",
            "message": "Inventory API is running"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# ============================================================================
# STEP 2: INSPECT /products ENDPOINT WITH FULL ERROR HANDLING
# ============================================================================
@app.get("/products")
async def get_products(
    search: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Get all products with optional filtering.
    
    FIXED ISSUES:
    - Wrapped in try/except to catch all errors
    - Converts MongoDB ObjectId to string
    - Returns empty array if no products (not an error)
    - Logs full traceback on error
    """
    try:
        logger.info(f"GET /products - search={search}, brand={brand}, category={category}")
        
        # Build query filter
        query = {}
        if search:
            query["$or"] = [
                {"name": {"$regex": search, "$options": "i"}},
                {"brand": {"$regex": search, "$options": "i"}},
                {"category": {"$regex": search, "$options": "i"}},
            ]
        if brand:
            query["brand"] = brand
        if category:
            query["category"] = category
        
        logger.debug(f"MongoDB query: {query}")
        
        # Fetch from database
        cursor = products_collection.find(query)
        products = await cursor.to_list(length=1000)
        
        logger.info(f"Found {len(products)} products")
        
        # ============================================================================
        # STEP 7: DEFENSIVE RESPONSE - EMPTY DB RETURNS [], NOT ERROR
        # ============================================================================
        if not products:
            logger.info("No products found, returning empty array")
            return []
        
        # ============================================================================
        # STEP 5: REMOVE INVALID TYPES - CONVERT TO JSON-SAFE FORMAT
        # ============================================================================
        result = [product_helper(product) for product in products]
        
        logger.debug(f"Returning {len(result)} products")
        return result
        
    except Exception as e:
        # ============================================================================
        # STEP 2: LOG THE EXACT EXCEPTION MESSAGE AND TRACEBACK
        # ============================================================================
        logger.error("=" * 80)
        logger.error("CRITICAL ERROR in GET /products")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {str(e)}")
        logger.error("Full Traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        
        # Return clear error to client
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch products: {type(e).__name__}: {str(e)}"
        )


@app.post("/products")
async def add_product(product: ProductCreate):
    """Add a new product"""
    try:
        logger.info(f"POST /products - Adding product: {product.name}")
        
        product_dict = product.model_dump(exclude_unset=True)
        
        result = await products_collection.insert_one(product_dict)
        
        # Fetch the created product
        new_product = await products_collection.find_one({"_id": result.inserted_id})
        
        if new_product is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve created product")
        
        logger.info(f"Product created with ID: {result.inserted_id}")
        
        # Convert ObjectId to string before returning
        return product_helper(new_product)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding product: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to add product: {str(e)}")


@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get a single product by ID"""
    try:
        logger.info(f"GET /products/{product_id}")
        
        if not ObjectId.is_valid(product_id):
            raise HTTPException(status_code=400, detail="Invalid product ID format")
        
        product = await products_collection.find_one({"_id": ObjectId(product_id)})
        
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return product_helper(product)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to fetch product: {str(e)}")


@app.put("/products/{product_id}")
async def update_product(product_id: str, product: ProductCreate):
    """Update a product"""
    try:
        logger.info(f"PUT /products/{product_id}")
        
        if not ObjectId.is_valid(product_id):
            raise HTTPException(status_code=400, detail="Invalid product ID format")
        
        product_dict = product.model_dump(exclude_unset=True)
        
        result = await products_collection.update_one(
            {"_id": ObjectId(product_id)},
            {"$set": product_dict}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Fetch updated product
        updated_product = await products_collection.find_one({"_id": ObjectId(product_id)})
        
        return product_helper(updated_product)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating product {product_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to update product: {str(e)}")


@app.delete("/products/{product_id}")
async def delete_product(product_id: str):
    """Delete a product"""
    try:
        logger.info(f"DELETE /products/{product_id}")
        
        if not ObjectId.is_valid(product_id):
            raise HTTPException(status_code=400, detail="Invalid product ID format")
        
        result = await products_collection.delete_one({"_id": ObjectId(product_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return {"message": "Product deleted successfully", "id": product_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting product {product_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to delete product: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
