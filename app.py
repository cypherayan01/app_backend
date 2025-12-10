"""
Job Search API - Main Application Entry Point

AI-powered job search using skills matching with FastAPI
"""

import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import configuration
from config.settings import logger, embedding_executor
from config.database import DatabasePool

# Import services
from services.embedding_service import LocalEmbeddingService
from services.vector_store import FAISSVectorStore
from services.gpt_service import GPTService
from services.course_service import CourseRecommendationService
from services.chat_service import EnhancedChatService, CVChatService

# Import controllers
from controllers import job_router, chat_router, cv_router, course_router
from controllers import job_controller, chat_controller, cv_controller, course_controller

# Import utils
from utils.cv_processor import CVProcessor


# Initialize services
embedding_service = LocalEmbeddingService()
vector_store = FAISSVectorStore()
gpt_service = GPTService()
course_service = CourseRecommendationService()
enhanced_chat_service = EnhancedChatService()
cv_chat_service = CVChatService()

cv_processor = CVProcessor(
    model_path="all-MiniLM-L6-v2",
    tesseract_path=r"C:\Users\WK929BY\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"  # Update path as needed
)


# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Job Search API Starting Up ===")

    # Validate env vars
    required_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_GPT_DEPLOYMENT",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing env vars: {missing_vars}")
        raise ValueError(f"Missing: {missing_vars}")

    logger.info("✓ Azure OpenAI environment variables validated")

    # Initialize DB pool
    try:
        await DatabasePool.initialize(
            min_size=5,
            max_size=20
        )
        logger.info("✓ Database pool initialized successfully")
    except Exception as e:
        logger.error(f"❌ Pool initialization failed: {e}")
        raise

    # Load FAISS index
    try:
        logger.info("Loading FAISS job search index...")
        await vector_store.load_jobs_from_db(embedding_service=embedding_service)
        logger.info("✓ Job search index loaded successfully")
    except Exception as e:
        logger.error(f"❌ Index load failed: {e}")
        raise

    # Initialize controller services
    job_controller.init_services(embedding_service, vector_store, gpt_service)
    chat_controller.init_services(enhanced_chat_service)
    cv_controller.init_services(embedding_service, vector_store, gpt_service, cv_chat_service, cv_processor)
    course_controller.init_services(course_service)

    logger.info("=== API Started Successfully ===")
    yield

    # Shutdown
    logger.info("=== API Shutting Down ===")
    try:
        await DatabasePool.close()
        embedding_executor.shutdown(wait=True)
        logger.info("✓ Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Job Search API",
    description="AI-powered job search using skills matching",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )


# Include routers
app.include_router(job_router)
app.include_router(chat_router)
app.include_router(cv_router)
app.include_router(course_router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8888))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
