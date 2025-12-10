import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'jobsearch'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}


class DatabasePool:
    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def initialize(cls, min_size: int = 5, max_size: int = 20):
        """Initialize pool with DB_CONFIG parameters"""
        try:
            logger.info(f"Connecting to {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

            # Test single connection first
            test_conn = await asyncpg.connect(**DB_CONFIG)
            version = await test_conn.fetchval('SELECT version()')
            logger.info(f"✓ Test connection successful: {version[:50]}")
            await test_conn.close()

            # Create pool with DB_CONFIG parameters
            cls._pool = await asyncpg.create_pool(
                **DB_CONFIG,
                min_size=min_size,
                max_size=max_size,
                max_inactive_connection_lifetime=300,
                server_settings={
                    'jit': 'off',
                    'search_path': 'public',
                    'application_name': 'ncs-job-search'
                }
            )

            # Verify pool
            async with cls._pool.acquire() as conn:
                count = await conn.fetchval('SELECT COUNT(*) FROM vacancies_summary')
                logger.info(f"✓ Pool initialized - Found {count:,} jobs in database")

        except Exception as e:
            logger.error(f"❌ Pool initialization failed: {type(e).__name__}: {e}")
            raise

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("✓ Database pool closed")

    @classmethod
    @asynccontextmanager
    async def acquire(cls):
        if not cls._pool:
            raise RuntimeError("Database pool not initialized")
        async with cls._pool.acquire() as conn:
            yield conn
