import os
import asyncio
import pickle
import threading
import logging
from typing import List, Dict

import numpy as np
import faiss
import asyncpg
from fastapi import HTTPException

from config.database import DB_CONFIG

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for job similarity search"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.job_metadata = []
        self.is_loaded = False
        self._lock = threading.Lock()
        self.index_file = "faiss_job_index.bin"
        self.metadata_file = "job_metadata.pkl"

    async def load_jobs_from_db(self, force_reload: bool = False, embedding_service=None):
        """Load jobs from PostgreSQL and build/load FAISS index"""
        if self.is_loaded and not force_reload:
            logger.info("FAISS index already loaded")
            return

        with self._lock:
            try:
                # Try to load existing index first
                if not force_reload and os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                    logger.info("Loading existing FAISS index from disk...")
                    self.index = faiss.read_index(self.index_file)

                    with open(self.metadata_file, 'rb') as f:
                        self.job_metadata = pickle.load(f)

                    logger.info(f"Loaded FAISS index with {self.index.ntotal} jobs")
                    self.is_loaded = True
                    return

                logger.info("Building new FAISS index from database...")

                # Connect to database and fetch jobs
                conn = await asyncpg.connect(**DB_CONFIG)
                try:
                    rows = await conn.fetch("""
                        SELECT ncspjobid, title, keywords, description
                        FROM vacancies_summary
                        WHERE (keywords IS NOT NULL AND keywords != '')
                           OR (description IS NOT NULL AND description != '')
                        ORDER BY ncspjobid;
                    """)

                    if not rows:
                        logger.warning("No jobs found in database")
                        return

                    logger.info(f"Found {len(rows)} jobs in database")

                    # Prepare job texts and metadata
                    job_texts = []
                    self.job_metadata = []

                    for row in rows:
                        # Combine title, keywords, and description for embedding
                        text_parts = []
                        if row['title']:
                            text_parts.append(row['title'])
                        if row['keywords']:
                            text_parts.append(row['keywords'])
                        if row['description']:
                            desc = row['description'][:500] if row['description'] else ""
                            if desc:
                                text_parts.append(desc)

                        job_text = " ".join(text_parts)
                        job_texts.append(job_text)

                        self.job_metadata.append({
                            'ncspjobid': row['ncspjobid'],
                            'title': row['title'],
                            'keywords': row['keywords'],
                            'description': row['description']
                        })

                    # Generate embeddings for all jobs
                    logger.info("Generating embeddings for all jobs...")
                    embeddings = await self._generate_job_embeddings(job_texts, embedding_service)

                    # Create FAISS index
                    self.index = faiss.IndexFlatIP(self.dimension)

                    # Add embeddings to index
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    faiss.normalize_L2(embeddings_array)
                    self.index.add(embeddings_array)

                    # Save index and metadata to disk
                    faiss.write_index(self.index, self.index_file)
                    with open(self.metadata_file, 'wb') as f:
                        pickle.dump(self.job_metadata, f)

                    logger.info(f"Built FAISS index with {self.index.ntotal} jobs and saved to disk")
                    self.is_loaded = True

                finally:
                    await conn.close()

            except Exception as e:
                logger.error(f"Failed to load jobs into FAISS: {e}")
                raise HTTPException(status_code=503, detail="Failed to initialize job search index")

    async def _generate_job_embeddings(self, job_texts: List[str], embedding_service) -> List[List[float]]:
        """Generate embeddings for job texts using the embedding service"""
        embeddings = []
        batch_size = 10

        for i in range(0, len(job_texts), batch_size):
            batch = job_texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(job_texts) + batch_size - 1)//batch_size}")

            batch_embeddings = []
            for text in batch:
                try:
                    embedding = await embedding_service.get_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for job text: {e}")
                    batch_embeddings.append([0.0] * self.dimension)

            embeddings.extend(batch_embeddings)
            await asyncio.sleep(0.1)

        return embeddings

    async def search_similar_jobs(self, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
        """Search for similar jobs using FAISS"""
        if not self.is_loaded or self.index is None:
            raise HTTPException(status_code=503, detail="Job search index not available")

        try:
            # Normalize query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # Search
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0:
                    job_data = self.job_metadata[idx].copy()
                    job_data['similarity'] = float(similarity)
                    results.append(job_data)

            logger.info(f"FAISS search returned {len(results)} similar jobs")
            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise HTTPException(status_code=500, detail="Job search failed")
