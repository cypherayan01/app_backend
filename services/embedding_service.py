import re
import asyncio
import logging
from typing import List, Union, Any

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

from config.settings import embedding_executor

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Local embedding service using Sentence Transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.model = SentenceTransformer(self.model_name, device=device)

            if hasattr(self.model, 'eval'):
                self.model.eval()

            test_embedding = self.model.encode("test input", convert_to_tensor=False)
            logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")

        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation for thread pool execution"""
        try:
            if not self.model:
                raise ValueError("Model not initialized")

            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            return embedding

        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}")
            raise

    async def get_embedding(self, text: Union[str, List[str], Any]) -> List[float]:
        """Generate embedding using local Sentence Transformer model"""

        # Normalize input to string
        try:
            if isinstance(text, list):
                processed_text = " ".join(str(item) for item in text if item)
            elif isinstance(text, (int, float)):
                processed_text = str(text)
            elif text is None:
                raise ValueError("Embedding input cannot be None")
            else:
                processed_text = str(text)

            processed_text = re.sub(r'\s+', ' ', processed_text.strip())

            if not processed_text or len(processed_text) == 0:
                raise ValueError("Embedding input must be non-empty after processing")

            if len(processed_text) > 2000:
                processed_text = processed_text[:2000]

        except Exception as e:
            logger.error(f"Input processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                embedding_executor,
                self._generate_embedding_sync,
                processed_text
            )

            if not embedding or len(embedding) == 0:
                raise ValueError("Empty embedding generated")

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
