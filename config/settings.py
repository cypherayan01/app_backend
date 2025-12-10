import os
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# JSON file path for legacy resume matching
JSON_FILE_PATH = r"naukriApiResponse.json"

# Global thread pool for CPU-intensive embedding operations
embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")

# Azure OpenAI client for text generation only
try:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint and not endpoint.endswith('/'):
        endpoint = endpoint + '/'

    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        azure_endpoint=endpoint
    )

    gpt_deployment = os.getenv("AZURE_GPT_DEPLOYMENT")
    if not gpt_deployment:
        raise ValueError("Missing AZURE_GPT_DEPLOYMENT in environment variables")

    logger.info(f"Initialized Azure client for GPT: {gpt_deployment}")

except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    raise

# PostgreSQL connection string (legacy)
DB_URL = os.getenv("DATABASE_URL")
