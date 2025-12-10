from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class CVUploadRequest(BaseModel):
    """CV upload request model"""
    pass


class CVUploadResponse(BaseModel):
    response: str
    profile_data: Optional[Dict] = None
    success: bool = True


class CVAnalysisResponse(BaseModel):
    """Enhanced CV analysis response"""
    success: bool
    message: str
    profile: Optional[Dict[str, Any]] = None
    jobs: Optional[List[Dict]] = None
    total_jobs_found: int = 0
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    recommendations: Optional[List[str]] = None
