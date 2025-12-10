from .chat import ChatRequest, ChatResponse, ChatWithCVRequest
from .job import (
    JobSearchRequest, JobSearchResponse, JobResult, JobData,
    LocationJobRequest, LocationJobResponse, ConsumeAPIResponse
)
from .cv import CVUploadRequest, CVUploadResponse, CVAnalysisResponse
from .course import CourseRecommendationRequest, CourseRecommendationResponse, CourseRecommendation

__all__ = [
    # Chat models
    'ChatRequest',
    'ChatResponse',
    'ChatWithCVRequest',
    # Job models
    'JobSearchRequest',
    'JobSearchResponse',
    'JobResult',
    'JobData',
    'LocationJobRequest',
    'LocationJobResponse',
    'ConsumeAPIResponse',
    # CV models
    'CVUploadRequest',
    'CVUploadResponse',
    'CVAnalysisResponse',
    # Course models
    'CourseRecommendationRequest',
    'CourseRecommendationResponse',
    'CourseRecommendation',
]
