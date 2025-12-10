import re
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class CourseRecommendationRequest(BaseModel):
    keywords_unmatched: List[str] = Field(..., min_items=1, max_items=20, description="List of unmatched keywords to get course recommendations for")

    @validator('keywords_unmatched', pre=True)
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')

        # Handle different input types
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]

        # Clean keywords
        cleaned_keywords = []
        for keyword in v:
            if isinstance(keyword, str) and keyword.strip():
                cleaned = keyword.strip()
                if len(cleaned) > 1:
                    cleaned_keywords.append(cleaned)

        if not cleaned_keywords:
            raise ValueError('No valid keywords found after cleaning')

        if len(cleaned_keywords) > 20:
            cleaned_keywords = cleaned_keywords[:20]

        return cleaned_keywords


class CourseRecommendation(BaseModel):
    course_name: str
    platform: str
    duration: str
    link: str
    educator: str
    skill_covered: str
    difficulty_level: Optional[str] = None
    rating: Optional[str] = None


class CourseRecommendationResponse(BaseModel):
    recommendations: List[CourseRecommendation]
    keywords_processed: List[str]
    total_recommendations: int
    processing_time_ms: int
