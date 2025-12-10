import asyncio
import logging

from fastapi import APIRouter, HTTPException

from models.course import (
    CourseRecommendationRequest, CourseRecommendationResponse, CourseRecommendation
)
from services.course_service import CourseRecommendationService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (will be injected from main app)
course_service: CourseRecommendationService = None


def init_services(course_svc: CourseRecommendationService):
    """Initialize services from main app"""
    global course_service
    course_service = course_svc


@router.post("/recommend_courses", response_model=CourseRecommendationResponse)
async def recommend_courses(request: CourseRecommendationRequest) -> CourseRecommendationResponse:
    """
    Get course recommendations for unmatched keywords
    """
    start_time = asyncio.get_event_loop().time()

    try:
        logger.info(f"Course recommendation request for {len(request.keywords_unmatched)} keywords: {request.keywords_unmatched}")

        # Get course recommendations from GPT
        recommendations_data = await course_service.get_course_recommendations(request.keywords_unmatched)

        # Convert to response format
        course_recommendations = []
        for course_data in recommendations_data:
            try:
                course_rec = CourseRecommendation(
                    course_name=course_data.get("course_name", "Unknown Course"),
                    platform=course_data.get("platform", "Unknown Platform"),
                    duration=course_data.get("duration", "Unknown Duration"),
                    link=course_data.get("link", ""),
                    educator=course_data.get("educator", "Unknown Educator"),
                    skill_covered=course_data.get("skill_covered", ""),
                    difficulty_level=course_data.get("difficulty_level"),
                    rating=course_data.get("rating")
                )
                course_recommendations.append(course_rec)
            except Exception as e:
                logger.error(f"Error processing course recommendation: {e}")
                continue

        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

        logger.info(f"Course recommendations completed: {len(course_recommendations)} courses returned in {processing_time_ms}ms")

        return CourseRecommendationResponse(
            recommendations=course_recommendations,
            keywords_processed=request.keywords_unmatched,
            total_recommendations=len(course_recommendations),
            processing_time_ms=processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Course recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Course recommendation failed")
