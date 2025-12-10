import logging
from dataclasses import dataclass
from typing import List, Optional

from fastapi import APIRouter

from models.chat import ChatRequest, ChatResponse, ChatWithCVRequest
from services.chat_service import EnhancedChatService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (will be injected from main app)
enhanced_chat_service: EnhancedChatService = None


def init_services(chat_svc: EnhancedChatService):
    """Initialize services from main app"""
    global enhanced_chat_service
    enhanced_chat_service = chat_svc


@router.post("/chat", response_model=ChatResponse)
async def enhanced_chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Enhanced chat endpoint with location awareness"""
    try:
        response = await enhanced_chat_service.handle_chat_message(request)
        return response
    except Exception as e:
        logger.error(f"Enhanced chat endpoint failed: {e}")
        return ChatResponse(
            response="I'm here to help you find jobs! You can tell me your skills or ask about jobs in specific locations like 'Mumbai jobs'.",
            message_type="text",
            chat_phase="profile_building",
            suggestions=["My skills are...", "Jobs in Mumbai", "Remote work", "Entry level"]
        )


@router.post("/chat_with_cv", response_model=ChatResponse)
async def chat_with_cv_context(request: ChatWithCVRequest) -> ChatResponse:
    """Handle chat with CV context for follow-up questions"""
    try:
        if request.cv_profile_data:
            # Create a simple CVProfile-like object
            @dataclass
            class CVProfileLike:
                name: str
                skills: List[str]
                experience: List
                education: List
                email: Optional[str]
                phone: Optional[str]
                summary: Optional[str]
                confidence_score: float

            cv_profile = CVProfileLike(
                name=request.cv_profile_data.get('name', ''),
                skills=request.cv_profile_data.get('skills', []),
                experience=request.cv_profile_data.get('experience', []),
                education=request.cv_profile_data.get('education', []),
                email=request.cv_profile_data.get('email'),
                phone=request.cv_profile_data.get('phone'),
                summary=request.cv_profile_data.get('summary'),
                confidence_score=request.cv_profile_data.get('confidence_score', 0.5)
            )

            # Create a ChatRequest for the CV service
            chat_request = ChatRequest(
                message=request.message,
                chat_phase=request.chat_phase,
                user_profile=request.user_profile,
                conversation_history=request.conversation_history
            )

            return await enhanced_chat_service.handle_cv_followup_chat(chat_request, cv_profile)
        else:
            # Fall back to regular chat
            chat_request = ChatRequest(
                message=request.message,
                chat_phase=request.chat_phase,
                user_profile=request.user_profile,
                conversation_history=request.conversation_history
            )
            response = await enhanced_chat_service.handle_chat_message(chat_request)

        return response
    except Exception as e:
        logger.error(f"Chat with CV context failed: {e}")
        return ChatResponse(
            response="I can help you find jobs. What would you like to do?",
            message_type="text",
            chat_phase="job_searching"
        )
