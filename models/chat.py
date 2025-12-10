from typing import List, Optional, Dict
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    chat_phase: str = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []


class ChatResponse(BaseModel):
    response: str
    message_type: str = "text"
    chat_phase: Optional[str] = None
    profile_data: Optional[Dict] = None
    jobs: Optional[List[Dict]] = None
    suggestions: Optional[List[str]] = []
    location_searched: Optional[str] = None
    location_matches: Optional[Dict] = None
    total_found: Optional[int] = None
    filters_applied: Optional[Dict] = None
    search_context: Optional[Dict] = None


class ChatWithCVRequest(BaseModel):
    message: str
    chat_phase: str = "intro"
    user_profile: Optional[Dict] = None
    conversation_history: Optional[List[Dict]] = []
    cv_profile_data: Optional[Dict] = None
