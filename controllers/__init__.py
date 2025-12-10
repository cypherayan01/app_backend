from .job_controller import router as job_router
from .chat_controller import router as chat_router
from .cv_controller import router as cv_router
from .course_controller import router as course_router

__all__ = [
    'job_router',
    'chat_router',
    'cv_router',
    'course_router',
]
