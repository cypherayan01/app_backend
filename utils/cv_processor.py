"""
CV Processor - Extract profile data from CVs/Resumes

This module provides functionality to extract structured profile data from
uploaded CV files (PDF, DOCX, images).
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CVProfile:
    """Data class representing extracted CV profile"""
    name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    experience: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    confidence_score: float = 0.0


class CVProcessor:
    """
    Enhanced CV processor for extracting profile data from uploaded CVs.

    This class handles multiple file formats including PDF, DOCX, and images
    using text extraction and OCR when necessary.
    """

    def __init__(self, model_path: str = "all-MiniLM-L6-v2", tesseract_path: str = None):
        """
        Initialize the CV processor.

        Args:
            model_path: Path to the sentence transformer model
            tesseract_path: Path to tesseract executable (for OCR)
        """
        self.model_path = model_path
        self.tesseract_path = tesseract_path

        # Common skill keywords for extraction
        self.skill_keywords = [
            # Programming Languages
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
            'kotlin', 'go', 'rust', 'typescript', 'scala', 'perl', 'r',

            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'laravel', 'asp.net', 'jquery',

            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
            'sqlite', 'cassandra', 'elasticsearch', 'dynamodb',

            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'terraform', 'ansible', 'ci/cd', 'git', 'linux',

            # Data & ML
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'data analysis', 'tableau',
            'power bi', 'spark', 'hadoop',

            # Business Skills
            'data entry', 'customer service', 'sales', 'marketing',
            'communication', 'leadership', 'project management', 'agile',
            'scrum', 'excel', 'word', 'powerpoint'
        ]

        logger.info("CVProcessor initialized")

    async def process_cv(self, file_content: bytes, filename: str) -> CVProfile:
        """
        Process CV file and extract profile data.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename for format detection

        Returns:
            CVProfile with extracted data
        """
        try:
            file_ext = filename.lower().split('.')[-1]

            # Extract text based on file type
            if file_ext == 'pdf':
                text = await self._extract_text_from_pdf(file_content)
            elif file_ext in ['doc', 'docx']:
                text = await self._extract_text_from_docx(file_content)
            elif file_ext in ['png', 'jpg', 'jpeg']:
                text = await self._extract_text_from_image(file_content)
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return CVProfile(confidence_score=0.0)

            # Extract profile data from text
            profile = self._extract_profile_from_text(text)

            logger.info(f"CV processed: {len(profile.skills)} skills, confidence: {profile.confidence_score}")
            return profile

        except Exception as e:
            logger.error(f"CV processing failed: {e}")
            return CVProfile(confidence_score=0.0)

    async def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            import pdfplumber
            import io

            text_parts = []
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""

    async def _extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            from io import BytesIO

            doc = Document(BytesIO(file_content))
            text_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text.strip())

            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""

    async def _extract_text_from_image(self, file_content: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            import pytesseract
            from PIL import Image
            import io

            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

            img = Image.open(io.BytesIO(file_content))
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def _extract_profile_from_text(self, text: str) -> CVProfile:
        """Extract structured profile data from CV text"""
        profile = CVProfile()

        if not text:
            return profile

        text_lower = text.lower()

        # Extract email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        if emails:
            profile.email = emails[0]

        # Extract phone
        phone_pattern = r'(?:\+91[-.\s]?)?(?:\d{10}|\d{5}[-.\s]?\d{5}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            profile.phone = phones[0]

        # Extract name (usually first line or after "Name:")
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 2 and len(line) < 50:
                # Skip lines that look like headers or emails
                if not any(x in line.lower() for x in ['curriculum', 'resume', 'cv', '@', 'phone', 'email']):
                    profile.name = line
                    break

        # Extract skills
        profile.skills = self._extract_skills(text_lower)

        # Calculate confidence score
        profile.confidence_score = self._calculate_confidence(profile)

        return profile

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from CV text"""
        found_skills = []

        for skill in self.skill_keywords:
            # Use word boundary matching for better accuracy
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                # Normalize skill name
                normalized = skill.title() if len(skill) > 2 else skill.upper()
                if normalized not in found_skills:
                    found_skills.append(normalized)

        return found_skills

    def _calculate_confidence(self, profile: CVProfile) -> float:
        """Calculate confidence score based on extracted data completeness"""
        score = 0.0

        if profile.name:
            score += 0.2
        if profile.email:
            score += 0.2
        if profile.phone:
            score += 0.1
        if len(profile.skills) >= 3:
            score += 0.3
        elif len(profile.skills) >= 1:
            score += 0.15
        if profile.experience:
            score += 0.1
        if profile.education:
            score += 0.1

        return min(1.0, score)

    def get_job_search_text(self, profile: CVProfile) -> str:
        """Generate text suitable for job search from profile"""
        parts = []

        if profile.skills:
            parts.append(" ".join(profile.skills[:15]))

        if profile.summary:
            parts.append(profile.summary[:200])

        for exp in profile.experience[:3]:
            if exp.get('title'):
                parts.append(exp['title'])
            if exp.get('description'):
                parts.append(exp['description'][:100])

        return " ".join(parts)

    def to_dict(self, profile: CVProfile) -> Dict[str, Any]:
        """Convert CVProfile to dictionary"""
        return {
            "name": profile.name,
            "email": profile.email,
            "phone": profile.phone,
            "skills": profile.skills,
            "experience": profile.experience,
            "education": profile.education,
            "summary": profile.summary,
            "confidence_score": profile.confidence_score
        }
