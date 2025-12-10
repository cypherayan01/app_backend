import re
from typing import List, Optional, Dict, Any, Literal, Tuple
from datetime import date
from pydantic import BaseModel, Field, validator


class JobSearchRequest(BaseModel):
    skills: List[str] = Field(..., min_items=1, max_items=50, description="List of skills to search for")
    limit: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of jobs to return")

    @validator('skills', pre=True)
    def validate_and_clean_skills(cls, v):
        if not v:
            raise ValueError('Skills list cannot be empty')

        # Handle different input types
        if isinstance(v, str):
            v = [s.strip() for s in v.split(',') if s.strip()]

        # Filter and clean skills
        cleaned_skills = []
        for skill in v:
            if isinstance(skill, str) and skill.strip():
                # Remove special characters but keep programming symbols
                cleaned = re.sub(r'[^\w\s+#.-]', '', skill.strip())
                if cleaned and len(cleaned) > 1:
                    cleaned_skills.append(cleaned)

        if not cleaned_skills:
            raise ValueError('No valid skills found after cleaning')

        if len(cleaned_skills) > 50:
            cleaned_skills = cleaned_skills[:50]

        return cleaned_skills


class JobResult(BaseModel):
    ncspjobid: str
    title: str
    match_percentage: float = Field(..., ge=0, le=100)
    similarity_score: Optional[float] = None
    keywords: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    organizationid: Optional[int] = None
    organization_name: Optional[str] = None
    numberofopenings: Optional[int] = None
    industryname: Optional[str] = None
    sectorname: Optional[str] = None
    functionalareaname: Optional[str] = None
    functionalrolename: Optional[str] = None
    aveexp: Optional[float] = None
    avewage: Optional[float] = None
    gendercode: Optional[str] = None
    highestqualification: Optional[str] = None
    statename: Optional[str] = None
    districtname: Optional[str] = None
    keywords_matched: Optional[List[str]] = None
    keywords_unmatched: Optional[List[str]] = None
    user_skills_matched: Optional[List[str]] = None
    keyword_match_score: Optional[float] = None


class JobSearchResponse(BaseModel):
    jobs: List[JobResult]
    query_skills: List[str]
    total_found: int
    processing_time_ms: int


class JobData(BaseModel):
    keywords: str
    ncsp_job_id: str
    date: date
    organization_id: int
    organization_name: str
    number_of_openings: int
    industry_name: str
    sector_name: str
    functional_area_name: str
    functional_role_name: str
    avg_experience: int
    avg_wage: int
    gender_code: str
    highest_qualification: str
    state_name: str
    district_name: str
    title: str
    description: str
    pin_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LocationJobRequest(BaseModel):
    location: str = Field(..., min_length=2, max_length=100, description="City, district, or state name")
    job_type: Optional[str] = Field(default=None, description="Optional job type filter")
    skills: Optional[List[str]] = Field(default=None, description="Optional skills filter")
    experience_range: Optional[Tuple[float, float]] = Field(default=None, description="Min, Max experience in years")
    salary_range: Optional[Tuple[float, float]] = Field(default=None, description="Min, Max salary range")
    limit: Optional[int] = Field(default=50, ge=1, le=200, description="Maximum jobs to return")
    sort_by: Optional[Literal["relevance", "salary", "experience", "date"]] = "relevance"


class LocationJobResponse(BaseModel):
    location_searched: str
    location_matches: Dict[str, List[str]]
    jobs: List[Dict[str, Any]]
    total_found: int
    returned_count: int
    processing_time_ms: int
    filters_applied: Dict[str, Any]
    search_context: Dict[str, Any]


class ConsumeAPIResponse(BaseModel):
    message: str
    jobs_processed: int
    success: bool
