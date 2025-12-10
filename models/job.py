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


# NCS API Authentication Models
class NCSLoginRequest(BaseModel):
    Username: str
    Password: str


class NCSLoginResponse(BaseModel):
    token: str
    message: Optional[str] = None


class NCSVacancyDataRequest(BaseModel):
    UserName: str
    OptionID: int = 1
    FromDate: str  # Format: "YYYY-MM-DD"
    ToDate: str  # Format: "YYYY-MM-DD"
    StateID: Optional[str] = ""
    DistrictID: Optional[str] = ""
    JobTitle: Optional[str] = ""
    Keywords: Optional[str] = ""
    Gender: Optional[str] = ""
    HighestEducation: Optional[str] = ""
    Age: Optional[str] = ""
    PageNumber: str = "1"
    PageSize: str = "25"


# Request model for consume_api endpoint
class ConsumeAPIRequest(BaseModel):
    username: str = Field(..., description="NCS API username for login")
    password: str = Field(..., description="NCS API password for login")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format")
    state_id: Optional[str] = Field(default="", description="Optional state ID filter")
    district_id: Optional[str] = Field(default="", description="Optional district ID filter")
    job_title: Optional[str] = Field(default="", description="Optional job title filter")
    keywords: Optional[str] = Field(default="", description="Optional keywords filter")
    gender: Optional[str] = Field(default="", description="Optional gender filter")
    highest_education: Optional[str] = Field(default="", description="Optional education filter")
    age: Optional[str] = Field(default="", description="Optional age filter")
    page_number: str = Field(default="1", description="Page number for pagination")
    page_size: str = Field(default="25", description="Number of results per page")


# JobData model matching the new NCS API response format
class JobData(BaseModel):
    JobID: int
    EmployerName: str
    JobTitle: str
    MinimumExperience: Optional[int] = 0
    MaximunExperience: Optional[int] = 0  # Note: API has typo "Maximun"
    AverageExp: Optional[int] = 0
    JobStartDate: Optional[str] = None
    JobExpiryDate: Optional[str] = None
    MaximunWages: Optional[int] = 0  # Note: API has typo "Maximun"
    MinimumWages: Optional[int] = 0
    AverageWage: Optional[int] = 0
    NumberofOpenings: Optional[int] = 1
    EmploymentType: Optional[str] = None
    IndustryID: Optional[int] = None
    IndustryName: Optional[str] = None
    SectorId: Optional[int] = None
    SectorName: Optional[str] = None
    Qualification: Optional[str] = None
    WageTypeDesc: Optional[str] = None
    StateName: Optional[str] = None
    DistrictName: Optional[str] = None
    FunctionalAreaName: Optional[str] = None
    FunctionalRoleName: Optional[str] = None
    JobDescription: Optional[str] = None
    VacancyURL: Optional[str] = None
    PostedDate: Optional[str] = None
    Skills: Optional[str] = None
    EmployerMobile: Optional[str] = None
    EmployerEmail: Optional[str] = None
    PersonName: Optional[str] = None


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
