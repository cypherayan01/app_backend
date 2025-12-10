import asyncio
import logging

from fastapi import APIRouter, HTTPException

from config.database import DatabasePool
from models.job import (
    JobSearchRequest, JobSearchResponse, JobResult,
    JobData, ConsumeAPIResponse, ConsumeAPIRequest,
    NCSLoginRequest, NCSVacancyDataRequest
)
from services.embedding_service import LocalEmbeddingService
from services.vector_store import FAISSVectorStore
from services.gpt_service import GPTService
from db.job_repository import get_complete_job_details

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (will be injected from main app)
embedding_service: LocalEmbeddingService = None
vector_store: FAISSVectorStore = None
gpt_service: GPTService = None


def init_services(embed_svc: LocalEmbeddingService, vec_store: FAISSVectorStore, gpt_svc: GPTService):
    """Initialize services from main app"""
    global embedding_service, vector_store, gpt_service
    embedding_service = embed_svc
    vector_store = vec_store
    gpt_service = gpt_svc


@router.post("/search_jobs", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest) -> JobSearchResponse:
    """
    Search for relevant job postings based on skills
    """
    start_time = asyncio.get_event_loop().time()

    try:
        logger.info(f"Job search request: {len(request.skills)} skills, limit: {request.limit}")

        # Combine skills into text for embedding
        skills_text = " ".join(request.skills)

        # Generate embedding for skills
        skills_embedding = await embedding_service.get_embedding(skills_text)
        logger.info(f"Generated embedding for skills: {skills_text[:100]}...")

        # Search similar jobs using FAISS
        similar_jobs = await vector_store.search_similar_jobs(skills_embedding, top_k=50)

        if not similar_jobs:
            return JobSearchResponse(
                jobs=[],
                query_skills=request.skills,
                total_found=0,
                processing_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000)
            )

        # Re-rank with Azure GPT
        ranked_jobs = await gpt_service.rerank_jobs(request.skills, similar_jobs)

        ranked_jobs.sort(key=lambda job: job.get("match_percentage", 0), reverse=True)

        job_ids = [job["ncspjobid"] for job in ranked_jobs[:request.limit]]
        complete_jobs = await get_complete_job_details(job_ids)
        # Convert to response format
        job_results = []
        for job_data in ranked_jobs[:request.limit]:
            # Find complete job data from database
            complete_job = next((j for j in complete_jobs if j["ncspjobid"] == job_data["ncspjobid"]), {})

            job_result = JobResult(
                ncspjobid=job_data["ncspjobid"],
                title=job_data["title"],
                match_percentage=job_data["match_percentage"],
                similarity_score=next((j.get("similarity") for j in similar_jobs if j["ncspjobid"] == job_data["ncspjobid"]), None),
                keywords=complete_job.get("keywords"),
                description=complete_job.get("description"),
                date=complete_job.get("date"),
                organizationid=complete_job.get("organizationid"),
                organization_name=complete_job.get("organization_name"),
                numberofopenings=complete_job.get("numberofopenings"),
                industryname=complete_job.get("industryname"),
                sectorname=complete_job.get("sectorname"),
                functionalareaname=complete_job.get("functionalareaname"),
                functionalrolename=complete_job.get("functionalrolename"),
                aveexp=complete_job.get("aveexp"),
                avewage=complete_job.get("avewage"),
                gendercode=complete_job.get("gendercode"),
                highestqualification=complete_job.get("highestqualification"),
                statename=complete_job.get("statename"),
                districtname=complete_job.get("districtname"),
                keywords_matched=job_data.get("keywords_matched"),
                keywords_unmatched=job_data.get("keywords_unmatched"),
                user_skills_matched=job_data.get("user_skills_matched"),
                keyword_match_score=job_data.get("keyword_matches")
            )
            job_results.append(job_result)

        processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

        logger.info(f"Job search completed: {len(job_results)} jobs returned in {processing_time_ms}ms")

        return JobSearchResponse(
            jobs=job_results,
            query_skills=request.skills,
            total_found=len(ranked_jobs),
            processing_time_ms=processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job search failed: {e}")
        raise HTTPException(status_code=500, detail="Job search failed")


@router.post("/consume_api", response_model=ConsumeAPIResponse)
async def consume_api_endpoint(request: ConsumeAPIRequest):
    """
    Endpoint to consume NCS Vacancy API and store job data in database.

    Flow:
    1. Authenticate with NCS login endpoint to get token
    2. Use token to fetch vacancy data from NCS API
    3. Store the job data in database
    """
    import httpx

    # NCS API URLs (Dev Environment)
    LOGIN_URL = "https://stagingncsapi.ncs.gov.in/api/login"
    VACANCY_DATA_URL = "https://stagingncsapi.ncs.gov.in/api/NCS/v1/NCSVacancyDataAPI/Data"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Step 1: Authenticate and get token
            logger.info("Authenticating with NCS API...")
            login_payload = {
                "Username": request.username,
                "Password": request.password
            }

            login_response = await client.post(LOGIN_URL, json=login_payload)
            login_response.raise_for_status()

            login_data = login_response.json()
            token = login_data.get("token")

            if not token:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication failed: No token received from NCS API"
                )

            logger.info("Successfully authenticated with NCS API")

            # Step 2: Fetch vacancy data using the token
            logger.info("Fetching vacancy data from NCS API...")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            vacancy_payload = {
                "UserName": request.username,
                "OptionID": 1,
                "FromDate": request.from_date,
                "ToDate": request.to_date,
                "StateID": request.state_id,
                "DistrictID": request.district_id,
                "JobTitle": request.job_title,
                "Keywords": request.keywords,
                "Gender": request.gender,
                "HighestEducation": request.highest_education,
                "Age": request.age,
                "PageNumber": request.page_number,
                "PageSize": request.page_size
            }

            vacancy_response = await client.post(
                VACANCY_DATA_URL,
                json=vacancy_payload,
                headers=headers
            )
            vacancy_response.raise_for_status()

            # Parse JSON response
            job_data_list = vacancy_response.json()

            if not isinstance(job_data_list, list):
                raise HTTPException(
                    status_code=400,
                    detail="API response should be a list of job objects"
                )

            logger.info(f"Received {len(job_data_list)} jobs from NCS API")

        # Step 3: Store data in database
        jobs_processed = 0

        async with DatabasePool.acquire() as conn:
            for job_item in job_data_list:
                try:
                    # Validate job data structure
                    job_data = JobData(**job_item)

                    # Insert into database with new schema
                    insert_query = """
                    INSERT INTO ncs_job_data
                    (job_id, employer_name, job_title, minimum_experience, maximum_experience,
                     average_experience, job_start_date, job_expiry_date, maximum_wages, minimum_wages,
                     average_wage, number_of_openings, employment_type, industry_id, industry_name,
                     sector_id, sector_name, qualification, wage_type_desc, state_name, district_name,
                     functional_area_name, functional_role_name, job_description, vacancy_url,
                     posted_date, skills, employer_mobile, employer_email, person_name, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, NOW())
                    ON CONFLICT (job_id) DO UPDATE SET
                        employer_name = EXCLUDED.employer_name,
                        job_title = EXCLUDED.job_title,
                        minimum_experience = EXCLUDED.minimum_experience,
                        maximum_experience = EXCLUDED.maximum_experience,
                        average_experience = EXCLUDED.average_experience,
                        job_start_date = EXCLUDED.job_start_date,
                        job_expiry_date = EXCLUDED.job_expiry_date,
                        maximum_wages = EXCLUDED.maximum_wages,
                        minimum_wages = EXCLUDED.minimum_wages,
                        average_wage = EXCLUDED.average_wage,
                        number_of_openings = EXCLUDED.number_of_openings,
                        employment_type = EXCLUDED.employment_type,
                        industry_id = EXCLUDED.industry_id,
                        industry_name = EXCLUDED.industry_name,
                        sector_id = EXCLUDED.sector_id,
                        sector_name = EXCLUDED.sector_name,
                        qualification = EXCLUDED.qualification,
                        wage_type_desc = EXCLUDED.wage_type_desc,
                        state_name = EXCLUDED.state_name,
                        district_name = EXCLUDED.district_name,
                        functional_area_name = EXCLUDED.functional_area_name,
                        functional_role_name = EXCLUDED.functional_role_name,
                        job_description = EXCLUDED.job_description,
                        vacancy_url = EXCLUDED.vacancy_url,
                        posted_date = EXCLUDED.posted_date,
                        skills = EXCLUDED.skills,
                        employer_mobile = EXCLUDED.employer_mobile,
                        employer_email = EXCLUDED.employer_email,
                        person_name = EXCLUDED.person_name,
                        updated_at = NOW()
                    """

                    await conn.execute(
                        insert_query,
                        job_data.JobID,
                        job_data.EmployerName,
                        job_data.JobTitle,
                        job_data.MinimumExperience,
                        job_data.MaximunExperience,
                        job_data.AverageExp,
                        job_data.JobStartDate,
                        job_data.JobExpiryDate,
                        job_data.MaximunWages,
                        job_data.MinimumWages,
                        job_data.AverageWage,
                        job_data.NumberofOpenings,
                        job_data.EmploymentType,
                        job_data.IndustryID,
                        job_data.IndustryName,
                        job_data.SectorId,
                        job_data.SectorName,
                        job_data.Qualification,
                        job_data.WageTypeDesc,
                        job_data.StateName,
                        job_data.DistrictName,
                        job_data.FunctionalAreaName,
                        job_data.FunctionalRoleName,
                        job_data.JobDescription,
                        job_data.VacancyURL,
                        job_data.PostedDate,
                        job_data.Skills,
                        job_data.EmployerMobile,
                        job_data.EmployerEmail,
                        job_data.PersonName
                    )

                    jobs_processed += 1

                except Exception as e:
                    logger.error(f"Error processing job item {job_item.get('JobID', 'unknown')}: {e}")
                    continue

        logger.info(f"Successfully processed {jobs_processed} jobs from NCS API")

        return ConsumeAPIResponse(
            message=f"Successfully processed {jobs_processed} jobs from NCS API",
            jobs_processed=jobs_processed,
            success=True
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error when calling NCS API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"NCS API error: {e.response.text}"
        )
    except httpx.HTTPError as e:
        logger.error(f"HTTP error when calling NCS API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from NCS API: {str(e)}")
    except Exception as e:
        logger.error(f"Error in consume_api_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/consume_api_test", response_model=ConsumeAPIResponse)
async def consume_api_test_endpoint():
    """
    Test endpoint to consume sample NCS job data and store in database.
    Uses sample data matching the new NCS API response format for testing.
    """
    try:
        # Sample job data matching the new NCS API response format
        job_data_list = [
            {
                "JobID": 5427169,
                "EmployerName": "blinkit",
                "JobTitle": "Grocery Delivery Executive",
                "MinimumExperience": 0,
                "MaximunExperience": 372,
                "AverageExp": 186,
                "JobStartDate": "2025-03-24T23:53:43",
                "JobExpiryDate": "2025-04-07T23:59:59",
                "MaximunWages": 50000,
                "MinimumWages": 40000,
                "AverageWage": 45000,
                "NumberofOpenings": 1,
                "EmploymentType": "Full Time",
                "IndustryID": 14,
                "IndustryName": "Specialized Professional Services",
                "SectorId": 10,
                "SectorName": "Company",
                "Qualification": None,
                "WageTypeDesc": "Monthly",
                "StateName": "Haryana",
                "DistrictName": "Gurugram",
                "FunctionalAreaName": "Others",
                "FunctionalRoleName": "Others",
                "JobDescription": "Hey there! Looking for a flexible and rewarding opportunity? Join our team as a Grocery Delivery Executive at Blinkit. You will be responsible for delivering groceries to customers in a timely manner.",
                "VacancyURL": "https://www.ncs.gov.in/job-seeker/Pages/ViewJobDetails.aspx?JSID=IwTJw8ew3sE%3D&RowID=IwTJw8ew3sE%3D",
                "PostedDate": "2025-03-24T00:00:00",
                "Skills": "delivery",
                "EmployerMobile": "",
                "EmployerEmail": "",
                "PersonName": "roshita silwani"
            },
            {
                "JobID": 5427170,
                "EmployerName": "MUTHOOT FINANCE LTD",
                "JobTitle": "Branch Sales Executive",
                "MinimumExperience": 0,
                "MaximunExperience": 60,
                "AverageExp": 30,
                "JobStartDate": "2025-03-25T10:00:00",
                "JobExpiryDate": "2025-04-25T23:59:59",
                "MaximunWages": 25000,
                "MinimumWages": 15000,
                "AverageWage": 20000,
                "NumberofOpenings": 700,
                "EmploymentType": "Full Time",
                "IndustryID": 8,
                "IndustryName": "Finance and Insurance",
                "SectorId": 10,
                "SectorName": "Company",
                "Qualification": "Graduate",
                "WageTypeDesc": "Monthly",
                "StateName": "Telangana",
                "DistrictName": "Hyderabad",
                "FunctionalAreaName": "Marketing & Sales",
                "FunctionalRoleName": "Sales Executive",
                "JobDescription": "Key Responsibilities: Assist in daily branch Gold loan operations and customer service. Support branch team in handling customer queries and resolving issues. Participate in lead generation, client acquisition, and retention activities.",
                "VacancyURL": "https://www.ncs.gov.in/job-seeker/Pages/ViewJobDetails.aspx?JSID=test123",
                "PostedDate": "2025-03-25T00:00:00",
                "Skills": "acquisition,Marketing,Retention,customer service",
                "EmployerMobile": "",
                "EmployerEmail": "hr@muthootfinance.com",
                "PersonName": "HR Team"
            }
        ]

        # Store data in database
        jobs_processed = 0

        async with DatabasePool.acquire() as conn:
            for job_item in job_data_list:
                try:
                    # Validate job data structure
                    job_data = JobData(**job_item)

                    # Insert into database with new schema
                    insert_query = """
                    INSERT INTO ncs_job_data
                    (job_id, employer_name, job_title, minimum_experience, maximum_experience,
                     average_experience, job_start_date, job_expiry_date, maximum_wages, minimum_wages,
                     average_wage, number_of_openings, employment_type, industry_id, industry_name,
                     sector_id, sector_name, qualification, wage_type_desc, state_name, district_name,
                     functional_area_name, functional_role_name, job_description, vacancy_url,
                     posted_date, skills, employer_mobile, employer_email, person_name, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, NOW())
                    ON CONFLICT (job_id) DO UPDATE SET
                        employer_name = EXCLUDED.employer_name,
                        job_title = EXCLUDED.job_title,
                        minimum_experience = EXCLUDED.minimum_experience,
                        maximum_experience = EXCLUDED.maximum_experience,
                        average_experience = EXCLUDED.average_experience,
                        job_start_date = EXCLUDED.job_start_date,
                        job_expiry_date = EXCLUDED.job_expiry_date,
                        maximum_wages = EXCLUDED.maximum_wages,
                        minimum_wages = EXCLUDED.minimum_wages,
                        average_wage = EXCLUDED.average_wage,
                        number_of_openings = EXCLUDED.number_of_openings,
                        employment_type = EXCLUDED.employment_type,
                        industry_id = EXCLUDED.industry_id,
                        industry_name = EXCLUDED.industry_name,
                        sector_id = EXCLUDED.sector_id,
                        sector_name = EXCLUDED.sector_name,
                        qualification = EXCLUDED.qualification,
                        wage_type_desc = EXCLUDED.wage_type_desc,
                        state_name = EXCLUDED.state_name,
                        district_name = EXCLUDED.district_name,
                        functional_area_name = EXCLUDED.functional_area_name,
                        functional_role_name = EXCLUDED.functional_role_name,
                        job_description = EXCLUDED.job_description,
                        vacancy_url = EXCLUDED.vacancy_url,
                        posted_date = EXCLUDED.posted_date,
                        skills = EXCLUDED.skills,
                        employer_mobile = EXCLUDED.employer_mobile,
                        employer_email = EXCLUDED.employer_email,
                        person_name = EXCLUDED.person_name,
                        updated_at = NOW()
                    """

                    await conn.execute(
                        insert_query,
                        job_data.JobID,
                        job_data.EmployerName,
                        job_data.JobTitle,
                        job_data.MinimumExperience,
                        job_data.MaximunExperience,
                        job_data.AverageExp,
                        job_data.JobStartDate,
                        job_data.JobExpiryDate,
                        job_data.MaximunWages,
                        job_data.MinimumWages,
                        job_data.AverageWage,
                        job_data.NumberofOpenings,
                        job_data.EmploymentType,
                        job_data.IndustryID,
                        job_data.IndustryName,
                        job_data.SectorId,
                        job_data.SectorName,
                        job_data.Qualification,
                        job_data.WageTypeDesc,
                        job_data.StateName,
                        job_data.DistrictName,
                        job_data.FunctionalAreaName,
                        job_data.FunctionalRoleName,
                        job_data.JobDescription,
                        job_data.VacancyURL,
                        job_data.PostedDate,
                        job_data.Skills,
                        job_data.EmployerMobile,
                        job_data.EmployerEmail,
                        job_data.PersonName
                    )

                    jobs_processed += 1

                except Exception as e:
                    logger.error(f"Error processing job item {job_item.get('JobID', 'unknown')}: {e}")
                    continue

        return ConsumeAPIResponse(
            message=f"Successfully processed {jobs_processed} test jobs",
            jobs_processed=jobs_processed,
            success=True
        )

    except Exception as e:
        logger.error(f"Error in consume_api_test_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
