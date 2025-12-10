import asyncio
import logging

from fastapi import APIRouter, HTTPException

from config.database import DatabasePool
from models.job import (
    JobSearchRequest, JobSearchResponse, JobResult,
    JobData, ConsumeAPIResponse
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
async def consume_api_endpoint():
    """
    Endpoint to consume external API and store job data in database
    """
    import httpx

    try:
        # External API URL - replace with actual API endpoint
        api_url = "YOUR_EXTERNAL_API_URL_HERE"

        # Make HTTP request to external API
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url)
            response.raise_for_status()

        # Parse JSON response
        job_data_list = response.json()

        if not isinstance(job_data_list, list):
            raise HTTPException(status_code=400, detail="API response should be a list of job objects")

        # Store data in database
        jobs_processed = 0

        async with DatabasePool.acquire() as conn:
            for job_item in job_data_list:
                try:
                    # Validate job data structure
                    job_data = JobData(**job_item)

                    # Insert into database
                    insert_query = """
                    INSERT INTO job_data
                    (keywords, ncsp_job_id, date, organization_id, organization_name,
                     number_of_openings, industry_name, sector_name, functional_area_name,
                     functional_role_name, avg_experience, avg_wage, gender_code,
                     highest_qualification, state_name, district_name, title, description,
                     pin_code, latitude, longitude, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, NOW())
                    ON CONFLICT (ncsp_job_id) DO UPDATE SET
                        keywords = EXCLUDED.keywords,
                        organization_name = EXCLUDED.organization_name,
                        number_of_openings = EXCLUDED.number_of_openings,
                        industry_name = EXCLUDED.industry_name,
                        sector_name = EXCLUDED.sector_name,
                        functional_area_name = EXCLUDED.functional_area_name,
                        functional_role_name = EXCLUDED.functional_role_name,
                        avg_experience = EXCLUDED.avg_experience,
                        avg_wage = EXCLUDED.avg_wage,
                        gender_code = EXCLUDED.gender_code,
                        highest_qualification = EXCLUDED.highest_qualification,
                        state_name = EXCLUDED.state_name,
                        district_name = EXCLUDED.district_name,
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        pin_code = EXCLUDED.pin_code,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        updated_at = NOW()
                    """

                    await conn.execute(
                        insert_query,
                        job_data.keywords,
                        job_data.ncsp_job_id,
                        job_data.date,
                        job_data.organization_id,
                        job_data.organization_name,
                        job_data.number_of_openings,
                        job_data.industry_name,
                        job_data.sector_name,
                        job_data.functional_area_name,
                        job_data.functional_role_name,
                        job_data.avg_experience,
                        job_data.avg_wage,
                        job_data.gender_code,
                        job_data.highest_qualification,
                        job_data.state_name,
                        job_data.district_name,
                        job_data.title,
                        job_data.description,
                        job_data.pin_code,
                        job_data.latitude,
                        job_data.longitude
                    )

                    jobs_processed += 1

                except Exception as e:
                    logger.error(f"Error processing job item: {e}")
                    continue

        return ConsumeAPIResponse(
            message=f"Successfully processed {jobs_processed} jobs from external API",
            jobs_processed=jobs_processed,
            success=True
        )

    except httpx.HTTPError as e:
        logger.error(f"HTTP error when calling external API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from external API: {str(e)}")
    except Exception as e:
        logger.error(f"Error in consume_api_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/consume_api_test", response_model=ConsumeAPIResponse)
async def consume_api_test_endpoint():
    """
    Test endpoint to consume sample job data and store in database
    Uses sample data instead of external API for testing
    """
    try:
        # Sample job data for testing
        job_data_list = [
            {
                "keywords": "acquisition,local language,Marketing,personalized,Retention,Tenured,visibility",
                "ncsp_job_id": "20V63-0942298873242J",
                "date": "2025-05-08",
                "organization_id": 2244622,
                "organization_name": "MUTHOOT FINANCE LTD",
                "number_of_openings": 700,
                "industry_name": "Finance and Insurance",
                "sector_name": "Company",
                "functional_area_name": "Marketing & Sales",
                "functional_role_name": "Sales Executive",
                "avg_experience": 0,
                "avg_wage": 11000,
                "gender_code": "A",
                "highest_qualification": "Graduate",
                "state_name": "Telangana",
                "district_name": "Multiple Districts",
                "title": "BRANCH SALES",
                "description": "Key Responsibilities Assist in daily branch Gold loan operations and customer service. Support branch team in handling customer queries and resolving issues. Participate in lead generation, client acquisition, and retention activities. Help execute market",
                "pin_code": None,
                "latitude": None,
                "longitude": None
            },
            {
                "keywords": "Active learning;Adaptability;Analytical & Critical Skills;Attention;Communication;Communication skills",
                "ncsp_job_id": "20V63-1550061073345J",
                "date": "2025-05-08",
                "organization_id": 2336905,
                "organization_name": "KARPAGA Assessment APP MATRIX Services Private Limited",
                "number_of_openings": 10,
                "industry_name": "Education",
                "sector_name": "Private",
                "functional_area_name": "Education",
                "functional_role_name": "Fresher",
                "avg_experience": 12,
                "avg_wage": 0,
                "gender_code": "A",
                "highest_qualification": "Graduate",
                "state_name": "Bihar",
                "district_name": "ALL",
                "title": "Public speaking teacher",
                "description": "JOB DESCRIPTION: It's a Public Speaking/English Teacher Job where you need to conduct demos on a regular basis and convert them to enrolments/sales. ABOUT THE COMPANY: Fantastiqo is an Edu-Tech company which provides online courses on different fields.",
                "pin_code": None,
                "latitude": None,
                "longitude": None
            }
        ]

        # Store data in database
        jobs_processed = 0

        async with DatabasePool.acquire() as conn:
            for job_item in job_data_list:
                try:
                    # Validate job data structure
                    job_data = JobData(**job_item)

                    # Insert into database
                    insert_query = """
                    INSERT INTO job_data
                    (keywords, ncsp_job_id, date, organization_id, organization_name,
                     number_of_openings, industry_name, sector_name, functional_area_name,
                     functional_role_name, avg_experience, avg_wage, gender_code,
                     highest_qualification, state_name, district_name, title, description,
                     pin_code, latitude, longitude, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, NOW())
                    ON CONFLICT (ncsp_job_id) DO UPDATE SET
                        keywords = EXCLUDED.keywords,
                        organization_name = EXCLUDED.organization_name,
                        number_of_openings = EXCLUDED.number_of_openings,
                        industry_name = EXCLUDED.industry_name,
                        sector_name = EXCLUDED.sector_name,
                        functional_area_name = EXCLUDED.functional_area_name,
                        functional_role_name = EXCLUDED.functional_role_name,
                        avg_experience = EXCLUDED.avg_experience,
                        avg_wage = EXCLUDED.avg_wage,
                        gender_code = EXCLUDED.gender_code,
                        highest_qualification = EXCLUDED.highest_qualification,
                        state_name = EXCLUDED.state_name,
                        district_name = EXCLUDED.district_name,
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        pin_code = EXCLUDED.pin_code,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        updated_at = NOW()
                    """

                    await conn.execute(
                        insert_query,
                        job_data.keywords,
                        job_data.ncsp_job_id,
                        job_data.date,
                        job_data.organization_id,
                        job_data.organization_name,
                        job_data.number_of_openings,
                        job_data.industry_name,
                        job_data.sector_name,
                        job_data.functional_area_name,
                        job_data.functional_role_name,
                        job_data.avg_experience,
                        job_data.avg_wage,
                        job_data.gender_code,
                        job_data.highest_qualification,
                        job_data.state_name,
                        job_data.district_name,
                        job_data.title,
                        job_data.description,
                        job_data.pin_code,
                        job_data.latitude,
                        job_data.longitude
                    )

                    jobs_processed += 1

                except Exception as e:
                    logger.error(f"Error processing job item: {e}")
                    continue

        return ConsumeAPIResponse(
            message=f"Successfully processed {jobs_processed} test jobs",
            jobs_processed=jobs_processed,
            success=True
        )

    except Exception as e:
        logger.error(f"Error in consume_api_test_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
