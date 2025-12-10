import logging
from typing import List, Dict
import asyncpg
from config.database import DB_CONFIG

logger = logging.getLogger(__name__)


async def get_complete_job_details(job_ids: List[str]) -> List[Dict]:
    """Fetch complete job details from database for given job IDs"""
    if not job_ids:
        return []

    try:
        # Use DB_CONFIG for connection
        conn = await asyncpg.connect(**DB_CONFIG)

        try:
            rows = await conn.fetch("""
                SELECT
                    ncspjobid, title, keywords, description,
                    CASE WHEN date IS NOT NULL THEN TO_CHAR(date, 'YYYY-MM-DD') ELSE NULL END as date,
                    organizationid, organization_name, numberofopenings,
                    industryname, sectorname, functionalareaname,
                    functionalrolename,COALESCE(CAST(ROUND(aveexp) AS INT), 0) AS aveexp,
                    COALESCE(ROUND(avewage::numeric, 2), 0.00) AS avewage, gendercode,
                    highestqualification, statename, districtname
                FROM vacancies_summary
                WHERE ncspjobid = ANY($1::text[])
                ORDER BY ncspjobid;
            """, job_ids)
            print(rows[0])
            return [dict(row) for row in rows]

        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Failed to fetch job details: {e}")
        return []
