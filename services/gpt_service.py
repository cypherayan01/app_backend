import re
import json
import logging
from typing import List, Dict, Any

from config.settings import azure_client, gpt_deployment

logger = logging.getLogger(__name__)


class GPTService:
    """Service class for GPT-based reranking using Azure OpenAI"""

    @staticmethod
    async def rerank_jobs(skills: List[str], jobs: List[Dict]) -> List[Dict]:
        """Re-rank jobs using Azure OpenAI GPT"""

        if not jobs:
            return []

        # Prepare job data for GPT
        processed_jobs = []
        for job in jobs[:25]:  # Limit to top 25 for GPT processing
            processed_job = {
                "ncspjobid": job["ncspjobid"],
                "title": job["title"],
                "keywords": job.get("keywords", "")[:200],
                "description": job.get("description", "")[:300] if job.get("description") else "",
                "similarity": round(job.get("similarity", 0), 3)
            }
            processed_jobs.append(processed_job)

        jobs_json = json.dumps(processed_jobs, indent=2)
        skills_str = ', '.join(skills)

        prompt = f"""
You are an expert job matcher. Analyze the job seeker's skills and rank the jobs by relevance.

Job Seeker Skills: {skills_str}

Jobs to rank:
{jobs_json}

Instructions:
1. Rank jobs from best to worst match based on skill alignment
2. Assign match_percentage between 100-40 based on how well skills align with job requirements
3. Consider exact skill matches, related skills, and transferable skills
4. Higher percentage for closer skill matches
5. Return ONLY valid JSON array
6. Give me only unique ncspjobid in the json array correctly.

Required format: [{{"ncspjobid": 123, "title": "Job Title", "match_percentage": 85}}, ...]
"""

        try:
            logger.info("Reranking jobs with Azure GPT...")

            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled career advisor. Return only valid JSON array. No explanation text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            # Clean response
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()

            try:
                ranked_jobs = json.loads(content)
                if isinstance(ranked_jobs, list) and len(ranked_jobs) > 0:
                    logger.info(f"Successfully ranked {len(ranked_jobs)} jobs")
                    return ranked_jobs
                else:
                    logger.warning("GPT returned empty or invalid list")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response: {e}")

            # Fallback to similarity-based ranking
            return GPTService._fallback_ranking(skills, processed_jobs)

        except Exception as e:
            logger.error(f"GPT reranking failed: {e}")
            return GPTService._fallback_ranking(skills, processed_jobs)

    @staticmethod
    def _fallback_ranking(skills: List[str], processed_jobs: List[Dict]) -> List[Dict]:
        """Fallback to intelligent skill-based ranking"""
        logger.info("Using intelligent skill-based fallback ranking")
        fallback_jobs = []

        for job in processed_jobs:
            similarity = job.get("similarity", 0.0)
            job_keywords = job.get("keywords", "").lower()
            job_title = job.get("title", "").lower()
            job_description = job.get("description", "").lower()

            # Parse job keywords into list
            job_keyword_list = [kw.strip() for kw in job.get("keywords", "").split(",") if kw.strip()]

            # Combine all job text for matching
            job_text = f"{job_keywords} {job_title} {job_description}"

            # Track skill matching details
            skill_matches = 0
            partial_matches = 0
            matched_job_keywords = []
            unmatched_job_keywords = []
            user_skills_matched = []
            total_skills = len(skills)

            # Check each job keyword against user skills
            for job_kw in job_keyword_list:
                job_kw_lower = job_kw.lower().strip()
                kw_matched = False

                for skill in skills:
                    skill_lower = skill.lower()

                    # Check if user skill matches this job keyword
                    if skill_lower == job_kw_lower or skill_lower in job_kw_lower or job_kw_lower in skill_lower:
                        matched_job_keywords.append(job_kw)
                        if skill not in user_skills_matched:
                            user_skills_matched.append(f"{skill} (keywords)")
                        kw_matched = True
                        break

                if not kw_matched:
                    unmatched_job_keywords.append(job_kw)

            # Now check user skills for matches in title/description if not found in keywords
            for skill in skills:
                skill_lower = skill.lower()
                already_matched = any(skill in matched for matched in user_skills_matched)

                if not already_matched:
                    # Title match (high weight)
                    if skill_lower in job_title:
                        skill_matches += 0.8
                        user_skills_matched.append(f"{skill} (title)")
                    # Description match (medium weight)
                    elif skill_lower in job_description:
                        skill_matches += 0.6
                        user_skills_matched.append(f"{skill} (description)")
                    # Partial match (low weight)
                    elif any(skill_lower in word or word in skill_lower for word in job_text.split() if len(word) > 2):
                        partial_matches += 0.3
                        user_skills_matched.append(f"{skill} (partial)")

            # Calculate keyword match score based on matched job keywords
            keyword_matches_count = len(matched_job_keywords)
            if job_keyword_list:
                keyword_match_score = keyword_matches_count / len(job_keyword_list)
            else:
                keyword_match_score = 0

            # Calculate user skill match score
            user_skill_score = len(user_skills_matched) / total_skills if total_skills > 0 else 0

            # Combine both scores (70% user skills, 30% keyword coverage)
            combined_score = (user_skill_score * 0.7) + (keyword_match_score * 0.3)

            # Add similarity component (20% weight)
            final_score = (combined_score * 0.8) + (similarity * 0.2)

            # Convert to percentage with realistic ranges
            if final_score >= 0.8:
                match_percentage = 85 + (final_score - 0.8) * 75  # 85-100%
            elif final_score >= 0.6:
                match_percentage = 70 + (final_score - 0.6) * 75  # 70-85%
            elif final_score >= 0.4:
                match_percentage = 55 + (final_score - 0.4) * 75  # 55-70%
            elif final_score >= 0.2:
                match_percentage = 40 + (final_score - 0.2) * 75  # 40-55%
            else:
                match_percentage = 25 + final_score * 75  # 25-40%

            # Cap at reasonable limits
            match_percentage = max(25, min(98, match_percentage))

            fallback_jobs.append({
                "ncspjobid": job["ncspjobid"],
                "title": job["title"],
                "match_percentage": round(match_percentage, 1),
                "keywords_matched": matched_job_keywords,
                "keywords_unmatched": unmatched_job_keywords,
                "user_skills_matched": user_skills_matched,
                "keyword_match_score": round(keyword_match_score, 2),
                "similarity_used": round(similarity, 3)
            })
        return fallback_jobs


class QueryClassificationService:
    """Service class for intelligent query classification using Azure OpenAI"""

    @staticmethod
    async def classify_and_extract(user_query: str) -> Dict[str, Any]:
        """
        Classify user query and extract skills/locations using Azure OpenAI.

        Returns:
            Dict with keys:
            - query_type: "skill_only", "location_only", "skill_location", or "general"
            - skills: List of extracted skills
            - location: Extracted location string (or None)
            - confidence: Confidence score (0-1)
        """

        if not user_query or len(user_query.strip()) < 3:
            return {
                'query_type': 'general',
                'skills': [],
                'location': None,
                'confidence': 0.0
            }

        prompt = f"""
You are an intelligent job search query analyzer. Analyze the user's query and extract job search intent.

User Query: "{user_query}"

Your task:
1. Determine the query type:
   - "skill_only": User is searching for jobs based on skills/technologies only
   - "location_only": User is searching for jobs in a specific location only
   - "skill_location": User is searching for jobs with both skills AND location
   - "general": General conversation, greetings, or unclear intent

2. Extract skills/technologies mentioned (programming languages, frameworks, tools, job roles, etc.)
   Examples: Java, Python, React, Data Analyst, Machine Learning, etc.

3. Extract location if mentioned (city, state, region)
   Examples: Mumbai, Bangalore, Maharashtra, etc.

4. Provide confidence score (0.0 to 1.0) for your classification

Return ONLY valid JSON in this exact format:
{{
  "query_type": "skill_only" | "location_only" | "skill_location" | "general",
  "skills": ["skill1", "skill2"],
  "location": "location_name" or null,
  "confidence": 0.95
}}

Examples:
- "Hey, I am a Java Developer. Can you find any job openings for me?"
  → {{"query_type": "skill_only", "skills": ["Java"], "location": null, "confidence": 0.95}}

- "Show me jobs in Mumbai"
  → {{"query_type": "location_only", "skills": [], "location": "Mumbai", "confidence": 0.98}}

- "I need Python developer jobs in Bangalore"
  → {{"query_type": "skill_location", "skills": ["Python"], "location": "Bangalore", "confidence": 0.97}}

- "Hello, how are you?"
  → {{"query_type": "general", "skills": [], "location": null, "confidence": 0.99}}
"""

        try:
            logger.info(f"Classifying query with Azure GPT: {user_query[:100]}...")

            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a job search query analyzer. Return ONLY valid JSON. No explanation text. No markdown."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Clean response
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()

            try:
                result = json.loads(content)

                # Validate result structure
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")

                # Ensure required fields exist
                query_type = result.get('query_type', 'general')
                skills = result.get('skills', [])
                location = result.get('location')
                confidence = result.get('confidence', 0.0)

                # Normalize skills list
                if not isinstance(skills, list):
                    skills = [str(skills)] if skills else []

                # Clean and validate skills
                skills = [s.strip() for s in skills if s and str(s).strip()]

                # Clean location
                if location:
                    location = str(location).strip()
                    if not location or location.lower() in ['null', 'none', 'n/a']:
                        location = None

                logger.info(f"✓ Query classified - Type: {query_type}, Skills: {skills}, Location: {location}, Confidence: {confidence}")

                return {
                    'query_type': query_type,
                    'skills': skills,
                    'location': location,
                    'confidence': float(confidence)
                }

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON: {e}")
                logger.error(f"Response content: {content}")
                return {
                    'query_type': 'general',
                    'skills': [],
                    'location': None,
                    'confidence': 0.0
                }

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return {
                'query_type': 'general',
                'skills': [],
                'location': None,
                'confidence': 0.0
            }
