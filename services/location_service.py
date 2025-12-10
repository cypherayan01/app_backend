import re
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional

import asyncpg

from config.database import DatabasePool
from models.job import LocationJobRequest, LocationJobResponse

logger = logging.getLogger(__name__)


class LocationMappingService:
    """Service to handle location name mapping and normalization"""

    def __init__(self):
        # Comprehensive location mappings
        self.city_to_state = {
            # Major Cities to States
            'mumbai': 'Maharashtra', 'bombay': 'Maharashtra',
            'pune': 'Maharashtra', 'nagpur': 'Maharashtra', 'nashik': 'Maharashtra',
            'aurangabad': 'Maharashtra', 'solapur': 'Maharashtra', 'kolhapur': 'Maharashtra',

            'delhi': 'Delhi', 'new delhi': 'Delhi', 'gurgaon': 'Haryana', 'gurugram': 'Haryana',
            'noida': 'Uttar Pradesh', 'ghaziabad': 'Uttar Pradesh', 'faridabad': 'Haryana',

            'bangalore': 'Karnataka', 'bengaluru': 'Karnataka', 'mysore': 'Karnataka',
            'hubli': 'Karnataka', 'mangalore': 'Karnataka', 'belgaum': 'Karnataka',

            'chennai': 'Tamil Nadu', 'madras': 'Tamil Nadu', 'coimbatore': 'Tamil Nadu',
            'madurai': 'Tamil Nadu', 'salem': 'Tamil Nadu', 'tiruchirapalli': 'Tamil Nadu',
            'trichy': 'Tamil Nadu', 'vellore': 'Tamil Nadu',

            'hyderabad': 'Telangana', 'secunderabad': 'Telangana', 'warangal': 'Telangana',

            'kolkata': 'West Bengal', 'calcutta': 'West Bengal', 'durgapur': 'West Bengal',
            'siliguri': 'West Bengal', 'howrah': 'West Bengal',

            'ahmedabad': 'Gujarat', 'surat': 'Gujarat', 'vadodara': 'Gujarat',
            'rajkot': 'Gujarat', 'bhavnagar': 'Gujarat', 'jamnagar': 'Gujarat',

            'jaipur': 'Rajasthan', 'udaipur': 'Rajasthan', 'jodhpur': 'Rajasthan',
            'kota': 'Rajasthan', 'bikaner': 'Rajasthan', 'ajmer': 'Rajasthan',

            'lucknow': 'Uttar Pradesh', 'kanpur': 'Uttar Pradesh', 'agra': 'Uttar Pradesh',
            'varanasi': 'Uttar Pradesh', 'meerut': 'Uttar Pradesh', 'allahabad': 'Uttar Pradesh',
            'prayagraj': 'Uttar Pradesh', 'bareilly': 'Uttar Pradesh',

            'bhopal': 'Madhya Pradesh', 'indore': 'Madhya Pradesh', 'gwalior': 'Madhya Pradesh',
            'jabalpur': 'Madhya Pradesh', 'ujjain': 'Madhya Pradesh',

            'patna': 'Bihar', 'gaya': 'Bihar', 'bhagalpur': 'Bihar', 'muzaffarpur': 'Bihar',

            'bhubaneswar': 'Odisha', 'cuttack': 'Odisha', 'rourkela': 'Odisha',

            'chandigarh': 'Chandigarh', 'amritsar': 'Punjab', 'ludhiana': 'Punjab',
            'jalandhar': 'Punjab', 'patiala': 'Punjab',

            'kochi': 'Kerala', 'cochin': 'Kerala', 'thiruvananthapuram': 'Kerala',
            'trivandrum': 'Kerala', 'kozhikode': 'Kerala', 'calicut': 'Kerala',
            'thrissur': 'Kerala', 'kollam': 'Kerala',

            'visakhapatnam': 'Andhra Pradesh', 'vijayawada': 'Andhra Pradesh',
            'guntur': 'Andhra Pradesh', 'nellore': 'Andhra Pradesh', 'tirupati': 'Andhra Pradesh',

            'guwahati': 'Assam', 'dibrugarh': 'Assam', 'jorhat': 'Assam',

            'ranchi': 'Jharkhand', 'jamshedpur': 'Jharkhand', 'dhanbad': 'Jharkhand',

            'raipur': 'Chhattisgarh', 'bilaspur': 'Chhattisgarh',

            'panaji': 'Goa', 'margao': 'Goa',

            'dehradun': 'Uttarakhand', 'haridwar': 'Uttarakhand'
        }

        # State name normalization
        self.state_aliases = {
            'mh': 'Maharashtra', 'maharashtra': 'Maharashtra',
            'ka': 'Karnataka', 'karnataka': 'Karnataka',
            'tn': 'Tamil Nadu', 'tamil nadu': 'Tamil Nadu', 'tamilnadu': 'Tamil Nadu',
            'ts': 'Telangana', 'telangana': 'Telangana',
            'ap': 'Andhra Pradesh', 'andhra pradesh': 'Andhra Pradesh',
            'wb': 'West Bengal', 'west bengal': 'West Bengal',
            'gj': 'Gujarat', 'gujarat': 'Gujarat',
            'rj': 'Rajasthan', 'rajasthan': 'Rajasthan',
            'up': 'Uttar Pradesh', 'uttar pradesh': 'Uttar Pradesh',
            'mp': 'Madhya Pradesh', 'madhya pradesh': 'Madhya Pradesh',
            'dl': 'Delhi', 'delhi': 'Delhi',
            'hr': 'Haryana', 'haryana': 'Haryana',
            'pb': 'Punjab', 'punjab': 'Punjab',
            'or': 'Odisha', 'odisha': 'Odisha', 'orissa': 'Odisha',
            'jh': 'Jharkhand', 'jharkhand': 'Jharkhand',
            'cg': 'Chhattisgarh', 'chhattisgarh': 'Chhattisgarh',
            'uk': 'Uttarakhand', 'uttarakhand': 'Uttarakhand',
            'br': 'Bihar', 'bihar': 'Bihar',
            'as': 'Assam', 'assam': 'Assam',
            'kl': 'Kerala', 'kerala': 'Kerala',
            'ga': 'Goa', 'goa': 'Goa'
        }

        # Regional keywords
        self.regional_keywords = {
            'north india': ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Uttarakhand', 'Rajasthan'],
            'south india': ['Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Kerala'],
            'west india': ['Maharashtra', 'Gujarat', 'Rajasthan', 'Goa'],
            'east india': ['West Bengal', 'Odisha', 'Jharkhand', 'Bihar'],
            'central india': ['Madhya Pradesh', 'Chhattisgarh'],
            'metro cities': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata'],
            'tier 1 cities': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad'],
        }

    def normalize_location(self, location_input: str) -> Dict[str, List[str]]:
        """
        Normalize location input and return potential states and districts
        Returns: {"states": [list], "districts": [list]}
        """
        location_lower = location_input.lower().strip()
        location_lower = re.sub(r'\b(city|district|state|region|area)\b', '', location_lower).strip()

        states = []
        districts = []

        # Check regional keywords first
        for region, region_states in self.regional_keywords.items():
            if region in location_lower:
                if region in ['metro cities', 'tier 1 cities']:
                    districts.extend(region_states)
                else:
                    states.extend(region_states)
                return {"states": states, "districts": districts}

        # Check if it's a direct state match
        if location_lower in self.state_aliases:
            states.append(self.state_aliases[location_lower])
            return {"states": states, "districts": districts}

        # Check if it's a city/district
        if location_lower in self.city_to_state:
            districts.append(location_input.title())
            corresponding_state = self.city_to_state[location_lower]
            if corresponding_state not in states:
                states.append(corresponding_state)
            return {"states": states, "districts": districts}

        # Fuzzy matching for typos or variations
        for city, state in self.city_to_state.items():
            if city in location_lower or location_lower in city:
                districts.append(city.title())
                if state not in states:
                    states.append(state)

        for alias, state in self.state_aliases.items():
            if alias in location_lower or location_lower in alias:
                if state not in states:
                    states.append(state)

        # If no matches found, treat as potential district name
        if not states and not districts:
            districts.append(location_input.title())

        return {"states": list(set(states)), "districts": list(set(districts))}


class LocationJobSearchService:
    """Enhanced service for location-based job searching"""

    def __init__(self):
        self.location_mapper = LocationMappingService()

    async def search_jobs_by_location(self, request: LocationJobRequest) -> LocationJobResponse:
        """Search jobs by location with comprehensive filtering"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Normalize location input
            location_matches = self.location_mapper.normalize_location(request.location)
            logger.info(f"Extracted Location location_matches : {location_matches}")

            if not location_matches["states"] and not location_matches["districts"]:
                return LocationJobResponse(
                    location_searched=request.location,
                    location_matches=location_matches,
                    jobs=[],
                    total_found=0,
                    returned_count=0,
                    processing_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000),
                    filters_applied={},
                    search_context={"error": "Location not recognized"}
                )

            # Build dynamic SQL query
            query, params = self._build_location_query(request, location_matches)

            # Execute query
            jobs = await self._execute_location_query(query, params)

            # Apply additional filtering if needed
            if request.skills:
                jobs = await self._filter_by_skills(jobs, request.skills)

            # Apply sorting
            jobs = self._apply_sorting(jobs, request.sort_by)

            # Limit results
            limited_jobs = jobs[:request.limit]

            processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return LocationJobResponse(
                location_searched=request.location,
                location_matches=location_matches,
                jobs=limited_jobs,
                total_found=len(jobs),
                returned_count=len(limited_jobs),
                processing_time_ms=processing_time_ms,
                filters_applied=self._get_applied_filters(request),
                search_context={
                    "query_executed": True,
                    "location_normalized": location_matches,
                    "total_before_limit": len(jobs)
                }
            )

        except Exception as e:
            logger.error(f"Location job search failed: {e}")
            processing_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return LocationJobResponse(
                location_searched=request.location,
                location_matches={},
                jobs=[],
                total_found=0,
                returned_count=0,
                processing_time_ms=processing_time_ms,
                filters_applied={},
                search_context={"error": str(e)}
            )

    def _build_location_query(
        self,
        request: LocationJobRequest,
        location_matches: Dict[str, List[str]]
    ) -> Tuple[str, List]:
        """
        Azure-compatible location query builder

        Issues Fixed:
        1. ILIKE operator compatibility
        2. Parameterized IN clauses
        3. Text array casting
        """
        base_query = """
            SELECT ncspjobid, title, keywords, description, organization_name,
                statename, districtname, industryname, sectorname,
                functionalareaname, functionalrolename, aveexp, avewage,
                numberofopenings, highestqualification, gendercode, date
            FROM vacancies_summary
        """

        conditions = []
        params = []
        idx = 1

        # Location conditions - FIXED: Use array approach
        if location_matches.get('states'):
            # Build OR conditions for states
            state_conditions = []
            for state in location_matches['states']:
                state_conditions.append(f"LOWER(statename) = LOWER(${idx})")
                params.append(state)
                idx += 1
            if state_conditions:
                conditions.append(f"({' OR '.join(state_conditions)})")

        if location_matches.get('districts'):
            district_conditions = []
            for district in location_matches['districts']:
                district_conditions.append(f"LOWER(districtname) = LOWER(${idx})")
                params.append(district)
                idx += 1
            if district_conditions:
                conditions.append(f"({' OR '.join(district_conditions)})")

        # Job type filter - FIXED: Avoid ILIKE on Azure
        if request.job_type:
            job_type_pattern = f"%{request.job_type.lower()}%"
            conditions.append(f"""(
                LOWER(title) LIKE ${idx} OR
                LOWER(keywords) LIKE ${idx} OR
                LOWER(functionalrolename) LIKE ${idx}
            )""")
            params.extend([job_type_pattern] * 3)
            idx += 3

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY ncspjobid DESC LIMIT 2000"

        logger.info(f"Built query with {len(params)} params")
        return base_query, params

    async def _execute_location_query(
        self,
        query: str,
        params: List
    ) -> List[Dict]:
        """Execute location query with connection pooling"""
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(query, *params)

                jobs = []
                for row in rows:
                    jobs.append({
                        'ncspjobid': row['ncspjobid'],
                        'title': row['title'],
                        'keywords': row['keywords'],
                        'description': row['description'],
                        'organization_name': row['organization_name'],
                        'statename': row['statename'],
                        'districtname': row['districtname'],
                        'industryname': row['industryname'],
                        'sectorname': row['sectorname'],
                        'functionalareaname': row['functionalareaname'],
                        'functionalrolename': row['functionalrolename'],
                        'aveexp': float(row['aveexp']) if row['aveexp'] else 0,
                        'avewage': float(row['avewage']) if row['avewage'] else 0,
                        'numberofopenings': int(row['numberofopenings']) if row['numberofopenings'] else 1,
                        'highestqualification': row['highestqualification'],
                        'gendercode': row['gendercode'],
                        'date': row['date'].isoformat() if row['date'] else None,
                        'match_percentage': 75
                    })

                return jobs

        except asyncpg.PostgresError as e:
            logger.error(f"Query execution failed: {e.sqlstate} - {e.message}")
            return []
        except Exception as e:
            logger.error(f"Unexpected query error: {e}")
            return []

    async def _filter_by_skills(self, jobs: List[Dict], skills: List[str]) -> List[Dict]:
        """Filter jobs by skills with enhanced matching and update match percentage"""
        if not skills:
            return jobs

        filtered_jobs = []
        skills_lower = [skill.lower().strip() for skill in skills]

        # Create skill variations for better matching
        skill_variations = {}
        for skill in skills_lower:
            variations = [skill]
            # Add common variations
            if 'data entry' in skill:
                variations.extend(['data processing', 'typing', 'data operator', 'data entry operator'])
            elif 'python' in skill:
                variations.extend(['python developer', 'python programming', 'django', 'flask'])
            elif 'javascript' in skill or 'js' in skill:
                variations.extend(['javascript developer', 'js developer', 'frontend developer'])
            elif 'customer service' in skill:
                variations.extend(['call center', 'voice process', 'customer support', 'telecalling'])

            skill_variations[skill] = variations

        for job in jobs:
            # Get job text for matching
            job_text = f"{job.get('title', '')} {job.get('keywords', '')} {job.get('description', '')} {job.get('functionalrolename', '')}".lower()

            skill_matches = 0
            matched_skills = []
            partial_matches = []

            for skill in skills_lower:
                # Check exact match first
                if skill in job_text:
                    skill_matches += 1
                    matched_skills.append(skill)
                # Check skill variations
                elif any(var in job_text for var in skill_variations.get(skill, [])):
                    skill_matches += 0.8
                    matched_skills.append(skill)
                # Check partial word matches
                elif any(word in job_text for word in skill.split() if len(word) > 2):
                    skill_matches += 0.3
                    partial_matches.append(skill)

            # More lenient filtering - include jobs with any skill match
            if skill_matches >= 0.3:  # Lower threshold for inclusion
                skill_match_ratio = skill_matches / len(skills)
                # Calculate match percentage based on skill matching
                base_score = 60 if matched_skills else 45
                job['match_percentage'] = min(95, base_score + (skill_match_ratio * 35))
                job['skills_matched'] = matched_skills if matched_skills else partial_matches
                filtered_jobs.append(job)

        # If filtering results in too few jobs, return original list with updated scores
        if len(filtered_jobs) < 5 and len(jobs) > 0:
            logger.info(f"Skill filtering too restrictive, returning all jobs with updated scores")
            for job in jobs[:50]:  # Return top 50 from location search
                job_text = f"{job.get('title', '')} {job.get('keywords', '')}".lower()
                # Calculate a basic match score
                matches = sum(1 for skill in skills_lower if skill in job_text)
                job['match_percentage'] = min(75, 40 + (matches / len(skills)) * 35)
                job['skills_matched'] = [s for s in skills_lower if s in job_text]
            return jobs[:50]

        return sorted(filtered_jobs, key=lambda x: x.get('match_percentage', 0), reverse=True)

    def _apply_sorting(self, jobs: List[Dict], sort_by: str) -> List[Dict]:
        """Apply sorting to job results"""
        if sort_by == "salary":
            return sorted(jobs, key=lambda x: x.get('avewage', 0), reverse=True)
        elif sort_by == "experience":
            return sorted(jobs, key=lambda x: x.get('aveexp', 0), reverse=True)
        elif sort_by == "date":
            return sorted(jobs, key=lambda x: x.get('date', ''), reverse=True)
        else:  # relevance (default)
            return sorted(jobs, key=lambda x: x.get('match_percentage', 0), reverse=True)

    def _get_applied_filters(self, request: LocationJobRequest) -> Dict[str, Any]:
        """Get summary of applied filters"""
        filters = {"location": request.location}

        if request.job_type:
            filters["job_type"] = request.job_type
        if request.skills:
            filters["skills"] = request.skills
        if request.experience_range:
            filters["experience_range"] = request.experience_range
        if request.salary_range:
            filters["salary_range"] = request.salary_range

        filters["sort_by"] = request.sort_by
        return filters
