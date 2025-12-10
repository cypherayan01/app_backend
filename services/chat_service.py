import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from models.chat import ChatRequest, ChatResponse
from models.job import LocationJobRequest

logger = logging.getLogger(__name__)


class EnhancedChatService:
    """Complete enhanced chat service that integrates location search with existing chat functionality"""

    def __init__(self):
        # Import here to avoid circular imports
        from services.location_service import LocationJobSearchService, LocationMappingService
        self.location_job_service = LocationJobSearchService()
        self.location_mapper = LocationMappingService()
        self.fallback_count = {}  # Track fallback responses to prevent loops
        self.conversation_state = {}  # Track conversation state per user

        # Complete skill keywords dictionary
        self.skill_keywords = {
            # Technical/IT Skills
            'python': ['python', 'py', 'django', 'flask', 'fastapi'],
            'javascript': ['javascript', 'js', 'node', 'nodejs', 'node.js'],
            'react': ['react', 'reactjs', 'react.js', 'nextjs', 'next.js'],
            'angular': ['angular', 'angularjs'],
            'vue': ['vue', 'vuejs', 'vue.js', 'nuxt'],
            'java': ['java', 'spring', 'springboot', 'spring boot'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'c sharp', '.net', 'dotnet'],
            'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'sqlite'],
            'mongodb': ['mongodb', 'mongo'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3', 'sass', 'scss', 'tailwind'],
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'git': ['git', 'github', 'gitlab'],
            'machine learning': ['ml', 'machine learning', 'tensorflow', 'pytorch', 'scikit-learn', 'keras'],
            'data science': ['data science', 'pandas', 'numpy', 'matplotlib', 'jupyter'],
            'typescript': ['typescript', 'ts'],
            'php': ['php', 'laravel', 'symfony'],
            'ruby': ['ruby', 'rails', 'ruby on rails'],

            # Business Process/BPO Skills
            'data entry': ['data entry', 'data processing', 'typing', 'keyboarding'],
            'voice process': ['voice process', 'call center', 'customer service', 'telecalling', 'telesales'],
            'chat process': ['chat process', 'chat support', 'live chat', 'online support'],
            'email support': ['email support', 'email handling', 'email management'],
            'back office': ['back office', 'administrative', 'admin work'],
            'content writing': ['content writing', 'copywriting', 'blogging', 'article writing'],
            'virtual assistant': ['virtual assistant', 'va', 'personal assistant'],

            # Finance & Accounting
            'accounting': ['accounting', 'bookkeeping', 'accounts', 'financial'],
            'tally': ['tally', 'tally erp'],
            'excel': ['excel', 'microsoft excel', 'spreadsheet', 'vlookup', 'pivot tables'],
            'sap': ['sap', 'sap fico', 'sap mm', 'sap hr'],
            'quickbooks': ['quickbooks', 'quick books'],
            'gst': ['gst', 'goods and services tax', 'taxation'],
            'payroll': ['payroll', 'salary processing', 'hr payroll'],

            # Sales & Marketing
            'sales': ['sales', 'selling', 'business development', 'lead generation'],
            'digital marketing': ['digital marketing', 'online marketing', 'internet marketing'],
            'seo': ['seo', 'search engine optimization'],
            'sem': ['sem', 'search engine marketing', 'google ads', 'ppc'],
            'social media': ['social media', 'facebook marketing', 'instagram marketing', 'linkedin'],
            'email marketing': ['email marketing', 'mailchimp', 'newsletter'],
            'content marketing': ['content marketing', 'inbound marketing'],

            # Healthcare & Medical
            'nursing': ['nursing', 'nurse', 'rn', 'lpn'],
            'medical': ['medical', 'healthcare', 'clinical'],
            'pharmacy': ['pharmacy', 'pharmacist', 'pharma'],

            # Education & Training
            'teaching': ['teaching', 'teacher', 'education', 'tutor'],
            'training': ['training', 'corporate training', 'soft skills'],

            # Manufacturing & Operations
            'manufacturing': ['manufacturing', 'production', 'operations'],
            'quality control': ['quality control', 'qc', 'quality assurance', 'qa'],
            'logistics': ['logistics', 'supply chain', 'warehouse'],

            # Human Resources
            'hr': ['hr', 'human resources', 'recruitment', 'talent acquisition'],

            # Design & Creative
            'graphic design': ['graphic design', 'design', 'photoshop', 'illustrator'],
            'ui/ux': ['ui', 'ux', 'user interface', 'user experience'],
            'video editing': ['video editing', 'premiere', 'after effects'],

            # Legal & Compliance
            'legal': ['legal', 'law', 'compliance', 'contracts'],
            'paralegal': ['paralegal', 'legal assistant'],

            # Project Management
            'project management': ['project management', 'pmp', 'agile', 'scrum'],
            'business analyst': ['business analyst', 'ba', 'requirements'],
        }

    async def handle_chat_message(self, request: ChatRequest) -> ChatResponse:
        """Main chat handler that routes queries based on intent (skill-only, location-only, or skill+location)"""
        message = request.message.lower().strip()
        user_id = id(request)

        try:
            # Reset fallback count if message has meaningful content
            if len(message) > 3 and not message in ['okay', 'ok', 'yes', 'no']:
                self.fallback_count[user_id] = 0

            # Parse query intent - detect skill-only, location-only, or combined location + skill queries
            query_intent = await self._parse_query_intent(message)
            query_type = query_intent.get('query_type', 'general')

            logger.info(f"=== Routing query with type: {query_type} ===")

            # Route based on query type
            if query_type in ['location_only', 'location_skill', 'skill_location']:
                # Location-based search (with or without skills)
                logger.info(f"Routing to location handler with skills: {query_intent.get('skills', [])}")
                return await self._handle_location_job_query(request, query_intent)

            elif query_type == 'skill_only':
                # Skill-based search (no location)
                logger.info(f"Routing to skill handler with skills: {query_intent.get('skills', [])}")
                return await self._handle_skill_job_query(request, query_intent)

            else:
                # General conversation or unclear intent - use regular chat
                logger.info("Routing to regular chat handler")
                return await self._handle_regular_chat(request)

        except Exception as e:
            logger.error(f"Enhanced chat failed: {e}")
            # Track fallback to prevent loops
            self.fallback_count[user_id] = self.fallback_count.get(user_id, 0) + 1

            if self.fallback_count.get(user_id, 0) > 2:
                # After 2 fallbacks, provide more specific guidance
                return ChatResponse(
                    response="Let me help you get started! Here are some specific examples:\n\n1. 'Show me Python jobs in Mumbai'\n2. 'Data Entry positions in Delhi'\n3. 'I know Python and React'\n4. Upload your CV using the file upload option",
                    message_type="text",
                    chat_phase="intro",
                    suggestions=["Python jobs in Mumbai", "Data Entry in Delhi", "Upload CV", "I know Python"]
                )

            return ChatResponse(
                response="I can help you find jobs! You can ask about specific locations like 'Jobs in Mumbai' or tell me about your skills.",
                message_type="text",
                chat_phase="profile_building",
                suggestions=["Jobs in Mumbai", "My skills are...", "Remote work", "Entry level jobs"]
            )

    async def _parse_query_intent(self, message: str) -> Dict[str, Any]:
        """Parse user query to extract location, skills, and intent - handles combined queries"""
        intent = {
            'has_location': False,
            'has_skills': False,
            'location': None,
            'skills': [],
            'job_type': None,
            'query_type': 'general'
        }

        # Import here to avoid circular imports
        from services.gpt_service import QueryClassificationService
        query_classifier = QueryClassificationService()

        # STEP 1: Try Azure OpenAI-based intelligent classification first
        try:
            logger.info("=== Using Azure OpenAI Query Classification ===")
            classification = await query_classifier.classify_and_extract(message)

            # If classification has high confidence and is not general, use it
            if classification['confidence'] >= 0.7 and classification['query_type'] != 'general':
                logger.info(f"Using AI classification: {classification}")

                # Map query_type from AI to internal format
                ai_query_type = classification['query_type']
                if ai_query_type == 'skill_only':
                    intent['query_type'] = 'skill_only'
                    intent['has_skills'] = True
                    intent['skills'] = classification['skills']
                elif ai_query_type == 'location_only':
                    intent['query_type'] = 'location_only'
                    intent['has_location'] = True
                    intent['location'] = classification['location']
                elif ai_query_type == 'skill_location':
                    intent['query_type'] = 'location_skill'
                    intent['has_skills'] = True
                    intent['has_location'] = True
                    intent['skills'] = classification['skills']
                    intent['location'] = classification['location']

                logger.info(f"AI-based intent - Location: {intent['location']}, Skills: {intent['skills']}, Type: {intent['query_type']}")
                return intent
            else:
                logger.info(f"AI classification confidence too low ({classification['confidence']}) or general query, falling back to regex")

        except Exception as e:
            logger.warning(f"AI classification failed, falling back to regex patterns: {e}")

        # STEP 2: Fallback to regex-based pattern matching
        logger.info("=== Using Regex-based Pattern Matching ===")

        # Known cities and states for better detection
        known_locations = [
            'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'hyderabad',
            'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri', 'patna',
            'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut',
            'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad',
            'amritsar', 'navi mumbai', 'allahabad', 'ranchi', 'howrah', 'coimbatore',
            'maharashtra', 'karnataka', 'tamil nadu', 'delhi', 'telangana', 'andhra pradesh',
            'west bengal', 'gujarat', 'rajasthan', 'uttar pradesh', 'madhya pradesh'
        ]

        # Enhanced patterns to detect combined queries
        combined_patterns = [
            # Pattern: "jobs in [location] on/for [skill]"
            r'(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)\s+(?:on|for|in|with|of)\s+([a-zA-Z\s]+)',
            # Pattern: "[skill] jobs in [location]"
            r'([a-zA-Z\s]+)\s+(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)',
            # Pattern: "show me [skill] in [location]"
            r'(?:show|find|get|search)\s+(?:me\s+)?([a-zA-Z\s]+)\s+(?:jobs?|positions?)?\s+in\s+([a-zA-Z\s]+)',
            # Pattern: "show/find/get/give me jobs in [location]" (location only)
            r'(?:show|find|get|give|search)\s+(?:me\s+)?(?:all\s+)?(?:jobs?|positions?|openings?|vacancies?)\s+in\s+([a-zA-Z\s]+)$',
            # Pattern: "jobs/positions in [location]" (location only - start of message)
            r'^(?:jobs?|positions?|openings?|vacancies?)\s+in\s+([a-zA-Z\s]+)$',
        ]

        # Check combined patterns first
        for pattern_idx, pattern in enumerate(combined_patterns):
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                # Handle location-only patterns (indices 3 and 4)
                if pattern_idx in [3, 4]:
                    # These patterns only extract location
                    part1 = match.group(1).strip()
                    part1_clean = re.sub(r'\b(the|all|any|some)\b', '', part1, flags=re.IGNORECASE).strip()

                    intent['location'] = part1_clean
                    intent['has_location'] = True
                    intent['query_type'] = 'location_only'
                    logger.info(f"Location-only query detected - Location: {intent['location']}")
                    return intent

                # Extract both parts for combined patterns
                part1 = match.group(1).strip()
                part2 = match.group(2).strip() if match.lastindex >= 2 else None

                if not part2:
                    continue

                # Clean up common words
                part1_clean = re.sub(r'\b(the|all|any|some)\b', '', part1, flags=re.IGNORECASE).strip()
                part2_clean = re.sub(r'\b(the|all|any|some)\b', '', part2, flags=re.IGNORECASE).strip()

                # Smart detection: Check which part is a known location
                part1_is_location = part1_clean.lower() in known_locations
                part2_is_location = part2_clean.lower() in known_locations

                # Check if parts contain location-related words
                part1_has_location_words = any(loc in part1_clean.lower() for loc in known_locations)
                part2_has_location_words = any(loc in part2_clean.lower() for loc in known_locations)

                logger.info(f"Pattern {pattern_idx}: part1='{part1_clean}' (is_loc={part1_is_location}), part2='{part2_clean}' (is_loc={part2_is_location})")

                # Determine location and skill based on detection
                if part2_is_location or part2_has_location_words:
                    # part2 is location, part1 is skill
                    intent['location'] = part2_clean
                    intent['skills'] = self._extract_skills_from_text(part1_clean)
                    logger.info(f"Detected: Location={part2_clean}, Skills from '{part1_clean}'")
                elif part1_is_location or part1_has_location_words:
                    # part1 is location, part2 is skill
                    intent['location'] = part1_clean
                    intent['skills'] = self._extract_skills_from_text(part2_clean)
                    logger.info(f"Detected: Location={part1_clean}, Skills from '{part2_clean}'")
                else:
                    # Use pattern-specific logic
                    if pattern_idx == 0:  # "jobs in [location] on [skill]"
                        intent['location'] = part1_clean
                        intent['skills'] = self._extract_skills_from_text(part2_clean)
                    else:  # "[skill] jobs in [location]" or "show me [skill] in [location]"
                        intent['skills'] = self._extract_skills_from_text(part1_clean)
                        intent['location'] = part2_clean

                # Validate we have both location and skills
                if intent['location'] or intent['skills']:
                    intent['has_location'] = bool(intent['location'])
                    intent['has_skills'] = bool(intent['skills'])
                    intent['query_type'] = 'combined' if (intent['has_location'] and intent['has_skills']) else ('location_only' if intent['has_location'] else 'skill_only')
                    logger.info(f"Query detected - Location: {intent['location']}, Skills: {intent['skills']}, Type: {intent['query_type']}")
                    return intent

        # If no combined pattern matched, extract separately
        logger.info("No combined pattern matched, extracting separately")
        intent['location'] = self._extract_location_from_message(message)
        intent['skills'] = self._extract_skills_from_text(message)
        intent['job_type'] = self._extract_job_type(message)

        intent['has_location'] = bool(intent['location'])
        intent['has_skills'] = bool(intent['skills'])

        if intent['has_location'] and intent['has_skills']:
            intent['query_type'] = 'location_skill'
        elif intent['has_location']:
            intent['query_type'] = 'location_only'
        elif intent['has_skills']:
            intent['query_type'] = 'skill_only'

        logger.info(f"Separate extraction - Location: {intent['location']}, Skills: {intent['skills']}, Type: {intent['query_type']}")
        return intent

    def _is_location_query(self, message: str) -> bool:
        """Detect if message is asking for location-based job search (including skill+location combos)"""
        message_lower = message.lower()

        # Comprehensive location patterns
        location_patterns = [
            r'\b(?:jobs?|openings?|vacancies?|positions?)\s+(?:in|at|for|near|from)\s+\w+',
            r'\b(?:show|find|get|give|search)\s+(?:me\s+)?(?:all\s+)?(?:jobs?|openings?|vacancies?)\s+(?:in|for|at)\s+\w+',
            r'\w+\s+(?:jobs?|openings?|positions?|vacancies?)\s+(?:in|at|for|near)\s+\w+',  # "Data Entry jobs in Mumbai"
            r'\ball\s+(?:the\s+)?(?:jobs?|openings?|vacancies?)\s+(?:in|for|at)\s+\w+',
            r'\b(?:jobs?|positions?)\s+(?:in|at|for)\s+[a-zA-Z\s]+\s+(?:on|for|in)\s+\w+',  # "jobs in Mumbai on Data Entry"
        ]

        for pattern in location_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.info(f"Location query detected by pattern: {pattern}")
                return True

        # Check for known locations in message
        known_locations = [
            'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'hyderabad',
            'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'patna', 'vadodara', 'ghaziabad',
            'noida', 'gurgaon', 'gurugram', 'faridabad', 'coimbatore', 'kochi',
            'visakhapatnam', 'ludhiana', 'agra', 'nashik', 'meerut', 'rajkot',
            'maharashtra', 'karnataka', 'tamil nadu', 'telangana', 'gujarat',
            'rajasthan', 'uttar pradesh', 'madhya pradesh', 'west bengal', 'kerala'
        ]

        # Check if message contains job keywords + location
        job_keywords = ['job', 'jobs', 'opening', 'openings', 'vacancy', 'vacancies',
                       'position', 'positions', 'work', 'employment', 'career', 'opportunity']

        has_job_keyword = any(keyword in message_lower for keyword in job_keywords)
        has_location = any(location in message_lower for location in known_locations)

        if has_job_keyword and has_location:
            logger.info(f"Location query detected: has job keyword and location")
            return True

        # Check for "in" + location patterns
        location_indicators = [' in ', ' at ', ' for ', ' near ', ' from ', ' around ']
        if has_job_keyword and any(indicator in message_lower for indicator in location_indicators):
            logger.info(f"Location query detected: has job keyword and location indicator")
            return True

        return False

    async def _handle_location_job_query(self, request: ChatRequest, query_intent: Dict[str, Any] = None) -> ChatResponse:
        """Handle location-based job queries with enhanced skill detection"""
        message = request.message.lower().strip()

        try:
            # Use parsed intent if available, otherwise extract
            if query_intent:
                location = query_intent.get('location')
                skills = query_intent.get('skills', [])
                job_type = query_intent.get('job_type')
            else:
                location = self._extract_location_from_message(message)
                skills = self._extract_skills_from_text(message)
                job_type = self._extract_job_type(message)

            logger.info(f"Extracted Location: {location}, Skills: {skills}, Job Type: {job_type}")

            if not location:
                return ChatResponse(
                    response="I'd be happy to help you find jobs by location! Please specify a city or state. For example:\n\n- 'Jobs in Mumbai'\n- 'Show openings in Karnataka'\n- 'IT positions in Delhi'\n- 'All jobs in Bangalore'",
                    message_type="text",
                    chat_phase="job_searching",
                    suggestions=["Jobs in Mumbai", "IT jobs in Bangalore", "Remote positions", "Entry level jobs in Delhi"]
                )

            # Extract additional parameters if not already in query_intent
            experience_range = self._extract_experience_range(message)
            salary_range = self._extract_salary_range(message)

            limit = 50
            if any(word in message for word in ['all', 'every', 'complete', 'full']):
                limit = 100
            elif any(word in message for word in ['few', 'some', 'top']):
                limit = 20

            location_request = LocationJobRequest(
                location=location,
                job_type=job_type,
                skills=skills,
                experience_range=experience_range,
                salary_range=salary_range,
                limit=limit,
                sort_by="relevance"
            )

            search_response = await self.location_job_service.search_jobs_by_location(location_request)

            if search_response.jobs:
                response_text = self._format_location_success_response(search_response, skills)

                return ChatResponse(
                    response=response_text,
                    message_type="job_results",
                    chat_phase="job_results",
                    jobs=self._convert_to_chat_job_format(search_response.jobs[:10]),
                    location_searched=search_response.location_searched,
                    location_matches=search_response.location_matches,
                    total_found=search_response.total_found,
                    filters_applied=search_response.filters_applied,
                    search_context=search_response.search_context,
                    suggestions=self._get_location_followup_suggestions(search_response)
                )
            else:
                response_text = self._format_location_no_results_response(search_response)

                return ChatResponse(
                    response=response_text,
                    message_type="text",
                    chat_phase="job_searching",
                    location_searched=search_response.location_searched,
                    location_matches=search_response.location_matches,
                    suggestions=self._get_location_alternative_suggestions(location)
                )

        except Exception as e:
            logger.error(f"Location job query failed: {e}")
            return ChatResponse(
                response=f"I had trouble searching for jobs in that location. Could you try rephrasing? For example: 'Show me jobs in Mumbai' or 'IT positions in Bangalore'",
                message_type="text",
                chat_phase="job_searching",
                suggestions=["Jobs in Mumbai", "IT jobs in Bangalore", "Remote work", "Entry level positions"]
            )

    async def _handle_skill_job_query(self, request: ChatRequest, query_intent: Dict[str, Any] = None) -> ChatResponse:
        """Handle skill-only job queries using vector embeddings and GPT ranking"""
        message = request.message.lower().strip()
        user_id = id(request)

        # Import services here to avoid circular imports
        from services.embedding_service import LocalEmbeddingService
        from services.vector_store import FAISSVectorStore
        from services.gpt_service import GPTService
        from db.job_repository import get_complete_job_details

        try:
            # Use parsed intent if available, otherwise extract
            if query_intent and query_intent.get('skills'):
                skills = query_intent.get('skills', [])
            else:
                skills = self._extract_skills_from_text(message)

            logger.info(f"Skill-only search - Extracted Skills: {skills}")

            if not skills:
                return ChatResponse(
                    response="I'd be happy to help you find jobs by skill! Please specify the skills you're looking for. For example:\n\n- 'Show me Python jobs'\n- 'Find React developer positions'\n- 'Data Entry jobs'\n- 'Customer Service openings'",
                    message_type="text",
                    chat_phase="job_searching",
                    suggestions=["Python jobs", "React developer", "Data Entry", "Customer Service"]
                )

            # Extract additional parameters
            experience_range = self._extract_experience_range(message)
            salary_range = self._extract_salary_range(message)

            # Determine result limit
            limit = 50
            if any(word in message for word in ['all', 'every', 'complete', 'full']):
                limit = 100
            elif any(word in message for word in ['few', 'some', 'top']):
                limit = 20

            # Note: For now, return a placeholder response
            # The actual implementation requires embedding_service and vector_store instances
            skills_formatted = ', '.join(skills)
            return ChatResponse(
                response=f"I found your skills: **{skills_formatted}**. To search for jobs matching these skills, please add a location. For example: '{skills[0]} jobs in Mumbai'",
                message_type="text",
                chat_phase="profile_building",
                profile_data={"skills": skills},
                suggestions=[
                    f"{skills[0]} jobs in Mumbai",
                    f"{skills[0]} jobs in Bangalore",
                    "Add location",
                    "Upload CV"
                ]
            )

        except Exception as e:
            logger.error(f"Skill job query handler failed: {e}")
            return ChatResponse(
                response="I had trouble searching for jobs by skill. Could you try rephrasing? For example:\n- 'Show me Python jobs'\n- 'Find Data Entry positions'\n- 'JavaScript developer openings'",
                message_type="text",
                chat_phase="job_searching",
                suggestions=["Python jobs", "Data Entry", "JavaScript developer", "Jobs in Mumbai"]
            )

    async def _handle_regular_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle regular chat using existing logic with improved fallback handling"""
        message = request.message.lower().strip()
        chat_phase = request.chat_phase
        user_profile = request.user_profile or {}
        user_id = id(request)

        try:
            if chat_phase == "intro":
                if any(word in message for word in ["upload", "cv", "resume", "file"]):
                    self.fallback_count[user_id] = 0  # Reset on valid interaction
                    return ChatResponse(
                        response="Great! Please click the paperclip icon to upload your CV. I support PDF, DOC, and DOCX files.",
                        message_type="text",
                        chat_phase="intro"
                    )
                elif any(word in message for word in ["chat", "talk", "build", "skills", "hello", "hi", "hey"]):
                    self.fallback_count[user_id] = 0  # Reset on valid interaction
                    return ChatResponse(
                        response="Perfect! Let's build your profile together. What are your main skills? (e.g., Python, React, Data Entry, Customer Service, etc.)\n\nYou can also ask about jobs in specific locations like 'Jobs in Mumbai'.",
                        message_type="text",
                        chat_phase="profile_building"
                    )
                else:
                    return ChatResponse(
                        response="I can help you find jobs in multiple ways:\n\n1. Upload your CV - I'll analyze it automatically\n2. Chat with me - I'll ask about your skills\n3. Ask about specific locations - 'Jobs in Mumbai'\n4. Combined search - 'Data Entry jobs in Mumbai'\n\nWhich would you prefer?",
                        message_type="text",
                        chat_phase="intro",
                        suggestions=["Upload CV", "Tell me your skills", "Jobs in Mumbai", "Data Entry jobs in Mumbai"]
                    )

            elif chat_phase == "profile_building":
                skills = self._extract_skills_from_text(message)

                if skills:
                    self.fallback_count[user_id] = 0  # Reset on valid skill extraction
                    return ChatResponse(
                        response=f"I understand your skills: {', '.join(skills)}. Would you like to search in a specific location? You can ask 'Show me {skills[0]} jobs in Mumbai' for example.",
                        message_type="text",
                        chat_phase="profile_building",
                        profile_data={"skills": skills},
                        suggestions=[f"{skills[0]} jobs in Mumbai", "Jobs in Bangalore",]
                    )
                else:
                    return ChatResponse(
                        response="I'd like to help you find jobs. Please tell me your skills. For example: 'I know Python and React' or 'I can do Data Entry and Customer Service'\n\nOr ask about jobs in specific locations like 'Jobs in Mumbai'.",
                        message_type="text",
                        chat_phase="profile_building",
                        suggestions=["I know Python", "Data Entry skills", "Jobs in Mumbai", "Customer Service"]
                    )

            else:
                return ChatResponse(
                    response="I can help you find more jobs or search in specific locations. What would you like to do?",
                    message_type="text",
                    chat_phase="job_searching",
                    suggestions=["Show more jobs", "Jobs in Mumbai", "Different skills", "Remote work"]
                )

        except Exception as e:
            logger.error(f"Regular chat error: {e}")
            # Track fallback to prevent loops
            self.fallback_count[user_id] = self.fallback_count.get(user_id, 0) + 1

            if self.fallback_count.get(user_id, 0) > 2:
                return ChatResponse(
                    response="Let's try a different approach! Here are specific examples:\n\n- 'Show me Python jobs in Mumbai'\n- 'I have Data Entry skills'\n- 'Customer Service positions in Delhi'\n- Or upload your CV for automatic matching",
                    message_type="text",
                    chat_phase="intro",
                    suggestions=["Python jobs in Mumbai", "Data Entry skills", "Upload CV", "Customer Service in Delhi"]
                )

            return ChatResponse(
                response="Let me help you find jobs. What skills do you have? Or ask me about jobs in specific locations like 'Mumbai jobs'.",
                message_type="text",
                chat_phase="profile_building",
                suggestions=["My skills are...", "Jobs in Mumbai", "Remote work", "Entry level"]
            )

    # =========================================================================
    # HELPER METHODS FOR LOCATION PROCESSING
    # =========================================================================

    def _extract_location_from_message(self, message: str) -> Optional[str]:
        """Extract location from chat message"""
        location_patterns = [
            r'\b(?:jobs?\s+in|openings?\s+in|vacancies?\s+in|positions?\s+in)\s+([a-zA-Z\s]+)',
            r'\b(?:show|find|get|give)\s+(?:me\s+)?(?:all\s+)?(?:jobs?|openings?|vacancies?|positions?)\s+(?:in|for|at)\s+([a-zA-Z\s]+)',
            r'\b([a-zA-Z\s]+)\s+(?:jobs?|openings?|vacancies?|positions?)',
            r'\blocation[:\s]+([a-zA-Z\s]+)',
            r'\bin\s+([a-zA-Z\s]+)(?:\s+city|\s+state|\s+region)?'
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                location = matches[0].strip()
                location = re.sub(r'\b(for|all|any|the|in|at|city|state|region|area|jobs?|openings?|vacancies?|positions?)\b', '', location, flags=re.IGNORECASE).strip()
                if location and len(location) > 1:
                    return location

        major_cities = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'hyderabad', 'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur']
        message_words = message.split()

        for word in message_words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in major_cities:
                return clean_word

        return None

    def _extract_skills_from_text(self, message: str) -> List[str]:
        """Extract skills using comprehensive keyword matching with smart conflict resolution"""
        skills = []
        message_lower = message.lower()
        matched_skills = set()  # Track already matched skills

        # Sort skills by variation length (longest first) to avoid false positives
        sorted_skills = sorted(
            self.skill_keywords.items(),
            key=lambda x: max(len(v) for v in x[1]),
            reverse=True
        )

        for main_skill, variations in sorted_skills:
            if main_skill in matched_skills:
                continue

            for variation in sorted(variations, key=len, reverse=True):
                if ' ' not in variation:  # Single word
                    pattern = r'\b' + re.escape(variation) + r'\b'
                    if re.search(pattern, message_lower):
                        if main_skill == 'c++':
                            skills.append('C++')
                        elif main_skill == 'c#':
                            skills.append('C#')
                        elif main_skill == 'javascript':
                            skills.append('JavaScript')
                            matched_skills.add('java')
                        elif main_skill == 'typescript':
                            skills.append('TypeScript')
                        elif main_skill == 'machine learning':
                            skills.append('Machine Learning')
                        elif main_skill == 'data science':
                            skills.append('Data Science')
                        elif main_skill == 'data entry':
                            skills.append('Data Entry')
                        elif main_skill == 'voice process':
                            skills.append('Voice Process')
                        elif main_skill == 'ui/ux':
                            skills.append('UI/UX')
                        else:
                            skills.append(main_skill.title())
                        matched_skills.add(main_skill)
                        break
                else:
                    if variation in message_lower:
                        if main_skill == 'machine learning':
                            skills.append('Machine Learning')
                        elif main_skill == 'data science':
                            skills.append('Data Science')
                        elif main_skill == 'data entry':
                            skills.append('Data Entry')
                        elif main_skill == 'voice process':
                            skills.append('Voice Process')
                        elif main_skill == 'customer service':
                            skills.append('Customer Service')
                        else:
                            skills.append(main_skill.title())
                        matched_skills.add(main_skill)
                        break

        # Experience pattern matching
        experience_patterns = {
            'years': r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            'months': r'(\d+)\s*(?:months?|mos?)\s*(?:of\s*)?(?:experience|exp)',
            'fresher': r'\b(?:fresher|fresh|new|entry\s*level|no\s*experience)\b'
        }

        for pattern_name, pattern in experience_patterns.items():
            matches = re.findall(pattern, message_lower)
            if matches:
                if pattern_name == 'years' and matches:
                    skills.append(f"{matches[0]} Years Experience")
                elif pattern_name == 'months' and matches:
                    skills.append(f"{matches[0]} Months Experience")
                elif pattern_name == 'fresher':
                    skills.append("Fresher")
                break

        return list(dict.fromkeys(skills))

    def _extract_job_type(self, message: str) -> Optional[str]:
        """Extract job type from message"""
        job_type_patterns = {
            'software': ['software', 'developer', 'programming', 'coding'],
            'it': ['it', 'information technology', 'tech'],
            'sales': ['sales', 'selling'],
            'marketing': ['marketing', 'digital marketing'],
            'data entry': ['data entry', 'typing'],
            'customer service': ['customer service', 'call center', 'support'],
            'finance': ['finance', 'financial', 'accounting'],
            'hr': ['hr', 'human resources', 'recruitment'],
            'healthcare': ['healthcare', 'medical', 'nursing'],
            'education': ['education', 'teaching', 'training']
        }

        for job_type, keywords in job_type_patterns.items():
            if any(keyword in message for keyword in keywords):
                return job_type

        return None

    def _extract_experience_range(self, message: str) -> Optional[Tuple[float, float]]:
        """Extract experience range from message"""
        if re.search(r'\bfresh(?:er)?|entry\s*level|no\s*experience\b', message, re.IGNORECASE):
            return (0, 2)

        range_match = re.search(r'\b(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)\b', message, re.IGNORECASE)
        if range_match:
            return (float(range_match.group(1)), float(range_match.group(2)))

        min_match = re.search(r'\b(?:minimum|min|at least)\s*(\d+)\s*(?:years?|yrs?)\b', message, re.IGNORECASE)
        if min_match:
            return (float(min_match.group(1)), 50)

        max_match = re.search(r'\b(?:maximum|max|up to)\s*(\d+)\s*(?:years?|yrs?)\b', message, re.IGNORECASE)
        if max_match:
            return (0, float(max_match.group(1)))

        exp_match = re.search(r'\b(\d+)(?:\+|\s*plus)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b', message, re.IGNORECASE)
        if exp_match:
            years = float(exp_match.group(1))
            if '+' in exp_match.group(0) or 'plus' in exp_match.group(0).lower():
                return (years, 50)
            else:
                return (max(0, years-1), years+2)

        return None

    def _extract_salary_range(self, message: str) -> Optional[Tuple[float, float]]:
        """Extract salary range from message"""
        def convert_salary(amount_str: str) -> float:
            amount = float(re.sub(r'[^\d.]', '', amount_str))
            if 'lakh' in message or 'l' in amount_str.lower():
                return amount * 100000
            elif 'k' in amount_str.lower():
                return amount * 1000
            return amount

        range_match = re.search(r'\b(?:rs\.?|₹)\s*(\d+(?:k|lakh|l)?)\s*(?:to|-)\s*(?:rs\.?|₹)?\s*(\d+(?:k|lakh|l)?)\b', message, re.IGNORECASE)
        if range_match:
            min_sal = convert_salary(range_match.group(1))
            max_sal = convert_salary(range_match.group(2))
            return (min_sal, max_sal)

        min_match = re.search(r'\b(?:salary|pay|wage)\s*(?:above|over|more than|>)\s*(?:rs\.?|₹)?\s*(\d+(?:k|lakh|l)?)\b', message, re.IGNORECASE)
        if min_match:
            min_sal = convert_salary(min_match.group(1))
            return (min_sal, 10000000)

        max_match = re.search(r'\b(?:salary|pay|wage)\s*(?:below|under|less than|<)\s*(?:rs\.?|₹)?\s*(\d+(?:k|lakh|l)?)\b', message, re.IGNORECASE)
        if max_match:
            max_sal = convert_salary(max_match.group(1))
            return (0, max_sal)

        return None

    # =========================================================================
    # RESPONSE FORMATTING METHODS
    # =========================================================================

    def _format_location_success_response(self, search_response, skills: List[str] = None) -> str:
        """Format successful location search response"""
        location = search_response.location_searched
        total = search_response.total_found
        returned = search_response.returned_count

        if skills:
            response_parts = [f"Found {total} {', '.join(skills)} job openings in {location}!\n"]
        else:
            response_parts = [f"Found {total} job openings in {location}!\n"]

        if search_response.location_matches:
            locations = []
            if search_response.location_matches.get("states"):
                locations.extend([f"State: {s}" for s in search_response.location_matches["states"]])
            if search_response.location_matches.get("districts"):
                locations.extend([f"City: {d}" for d in search_response.location_matches["districts"]])

            if locations:
                response_parts.append(f"Locations: {' | '.join(locations)}\n")

        filters = search_response.filters_applied
        filter_info = []

        if filters.get("job_type"):
            filter_info.append(f"Type: {filters['job_type'].title()}")
        if filters.get("skills"):
            filter_info.append(f"Skills: {', '.join(filters['skills'])}")
        if filters.get("experience_range"):
            min_exp, max_exp = filters["experience_range"]
            if max_exp == 50:
                filter_info.append(f"Experience: {min_exp}+ years")
            else:
                filter_info.append(f"Experience: {min_exp}-{max_exp} years")
        if filters.get("salary_range"):
            min_sal, max_sal = filters["salary_range"]
            if max_sal >= 10000000:
                filter_info.append(f"Salary: {self._format_salary(min_sal)}+")
            else:
                filter_info.append(f"Salary: {self._format_salary(min_sal)}-{self._format_salary(max_sal)}")

        if filter_info:
            response_parts.append(f"Filters: {' | '.join(filter_info)}\n")

        response_parts.append(f"Results: Showing top {returned} opportunities")

        if search_response.processing_time_ms:
            response_parts.append(f" (processed in {search_response.processing_time_ms}ms)")

        return "\n".join(response_parts)

    def _format_location_no_results_response(self, search_response) -> str:
        """Format no results response"""
        location = search_response.location_searched

        response_parts = [f"No jobs found for {location}\n"]

        if search_response.location_matches:
            matched_locations = []
            if search_response.location_matches.get("states"):
                matched_locations.extend(search_response.location_matches["states"])
            if search_response.location_matches.get("districts"):
                matched_locations.extend(search_response.location_matches["districts"])

            if matched_locations:
                response_parts.append(f"I searched in: {', '.join(matched_locations)}\n")
            else:
                response_parts.append(f"Location '{location}' might not be in our database.\n")

        response_parts.extend([
            "Try these alternatives:",
            "- Search in nearby cities or states",
            "- Remove specific skill requirements",
            "- Try broader job categories",
            "- Check for remote work opportunities"
        ])

        return "\n".join(response_parts)

    def _convert_to_chat_job_format(self, location_jobs: List[Dict]) -> List[Dict]:
        """Convert location job results to chat job format"""
        chat_jobs = []

        for job in location_jobs:
            chat_job = {
                "ncspjobid": job["ncspjobid"],
                "title": job["title"],
                "organization_name": job["organization_name"],
                "match_percentage": job.get("match_percentage", 75),
                "statename": job["statename"],
                "districtname": job["districtname"],
                "avewage": job["avewage"],
                "aveexp": job["aveexp"],
                "functionalrolename": job.get("functionalrolename"),
                "industryname": job.get("industryname"),
                "keywords": job.get("keywords"),
                "skills_matched": job.get("skills_matched", [])
            }
            chat_jobs.append(chat_job)

        return chat_jobs

    def _get_location_followup_suggestions(self, search_response) -> List[str]:
        """Get follow-up suggestions for location searches"""
        suggestions = []

        if search_response.total_found > search_response.returned_count:
            suggestions.append("Show more jobs")

        if search_response.location_matches.get("states"):
            for state in search_response.location_matches["states"][:1]:
                if state != search_response.location_searched:
                    suggestions.append(f"Jobs in other cities of {state}")

        suggestions.extend([
            "Filter by salary range",
            "Filter by experience level",
            "Show remote jobs"
        ])

        return suggestions[:4]

    def _get_location_alternative_suggestions(self, location: str) -> List[str]:
        """Get alternative suggestions when no results found"""
        suggestions = [
            f"Remote jobs (work from {location})",
            "Jobs in nearby cities",
            "Entry level positions",
            "Browse all locations"
        ]

        major_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]
        if location.title() not in major_cities:
            suggestions.append(f"Jobs in {major_cities[0]}")

        return suggestions[:4]

    def _format_salary(self, amount: float) -> str:
        """Format salary amount for display"""
        if amount >= 100000:
            return f"{amount/100000:.1f}L"
        elif amount >= 1000:
            return f"{int(amount/1000)}K"
        else:
            return str(int(amount))

    async def handle_cv_followup_chat(self, request: ChatRequest, cv_profile) -> ChatResponse:
        """Handle follow-up conversations after CV analysis with location awareness"""
        message = request.message.lower().strip()

        try:
            # Check if this is a location-based follow-up
            if self._is_location_query(message):
                return await self._handle_cv_location_followup(request, cv_profile)

            # Handle more jobs request
            if any(word in message for word in ["show more jobs", "show more", "additional", "other jobs"]):
                return await self._handle_cv_more_jobs(cv_profile)

            # Handle skill addition
            elif any(word in message for word in ["add skill", "more skill", "also know", "i can", "i have experience"]):
                return await self._handle_cv_skill_addition(request, cv_profile)

            # Default response
            else:
                return ChatResponse(
                    response="I can help you with:\n- Show more job opportunities\n- Add skills to your profile\n- Search by location\n- Start a new search\n\nWhat would you like to do?",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Show more jobs", "Add skills", "Search by location", "Jobs in Mumbai"]
                )

        except Exception as e:
            logger.error(f"CV followup chat failed: {e}")
            return ChatResponse(
                response="I can help you find more jobs or refine your search. What would you like to do?",
                message_type="text",
                chat_phase="job_results"
            )

    async def _handle_cv_location_followup(self, request: ChatRequest, cv_profile) -> ChatResponse:
        """Handle location-based queries after CV analysis"""
        message = request.message.lower().strip()
        location = self._extract_location_from_message(message)

        if not location:
            return ChatResponse(
                response="Which location would you like me to search in? For example: 'Jobs in Mumbai' or 'Show me positions in Bangalore'",
                message_type="text",
                chat_phase="job_results",
                suggestions=["Jobs in Mumbai", "Positions in Delhi", "Bangalore opportunities", "Remote work"]
            )

        try:
            location_request = LocationJobRequest(
                location=location,
                skills=cv_profile.skills[:10] if hasattr(cv_profile, 'skills') else [],
                limit=20,
                sort_by="relevance"
            )

            search_response = await self.location_job_service.search_jobs_by_location(location_request)

            if search_response.jobs:
                response_text = f"Found {search_response.total_found} jobs in {location} matching your CV skills!\n\nLocation: {location}\nSkills from CV: {', '.join(cv_profile.skills[:5]) if hasattr(cv_profile, 'skills') else 'Various skills'}\nResults: Showing top {search_response.returned_count} opportunities"

                return ChatResponse(
                    response=response_text,
                    message_type="job_results",
                    chat_phase="job_results",
                    jobs=self._convert_to_chat_job_format(search_response.jobs[:8]),
                    location_searched=search_response.location_searched,
                    location_matches=search_response.location_matches,
                    total_found=search_response.total_found,
                    suggestions=["Show more jobs", f"Other cities in {location}", "Different location", "Salary filter"]
                )
            else:
                return ChatResponse(
                    response=f"No jobs found in {location} with your current skills. Try:\n- Different location nearby\n- Broader skill categories\n- Remote opportunities",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Remote jobs", "Jobs in Mumbai", "Nearby cities", "Different skills"]
                )

        except Exception as e:
            logger.error(f"CV location followup failed: {e}")
            return ChatResponse(
                response="I had trouble searching in that location. Try asking about jobs in major cities like Mumbai, Delhi, or Bangalore.",
                message_type="text",
                chat_phase="job_results",
                suggestions=["Jobs in Mumbai", "Delhi positions", "Bangalore jobs", "Remote work"]
            )

    async def _handle_cv_more_jobs(self, cv_profile) -> ChatResponse:
        """Handle request for more jobs based on CV"""
        return ChatResponse(
            response="I'll search for more opportunities. What specific type of jobs are you most interested in?",
            message_type="text",
            chat_phase="job_results",
            suggestions=["Software jobs", "Remote work", "Entry level", "Senior positions"]
        )

    async def _handle_cv_skill_addition(self, request: ChatRequest, cv_profile) -> ChatResponse:
        """Handle adding skills to CV profile"""
        message = request.message
        additional_skills = self._extract_skills_from_text(message)

        if not additional_skills:
            return ChatResponse(
                response="What additional skills would you like to add to your profile? For example: 'I also know Data Entry and Voice Process' or 'I have experience in Customer Service'",
                message_type="text",
                chat_phase="profile_refinement",
                suggestions=["I also know Python", "Customer service experience", "Data entry skills", "Sales experience"]
            )

        return ChatResponse(
            response=f"I noted your additional skills: {', '.join(additional_skills)}. Would you like to search for jobs with these skills in a specific location?",
            message_type="text",
            chat_phase="profile_refinement",
            profile_data={"skills": additional_skills},
            suggestions=[f"{additional_skills[0]} jobs in Mumbai", "Jobs in Bangalore", "Search all locations", "Add more skills"]
        )


class CVChatService:
    """Enhanced chat service specifically for CV upload interactions"""

    @staticmethod
    async def handle_cv_upload_chat(cv_profile) -> ChatResponse:
        """Handle chat after CV upload with enhanced profile data"""
        try:
            if cv_profile.skills and len(cv_profile.skills) >= 3:
                profile_summary = {
                    "name": cv_profile.name,
                    "skills": cv_profile.skills[:10],
                    "experience_count": len(cv_profile.experience),
                    "confidence": cv_profile.confidence_score
                }

                return ChatResponse(
                    response=f"Perfect! I've analyzed your CV and found your skills: {', '.join(cv_profile.skills[:5])}{'...' if len(cv_profile.skills) > 5 else ''}. To find matching jobs, please specify a location like 'Jobs in Mumbai'.",
                    message_type="cv_results",
                    chat_phase="job_results",
                    profile_data=profile_summary,
                    suggestions=[f"{cv_profile.skills[0]} jobs in Mumbai", "Jobs in Bangalore", "Remote positions", "Show all jobs"]
                )
            else:
                return ChatResponse(
                    response="I've processed your CV but found limited technical skills. Let's chat to build a complete profile for better job matching.",
                    message_type="text",
                    chat_phase="profile_building"
                )

        except Exception as e:
            logger.error(f"CV chat integration failed: {e}")
            return ChatResponse(
                response="I've processed your CV! Let's discuss your skills to find the best job matches.",
                message_type="text",
                chat_phase="profile_building"
            )

    @staticmethod
    async def handle_cv_followup_chat(request: ChatRequest, cv_profile) -> ChatResponse:
        """Handle follow-up chat after CV analysis"""
        message = request.message.lower().strip()

        try:
            if any(word in message for word in ["more jobs", "show more", "additional", "other jobs"]):
                return ChatResponse(
                    response="I can search for more jobs. What location are you interested in? Try 'Jobs in Mumbai' or 'Bangalore positions'.",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Jobs in Mumbai", "Bangalore positions", "Remote work", "Delhi jobs"]
                )

            elif any(word in message for word in ["add skill", "more skill", "also know", "i can"]):
                return ChatResponse(
                    response="What additional skills would you like to add to your profile? For example: 'I also know Docker and AWS'",
                    message_type="text",
                    chat_phase="profile_refinement"
                )

            else:
                return ChatResponse(
                    response="I can help you with:\n- Show more job opportunities\n- Add skills to your profile\n- Search by location\n- Start a new search\n\nWhat would you like to do?",
                    message_type="text",
                    chat_phase="job_results",
                    suggestions=["Show more jobs", "Add skills", "Search by location", "Start over"]
                )

        except Exception as e:
            logger.error(f"CV followup chat failed: {e}")
            return ChatResponse(
                response="I can help you find more jobs or refine your search. What would you like to do?",
                message_type="text",
                chat_phase="job_results"
            )
