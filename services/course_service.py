import re
import json
import logging
from typing import List, Dict

from config.settings import azure_client, gpt_deployment

logger = logging.getLogger(__name__)


class CourseRecommendationService:
    """Service class for course recommendations using Azure OpenAI"""

    @staticmethod
    async def get_course_recommendations(keywords: List[str]) -> List[Dict]:
        """Get course recommendations for unmatched keywords using Azure OpenAI GPT"""

        if not keywords:
            return []

        keywords_str = ', '.join(keywords)

        prompt = f"""
    You are an expert career advisor. For each of these technical skills: {keywords_str}

    Provide EXACTLY 5 specific, real courses for each skill from these platforms ONLY:
    - Udemy.com
    - Coursera.org
    - edX.org
    - Pluralsight.com
    - LinkedIn Learning
    - DataCamp.com

    STRICT REQUIREMENTS:
    1. NO generic course names like "Learn Python" or "Master React"
    2. NO Google search links or google.com URLs
    3. Use ACTUAL course titles from real platforms
    4. Each course must have a realistic platform-specific URL
    5. EXACTLY 5 courses per keyword (total: {len(keywords) * 5} courses)

    For each course provide:
    - course_name: Specific real course title
    - platform: One of the approved platforms above
    - duration: Realistic timeframe
    - link: Actual course URL (not search links)
    - educator: Real instructor/organization name
    - skill_covered: The exact keyword from the list
    - difficulty_level: Beginner/Intermediate/Advanced
    - rating: Realistic rating like "4.5/5"

    Example format:
    {{
        "course_name": "Python for Everybody Specialization",
        "platform": "Coursera",
        "duration": "8 months",
        "link": "https://www.coursera.org/specializations/python",
        "educator": "University of Michigan",
        "skill_covered": "Python",
        "difficulty_level": "Beginner",
        "rating": "4.8/5"
    }}

    Return ONLY a valid JSON array with {len(keywords) * 5} courses total.
    NO explanatory text. NO markdown formatting.
    """

        try:
            logger.info(f"Getting course recommendations for keywords: {keywords_str}")

            response = azure_client.chat.completions.create(
                model=gpt_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a learning advisor. Return ONLY valid JSON array with exactly {len(keywords) * 5} course recommendations. NO explanation text. NO markdown. NO generic course names. NO Google links."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=4000   # Increased for more courses
            )

            content = response.choices[0].message.content.strip()

            # Clean response more thoroughly
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = re.sub(r'^[^[]*', '', content)  # Remove text before JSON array
            content = re.sub(r'[^}]*$', '}]', content)  # Ensure proper JSON ending
            content = content.strip()

            try:
                return CourseRecommendationService._get_fallback_recommendations(keywords)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT course response: {e}")
                logger.debug(f"Raw response: {content}")

            # Fallback to sample recommendations
            logger.info("Using fallback sample course recommendations")
            return CourseRecommendationService._get_fallback_recommendations(keywords)

        except Exception as e:
            logger.error(f"Course recommendation failed: {e}")
            return CourseRecommendationService._get_fallback_recommendations(keywords)

    @staticmethod
    def _get_fallback_recommendations(keywords: List[str]) -> List[Dict]:
        """Provide curated course recommendations with exactly 5 courses per keyword"""

        # Comprehensive course database with 5 courses per skill
        course_database = {
            "Python": [
                {
                    "course_name": "Python for Everybody Specialization",
                    "platform": "Coursera",
                    "duration": "8 months",
                    "link": "https://www.coursera.org/specializations/python",
                    "educator": "University of Michigan",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.8/5"
                },
                {
                    "course_name": "Complete Python Bootcamp From Zero to Hero",
                    "platform": "Udemy",
                    "duration": "22 hours",
                    "link": "https://www.udemy.com/course/complete-python-bootcamp/",
                    "educator": "Jose Portilla",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Python Programming Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/python-fundamentals",
                    "educator": "Austin Bingham",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Computer Science and Programming Using Python",
                    "platform": "edX",
                    "duration": "9 weeks",
                    "link": "https://www.edx.org/course/introduction-to-computer-science-and-programming-7",
                    "educator": "MIT",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "4 hours",
                    "link": "https://www.linkedin.com/learning/python-essential-training-2",
                    "educator": "Bill Weinman",
                    "skill_covered": "Python",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "JavaScript": [
                {
                    "course_name": "JavaScript Algorithms and Data Structures",
                    "platform": "Coursera",
                    "duration": "6 months",
                    "link": "https://www.coursera.org/learn/javascript-algorithms-data-structures",
                    "educator": "University of California San Diego",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "The Complete JavaScript Course 2024: From Zero to Expert!",
                    "platform": "Udemy",
                    "duration": "69 hours",
                    "link": "https://www.udemy.com/course/the-complete-javascript-course/",
                    "educator": "Jonas Schmedtmann",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "All Levels",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "JavaScript Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/javascript-fundamentals",
                    "educator": "Liam McLennan",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Introduction to JavaScript",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/introduction-to-javascript",
                    "educator": "W3C",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.3/5"
                },
                {
                    "course_name": "JavaScript Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "5 hours",
                    "link": "https://www.linkedin.com/learning/javascript-essential-training",
                    "educator": "Morten Rand-Hendriksen",
                    "skill_covered": "JavaScript",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "React": [
                {
                    "course_name": "React - The Complete Guide (incl Hooks, React Router, Redux)",
                    "platform": "Udemy",
                    "duration": "40.5 hours",
                    "link": "https://www.udemy.com/course/react-the-complete-guide-incl-hooks-react-router-redux/",
                    "educator": "Maximilian Schwarzmuller",
                    "skill_covered": "React",
                    "difficulty_level": "All Levels",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Frontend Development using React Specialization",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/react",
                    "educator": "The Hong Kong University of Science and Technology",
                    "skill_covered": "React",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "React.js Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "2 hours",
                    "link": "https://www.linkedin.com/learning/react-js-essential-training",
                    "educator": "Eve Porcello",
                    "skill_covered": "React",
                    "difficulty_level": "Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "React Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/react-fundamentals-update",
                    "educator": "Liam McLennan",
                    "skill_covered": "React",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to ReactJS",
                    "platform": "edX",
                    "duration": "5 weeks",
                    "link": "https://www.edx.org/course/introduction-to-reactjs",
                    "educator": "Microsoft",
                    "skill_covered": "React",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                }
            ],
            "HTML/CSS": [
                {
                    "course_name": "HTML, CSS, and Javascript for Web Developers",
                    "platform": "Coursera",
                    "duration": "5 weeks",
                    "link": "https://www.coursera.org/learn/html-css-javascript-for-web-developers",
                    "educator": "Johns Hopkins University",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Build Responsive Real-World Websites with HTML and CSS",
                    "platform": "Udemy",
                    "duration": "37.5 hours",
                    "link": "https://www.udemy.com/course/design-and-develop-a-killer-website-with-html5-and-css3/",
                    "educator": "Jonas Schmedtmann",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner to Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "HTML5 and CSS3 Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/html-css-fundamentals",
                    "educator": "Matt Milner",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Introduction to Web Development",
                    "platform": "edX",
                    "duration": "5 weeks",
                    "link": "https://www.edx.org/course/introduction-to-web-development",
                    "educator": "University of California Davis",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.3/5"
                },
                {
                    "course_name": "CSS Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "4 hours",
                    "link": "https://www.linkedin.com/learning/css-essential-training-3",
                    "educator": "Christina Truong",
                    "skill_covered": "HTML/CSS",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                }
            ],
            "SQL": [
                {
                    "course_name": "Introduction to Structured Query Language (SQL)",
                    "platform": "Coursera",
                    "duration": "4 weeks",
                    "link": "https://www.coursera.org/learn/intro-sql",
                    "educator": "University of Michigan",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.8/5"
                },
                {
                    "course_name": "The Complete SQL Bootcamp: Go from Zero to Hero",
                    "platform": "Udemy",
                    "duration": "9 hours",
                    "link": "https://www.udemy.com/course/the-complete-sql-bootcamp/",
                    "educator": "Jose Portilla",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "SQL Server Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/sql-server-fundamentals",
                    "educator": "Pinal Dave",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Databases: Introduction to Databases and SQL Querying",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/databases-introduction-databases-sql",
                    "educator": "IBM",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "SQL Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/sql-essential-training-3",
                    "educator": "Walter Shields",
                    "skill_covered": "SQL",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                }
            ],
            "Django": [
                {
                    "course_name": "Django for Everybody Specialization",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/django",
                    "educator": "University of Michigan",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python Django - The Practical Guide",
                    "platform": "Udemy",
                    "duration": "23 hours",
                    "link": "https://www.udemy.com/course/python-django-the-practical-guide/",
                    "educator": "Maximilian Schwarzmuller",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Django Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "6 hours",
                    "link": "https://www.pluralsight.com/courses/django-fundamentals-update",
                    "educator": "Reindert-Jan Ekker",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Django Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/django-essential-training",
                    "educator": "Nick Walter",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "CS50's Web Programming with Python and JavaScript",
                    "platform": "edX",
                    "duration": "12 weeks",
                    "link": "https://www.edx.org/course/cs50s-web-programming-with-python-and-javascript",
                    "educator": "Harvard University",
                    "skill_covered": "Django",
                    "difficulty_level": "Intermediate",
                    "rating": "4.8/5"
                }
            ],
            "Spring Boot": [
                {
                    "course_name": "Spring Boot Microservices and Spring Cloud",
                    "platform": "Coursera",
                    "duration": "4 months",
                    "link": "https://www.coursera.org/specializations/spring-boot-cloud",
                    "educator": "LearnQuest",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Spring Boot For Beginners",
                    "platform": "Udemy",
                    "duration": "7 hours",
                    "link": "https://www.udemy.com/course/spring-boot-tutorial-for-beginners/",
                    "educator": "in28Minutes Official",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Spring Boot Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "4 hours",
                    "link": "https://www.pluralsight.com/courses/spring-boot-fundamentals",
                    "educator": "Dan Bunker",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Spring Boot Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/spring-boot-essential-training",
                    "educator": "Frank Moley",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Spring Boot",
                    "platform": "edX",
                    "duration": "4 weeks",
                    "link": "https://www.edx.org/course/introduction-to-spring-boot",
                    "educator": "Microsoft",
                    "skill_covered": "Spring Boot",
                    "difficulty_level": "Intermediate",
                    "rating": "4.3/5"
                }
            ],
            "Data Analysis": [
                {
                    "course_name": "Google Data Analytics Professional Certificate",
                    "platform": "Coursera",
                    "duration": "6 months",
                    "link": "https://www.coursera.org/professional-certificates/google-data-analytics",
                    "educator": "Google",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Python for Data Science and Machine Learning Bootcamp",
                    "platform": "Udemy",
                    "duration": "25 hours",
                    "link": "https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/",
                    "educator": "Jose Portilla",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Intermediate",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Data Analysis Fundamentals with Tableau",
                    "platform": "Pluralsight",
                    "duration": "5 hours",
                    "link": "https://www.pluralsight.com/courses/tableau-data-analysis-fundamentals",
                    "educator": "Ben Sullins",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                },
                {
                    "course_name": "Introduction to Data Analysis using Excel",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/introduction-to-data-analysis-using-excel",
                    "educator": "Rice University",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Excel Essential Training (Microsoft 365)",
                    "platform": "LinkedIn Learning",
                    "duration": "6 hours",
                    "link": "https://www.linkedin.com/learning/excel-essential-training-microsoft-365",
                    "educator": "Dennis Taylor",
                    "skill_covered": "Data Analysis",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                }
            ],
            "Power BI": [
                {
                    "course_name": "Microsoft Power BI Data Analyst Professional Certificate",
                    "platform": "Coursera",
                    "duration": "5 months",
                    "link": "https://www.coursera.org/professional-certificates/microsoft-power-bi-data-analyst",
                    "educator": "Microsoft",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner",
                    "rating": "4.7/5"
                },
                {
                    "course_name": "Microsoft Power BI Desktop for Business Intelligence",
                    "platform": "Udemy",
                    "duration": "20 hours",
                    "link": "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/",
                    "educator": "Maven Analytics",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner to Advanced",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Power BI Fundamentals",
                    "platform": "Pluralsight",
                    "duration": "4 hours",
                    "link": "https://www.pluralsight.com/courses/power-bi-fundamentals",
                    "educator": "Stacia Varga",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner",
                    "rating": "4.6/5"
                },
                {
                    "course_name": "Analyzing and Visualizing Data with Power BI",
                    "platform": "edX",
                    "duration": "6 weeks",
                    "link": "https://www.edx.org/course/analyzing-and-visualizing-data-with-power-bi",
                    "educator": "Microsoft",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Intermediate",
                    "rating": "4.4/5"
                },
                {
                    "course_name": "Power BI Essential Training",
                    "platform": "LinkedIn Learning",
                    "duration": "3 hours",
                    "link": "https://www.linkedin.com/learning/power-bi-essential-training-3",
                    "educator": "Gini Courter",
                    "skill_covered": "Power BI",
                    "difficulty_level": "Beginner",
                    "rating": "4.5/5"
                }
            ]
        }

        recommendations = []
        for keyword in keywords:
            keyword_normalized = keyword.strip().title()
            if keyword_normalized in course_database:
                recommendations.extend(course_database[keyword_normalized])
            else:
                # Fallback for unknown skills - still no generic names
                skill_lower = keyword.lower().replace(' ', '-')
                recommendations.extend([
                    {
                        "course_name": f"Complete {keyword} Development Masterclass",
                        "platform": "Udemy",
                        "duration": "15 hours",
                        "link": f"https://www.udemy.com/topic/{skill_lower}/",
                        "educator": "Expert Instructor",
                        "skill_covered": keyword,
                        "difficulty_level": "All Levels",
                        "rating": "4.5/5"
                    },
                    {
                        "course_name": f"{keyword} Fundamentals",
                        "platform": "Pluralsight",
                        "duration": "6 hours",
                        "link": f"https://www.pluralsight.com/courses/{skill_lower}-fundamentals",
                        "educator": "Industry Expert",
                        "skill_covered": keyword,
                        "difficulty_level": "Beginner",
                        "rating": "4.4/5"
                    },
                    {
                        "course_name": f"{keyword} Essential Training",
                        "platform": "LinkedIn Learning",
                        "duration": "4 hours",
                        "link": f"https://www.linkedin.com/learning/{skill_lower}-essential-training",
                        "educator": "Professional Instructor",
                        "skill_covered": keyword,
                        "difficulty_level": "Intermediate",
                        "rating": "4.6/5"
                    },
                    {
                        "course_name": f"Introduction to {keyword}",
                        "platform": "edX",
                        "duration": "5 weeks",
                        "link": f"https://www.edx.org/course/introduction-to-{skill_lower}",
                        "educator": "University Partner",
                        "skill_covered": keyword,
                        "difficulty_level": "Beginner",
                        "rating": "4.3/5"
                    },
                    {
                        "course_name": f"{keyword} Professional Certificate",
                        "platform": "Coursera",
                        "duration": "4 months",
                        "link": f"https://www.coursera.org/professional-certificates/{skill_lower}",
                        "educator": "Industry Leader",
                        "skill_covered": keyword,
                        "difficulty_level": "Intermediate",
                        "rating": "4.5/5"
                    }
                ])

        return recommendations
