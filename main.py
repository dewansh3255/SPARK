import psycopg2
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from rapidfuzz import process, fuzz
import pandas as pd
from sqlalchemy import create_engine
import traceback

class CareerNavigator:
    def __init__(self):
        # --- Connection Details for Distributed Setup ---
        load_dotenv()
        pg_user = os.getenv("PG_USER")
        pg_pass = os.getenv("PG_PASS")
        pg_host = os.getenv("PG_HOST")
        pg_db = os.getenv("PG_DB")
        pg_port = os.getenv("PG_PORT")

        mysql_user = os.getenv("MYSQL_USER")
        mysql_pass = os.getenv("MYSQL_PASS")
        mysql_host = os.getenv("MYSQL_HOST")
        mysql_db = os.getenv("MYSQL_DB")
        mysql_port = os.getenv("MYSQL_PORT")
        
        # Create SQLAlchemy Engines for Pandas
        pg_uri = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        mysql_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/{mysql_db}"
        self.pg_engine = create_engine(pg_uri)
        self.mysql_engine = create_engine(mysql_uri)
        
        # Keep raw connection parameters for transactional queries
        self.pg_conn_params = { 
            'dbname': pg_db, 'user': pg_user, 'password': pg_pass, 
            'host': pg_host, 'port': pg_port 
        }
        self.mysql_conn_params = { 
            'host': mysql_host, 'database': mysql_db, 'user': mysql_user, 
            'password': mysql_pass, 'port': mysql_port 
        }

        # Configure LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("Google API Key not found.")
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        print("Career Navigator initialized.")

    def execute_general_query(self, query):
        """The master function to analyze and route any user query."""
        intent_prompt = f"""
        Analyze the user's query to classify its intent and extract entities.

        **Intents:**
        - 'CAREER_PATH': User wants a skill gap analysis for a specific job.
        - 'FIND_JOBS': User wants to find job listings with optional filters.
        - 'ELIGIBLE_JOBS': User wants to find jobs that match their own profile's skills.
        - 'CANDIDATE_SEARCH': User wants to find people who are eligible for a specified job title (e.g., 'List all candidates for data scientist jobs').
        - 'PROFILE_AGGREGATION': User wants to count or aggregate profiles based on skills, company, or experience (e.g., 'How many candidates have Python skill?').
        - 'SKILL_LOOKUP': User wants a simple list of skills for a job, without personalization.
        - 'USER_SKILL_LOOKUP': User wants to list the skills of a specific person.
        - 'SKILL_FORECAST': A manager wants a "Build vs. Buy" analysis for a role at their company.
        - 'UNKNOWN': The query is unclear.

        **Entities to Extract:**
        - 'user_name': The full name of a person mentioned.
        - 'company_name': The name of a company.
        - 'target_job': A specific job title.
        - 'location': A city or geographical area.
        - 'target_skill': A specific skill for aggregation.

        **Instructions:**
        - Return ONLY a raw JSON object.
        - If an entity is not mentioned, its value must be null.

        **Examples:**
        - Query: "How many people have the skill Machine Learning?" -> {{"intent": "PROFILE_AGGREGATION", "user_name": null, "company_name": null, "target_job": null, "location": null, "target_skill": "Machine Learning"}}
        - Query: "List all candidates which are eligible for data scientist jobs" -> {{"intent": "CANDIDATE_SEARCH", "user_name": null, "company_name": null, "target_job": "data scientist", "location": null, "target_skill": null}}
        
        **User Query:** "{query}"

        **JSON Output:**
        """
        try:
            response = self.llm.generate_content(intent_prompt)
            # Clean the JSON response
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            analysis = json.loads(json_text)
            intent = analysis.get("intent")
            
            if intent == "CAREER_PATH": return self.run_dynamic_query(analysis)
            elif intent == "FIND_JOBS": return self.find_jobs_by_criteria(analysis.get("company_name"), analysis.get("target_job"), analysis.get("location"))
            elif intent == "ELIGIBLE_JOBS": return self.find_eligible_jobs_for_user(analysis.get("user_name"))
            elif intent == "CANDIDATE_SEARCH": return self.find_eligible_candidates_for_job(analysis.get("target_job"))
            elif intent == "PROFILE_AGGREGATION": return self.aggregate_profile_query(analysis.get("target_skill"), analysis.get("company_name"))
            elif intent == "SKILL_LOOKUP": return self.get_skills_for_job(analysis.get("target_job"), analysis.get("company_name"))
            elif intent == "USER_SKILL_LOOKUP": return self.get_skills_for_user(analysis.get("user_name"))
            elif intent == "SKILL_FORECAST": return self.run_skill_forecast(analysis.get("company_name"), analysis.get("target_job"))
            else: return "Sorry, I can only answer questions about career paths, job eligibility, job listings, skills, or corporate skill forecasts."
        except Exception as e:
            # Added more robust error handling for LLM failure
            print(f"Error in general query execution: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Failed to parse LLM response: {response.text}")
            return "Sorry, I had trouble understanding your request."
    
    def aggregate_profile_query(self, target_skill=None, company_name=None):
        """
        Handles PROFILE_AGGREGATION intent. Counts profiles matching criteria (e.g., by skill).
        """
        if not target_skill and not company_name:
            return "Please specify a skill or company name for aggregation."

        try:
            sql = """
                SELECT COUNT(DISTINCT p.ProfileID)
                FROM Profiles p
            """
            params = []
            
            if target_skill:
                # Need to join with skills mapping
                sql += " JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID JOIN Skills s ON psm.SkillID = s.SkillID"
            
            conditions = []
            
            if target_skill:
                conditions.append("s.SkillName ILIKE %s")
                params.append(f"%{target_skill}%")
                
            if company_name:
                conditions.append("p.CompanyName ILIKE %s")
                params.append(f"%{company_name}%")
                
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            # Execute the query against the Postgres engine
            df = pd.read_sql_query(sql, self.pg_engine, params=tuple(params))
            
            count = df.iloc[0][0]
            
            # --- Task 3 Innovation: Contextual Output ---
            criteria = []
            if target_skill: criteria.append(f"the skill **{target_skill}**")
            if company_name: criteria.append(f"currently working at **{company_name}**")
            
            criteria_str = " and ".join(criteria)
            
            if count == 0:
                response = f"I found **{count}** candidates matching the criteria ({criteria_str})."
            else:
                response = f"There are **{count}** candidates in the database matching the criteria ({criteria_str})."
            
            return response
            
        except Exception as e:
            print(f"Error in aggregate_profile_query: {e}")
            return "Sorry, an error occurred while calculating the profile aggregation."

    def find_eligible_candidates_for_job(self, target_job_title):
        """
        Handles CANDIDATE_SEARCH intent. Multi-step federation: MySQL (Job Skills) -> Postgres (Profiles).
        """
        if not target_job_title:
            return "Please specify a job title to search for eligible candidates."

        # 1. Fetch skills required for the target job (from MySQL)
        target_skills = self.get_all_skills_for_job_title(target_job_title)
        
        if not target_skills:
            return f"Could not find any jobs with the title '{target_job_title}' to extract required skills."

        # 2. Find profiles matching those skills (query Postgres)
        # Use a high threshold (60%) for eligibility
        matching_candidates_df = self.find_matching_candidates(target_skills, threshold=60)
        
        if matching_candidates_df.empty:
            return f"No candidates were found who match at least 60% of the required skills for a **{target_job_title}** role."

        # 3. Task 3 Innovation: Integrate and Render
        total_candidates = len(matching_candidates_df)
        
        summary_prompt = f"""
        You are an HR recruiter. The user requested a list of eligible candidates for the job title: '{target_job_title}'.
        
        The database returned {total_candidates} candidates matching at least 60% of the required skills.
        
        Based on this, generate a professional, contextual summary that:
        1. Confirms the search (e.g., "Found X candidates...").
        2. Lists the top 3 companies where these eligible candidates currently work.
        3. Presents the final data table.
        
        Use markdown for formatting.
        """
        
        try:
            summary_response = self.llm.generate_content(summary_prompt).text
            # Return the generated text and the DataFrame
            return summary_response, matching_candidates_df
        except Exception as e:
            print(f"Error generating candidate search summary: {e}")
            return f"Found {total_candidates} candidates eligible for **{target_job_title}**. See the table for details.", matching_candidates_df

    def find_matching_candidates(self, required_skills, threshold=0):
        """
        Finds profiles in Postgres that match a percentage of the required skills.
        """
        if not required_skills: return pd.DataFrame()
        try:
            # Create placeholders for the SQL IN clause
            skill_placeholders = ', '.join(['%s'] * len(required_skills))
            
            # The SQL calculates the percentage of the REQUIRED skills that the candidate possesses
            sql = f"""
                SELECT 
                    p.FullName, 
                    p.Headline, 
                    p.YearsOfExperience, 
                    p.CompanyName,
                    STRING_AGG(s.SkillName, ', ') AS MatchedSkills, 
                    (COUNT(DISTINCT s.SkillName) * 100.0 / %s) AS MatchPercentage
                FROM Profiles p
                JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID
                JOIN Skills s ON psm.SkillID = s.SkillID
                WHERE s.SkillName IN ({skill_placeholders})
                GROUP BY p.ProfileID, p.FullName, p.Headline, p.YearsOfExperience, p.CompanyName
                HAVING (COUNT(DISTINCT s.SkillName) * 100.0 / %s) >= %s
                ORDER BY MatchPercentage DESC
                LIMIT 50;
            """
            
            num_required_skills = len(required_skills)
            # FIX: Ensure parameters are passed in the correct order for the 3 placeholders: %s (count), skill_placeholders, %s (count), %s (threshold)
            params = (num_required_skills,) + tuple(required_skills) + (num_required_skills, threshold)

            df = pd.read_sql_query(sql, self.pg_engine, params=params)
            
            # Rename columns for clarity in output
            df = df.rename(columns={'fullname': 'Candidate Name', 'companyname': 'Current Company', 'yearsofexperience': 'YoE', 'headline': 'Headline'})
            return df
        except Exception as e:
            print(f"Error finding matching candidates: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

    def get_all_skills_for_job_title(self, job_title):
        """Fetches all unique skills required across all jobs matching a title."""
        try:
            sql = "SELECT DISTINCT s.SkillName FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID JOIN Jobs j ON jsm.JobID = j.JobID WHERE j.JobTitle LIKE %s"
            params = (f"%{job_title}%",)
            df = pd.read_sql_query(sql, self.mysql_engine, params=params)
            return df['SkillName'].tolist()
        except Exception as e:
            print(f"Error fetching job skills for candidate search: {e}")
            return []

    def find_eligible_jobs_for_user(self, user_name):
        """
        Handles ELIGIBLE_JOBS intent.
        This function implements the multi-step federation: Postgres (Profile) -> MySQL (Jobs).
        """
        if not user_name:
            return "Please specify your name so I can find your profile."
        
        status, profile_data = self.get_user_profile(user_name)
        if status == "AMBIGUOUS":
            return f"Multiple profiles found matching '{user_name}'. Please be more specific with the full name."
        if status != "UNIQUE_MATCH":
            return f"Could not find a unique profile for '{user_name}'. Please try again with a full name."

        # Extract profile details and skills
        profile_id, full_name, headline, experience, company_name = profile_data
        user_skills = self.get_user_skills(profile_id)
        
        # --- Task 3 Innovation: Federation Output Enhancement Module (Step 1) ---
        profile_summary_prompt = f"""
        You are a career coach. Based on the following user data, generate a welcoming summary of their profile and a brief assessment of their skills' potential.
        - Name: {full_name}
        - Headline: {headline}
        - Experience: {experience} years
        - Company: {company_name}
        - Skills: {', '.join(user_skills)}
        
        Keep it friendly, professional, and use markdown. Clearly list the user's skills.
        """
        try:
            profile_summary = self.llm.generate_content(profile_summary_prompt).text
        except Exception as e:
            print(f"Error generating profile summary: {e}")
            profile_summary = f"Hello **{full_name}**! We've retrieved your profile: {headline} with {experience} years of experience at {company_name}. Skills: {', '.join(user_skills)}."


        if not user_skills:
            return f"{profile_summary}\n\n**{full_name}** has no skills listed. Cannot find eligible jobs."

        # Find jobs with at least a 40% skill match (MySQL query)
        matching_jobs = self.find_matching_jobs(user_skills, threshold=40)
        
        if not matching_jobs.empty:
            response_text = f"{profile_summary}\n\n---\n\n### Eligible Jobs Found\nFound **{len(matching_jobs)}** jobs that could be a great fit based on your skills (Match $\geq$ 40%):"
            return response_text, matching_jobs
        else:
            top_skills = self.get_top_skills(limit=3)
            return f"{profile_summary}\n\nSorry **{full_name}**, no jobs were found with a significant match to your current skills. The top 3 most in-demand skills right now are **{', '.join(top_skills)}**. Focusing on one of these could greatly improve your job prospects!"

    def run_dynamic_query(self, analysis):
        user_name = analysis.get('user_name')
        target_job = analysis.get('target_job')
        target_company = analysis.get('company_name')
        if not all([user_name, target_job, target_company]): return "For a career path, please specify your name, a target job, and a company."
        
        status, data = self.get_user_profile(user_name)
        if status == "AMBIGUOUS": return f"Multiple profiles found: **{', '.join([p[1] for p in data])}**. Please be more specific."
        if status != "UNIQUE_MATCH": return f"Sorry, a profile for '{user_name}' was not found."
        
        profile_data = data
        my_skills = self.get_user_skills(profile_data[0])
        req_skills = self.get_target_job_skills(target_job, target_company)
        if not req_skills: return f"Sorry, no data found for '{target_job}' at '{target_company}'."
        
        mand_gap, pref_gap = self.analyze_skill_gap(my_skills, req_skills)
        return self.generate_learning_path(profile_data, mand_gap, pref_gap)
        
    def get_skills_for_user(self, user_name):
        if not user_name: return "Please specify a name to look up their skills."
        status, data = self.get_user_profile(user_name)
        if status == "AMBIGUOUS": return f"Multiple profiles found: **{', '.join([p[1] for p in data])}**. Please be more specific."
        if status != "UNIQUE_MATCH": return f"Sorry, a profile for '{user_name}' was not found."
        user_skills = self.get_user_skills(data[0])
        if not user_skills: return f"**{data[1]}** has no skills listed."
        return f"Here are the skills for **{data[1]}**:\n\n- {'\n- '.join(user_skills)}"

    def get_skills_for_job(self, job_title, company_name):
        if not job_title or not company_name: return "Please specify both a job title and a company name."
        skills = self.get_target_job_skills(job_title, company_name)
        if not skills: return f"Sorry, no data found for the role '{job_title}' at '{company_name}'."
        # Use set to get unique skills
        unique_skills = set(s[0] for s in skills)
        mandatory = {s for s, i in skills if i == 'Mandatory'}
        preferred = unique_skills - mandatory
        response = f"Skills for a **{job_title}** at **{company_name}**:\n\n"
        if mandatory: response += f"**Mandatory:**\n- {'\n- '.join(sorted(list(mandatory)))}\n\n"
        if preferred: response += f"**Preferred:**\n- {'\n- '.join(sorted(list(preferred)))}"
        return response

    def find_jobs_by_criteria(self, company_name=None, job_title=None, location=None):
        if not any([company_name, job_title, location]): return "Please specify criteria to find jobs."
        try:
            sql = "SELECT j.JobTitle, j.CompanyName, j.Location, GROUP_CONCAT(DISTINCT s.SkillName SEPARATOR ', ') AS RequiredSkills FROM Jobs j LEFT JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID LEFT JOIN Skills s ON jsm.SkillID = s.SkillID"
            conditions, params = [], []
            if company_name: conditions.append("j.CompanyName LIKE %s"); params.append(f"%{company_name}%")
            if job_title: conditions.append("j.JobTitle LIKE %s"); params.append(f"%{job_title}%")
            if location: conditions.append("j.Location LIKE %s"); params.append(f"%{location}%")
            if conditions: sql += " WHERE " + " AND ".join(conditions)
            sql += " GROUP BY j.JobID ORDER BY j.CompanyName, j.JobTitle;"
            df = pd.read_sql_query(sql, self.mysql_engine, params=tuple(params))
            if df.empty: return "No job openings found matching your criteria."
            return f"Found **{len(df)}** job openings matching your criteria:", df
        except Exception as e:
            print(f"Error in find_jobs_by_criteria: {e}")
            return "Sorry, an error occurred while retrieving job data."

    def find_matching_jobs(self, user_skills, threshold=0):
        if not user_skills: return pd.DataFrame()
        try:
            # Create placeholders for the SQL IN clause for user_skills
            skill_placeholders = ', '.join(['%s'] * len(user_skills))
            
            # The SQL calculates the percentage of job skills that the user possesses
            sql = f"""
                SELECT 
                    j.JobTitle, 
                    j.CompanyName, 
                    j.Location, 
                    (SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) / COUNT(DISTINCT jsm.SkillID)) * 100 AS MatchPercentage
                FROM Jobs j 
                LEFT JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID 
                LEFT JOIN Skills s ON jsm.SkillID = s.SkillID 
                GROUP BY j.JobID, j.JobTitle, j.CompanyName, j.Location
                HAVING 
                    SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) > 0 
                    AND (SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) / COUNT(DISTINCT jsm.SkillID)) * 100 >= %s
                ORDER BY MatchPercentage DESC 
                LIMIT 50;
            """
            
            # The parameter list must contain the list of skills repeated three times (for the three IN clauses) 
            # followed by the threshold (for the >= %s clause).
            params = tuple(user_skills) * 3 + (threshold,)

            df = pd.read_sql_query(sql, self.mysql_engine, params=params)
            return df
        except Exception as e:
            print(f"Error finding matching jobs: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

    def get_top_skills(self, limit=5):
        try:
            sql = f"SELECT s.SkillName, COUNT(jsm.SkillID) as SkillCount FROM Job_Skills_Mapping jsm JOIN Skills s ON jsm.SkillID = s.SkillID GROUP BY s.SkillName ORDER BY SkillCount DESC LIMIT {limit};"
            return pd.read_sql_query(sql, self.mysql_engine)['SkillName'].tolist()
        except Exception as e:
            print(f"Error getting top skills: {e}")
            return []

    def get_user_profile(self, full_name):
        try:
            # First query to get all profiles for fuzzy matching (rapidfuzz)
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ProfileID, FullName FROM Profiles;")
                    all_profiles = cur.fetchall()
            
            profile_choices = {name: pid for pid, name in all_profiles}
            
            # Use fuzzy matching to handle partial or slight misspellings
            best_matches = process.extract(full_name, profile_choices.keys(), scorer=fuzz.WRatio, limit=5, score_cutoff=80)
            
            if not best_matches: return "NO_MATCH", None
            
            # Determine which profile IDs to fetch data for (only the best ones)
            top_match_name, top_match_score, _ = best_matches[0]
            if top_match_score > 95: 
                # Very high confidence match, select only the top one
                matched_profile_ids = [profile_choices[top_match_name]]
            else: 
                # Ambiguous matches (multiple profiles with scores > 80)
                matched_profile_ids = [profile_choices[match[0]] for match in best_matches]
            
            # Second query to get full profile data
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    query_placeholders = ','.join(['%s'] * len(matched_profile_ids))
                    # Fetching all required fields for the output
                    cur.execute(f"SELECT ProfileID, FullName, Headline, YearsOfExperience, CompanyName FROM Profiles WHERE ProfileID IN ({query_placeholders});", tuple(matched_profile_ids))
                    matched_profiles_data = cur.fetchall()
            
            if len(matched_profiles_data) == 1: 
                # Return tuple of (ProfileID, FullName, Headline, YearsOfExperience, CompanyName)
                return "UNIQUE_MATCH", matched_profiles_data[0]
            else: 
                return "AMBIGUOUS", matched_profiles_data
        except Exception as e:
            print(f"PostgreSQL Error: {e}")
            return "ERROR", None

    def get_user_skills(self, profile_id):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT s.SkillName FROM Skills s JOIN Profile_Skills_Mapping psm ON s.SkillID = psm.SkillID WHERE psm.ProfileID = %s;", (profile_id,))
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error fetching skills: {e}")
            return []
        
    def get_target_job_skills(self, job_title, company_name):
        try:
            with mysql.connector.connect(**self.mysql_conn_params) as conn:
                with conn.cursor() as cursor:
                    # Use LIKE for case-insensitive/partial matching in a non-exact query context
                    cursor.execute("SELECT s.SkillName, jsm.Importance FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID JOIN Jobs j ON jsm.JobID = j.JobID WHERE j.JobTitle LIKE %s AND j.CompanyName LIKE %s;", (f"%{job_title}%", f"%{company_name}%"))
                    return cursor.fetchall() or None
        except Error as e:
            print(f"MySQL Error: {e}")
            return None

    def analyze_skill_gap(self, current_skills, required_skills):
        current_skills_set = set(current_skills)
        # Use set to handle potential duplicate skills from DB
        unique_required_skills = {s[0] for s in required_skills}
        mandatory_skills = {s[0] for s in required_skills if s[1] == 'Mandatory'}
        
        mandatory_gap = sorted(list(mandatory_skills - current_skills_set))
        preferred_gap = sorted(list((unique_required_skills - mandatory_skills) - current_skills_set))
        return mandatory_gap, preferred_gap

    def generate_learning_path(self, profile_data, mandatory_gap, preferred_gap):
        if not mandatory_gap and not preferred_gap: return "Congratulations! You have all the required skills for this role."
        user_name = profile_data[1]
        prompt = f"""You are a career coach. Create a learning path for {user_name}.
        Gaps: Mandatory: {', '.join(mandatory_gap)}; Preferred: {', '.join(preferred_gap)}.
        For each, explain its importance and recommend one free resource with a link. Use markdown."""
        try:
            return self.llm.generate_content(prompt).text
        except Exception as e:
            return f"An error occurred while generating the learning path: {e}"

    def get_internal_skill_audit(self, company_name, target_skills):
        """
        Analyzes the internal employee database (Postgres) for upskilling potential,
        pre-filtered by a list of target skills.
        
        This function ONLY returns the raw data.
        """
        try:
            if not target_skills:
                # No skills to search for
                return pd.DataFrame() 

            # Create placeholders for the SQL IN clause
            skill_placeholders = ', '.join(['%s'] * len(target_skills))
            
            # Find employees at the company who have AT LEAST 2 of the required skills
            # Order by the best candidates (most skills) first
            sql = f"""
                SELECT p.FullName, STRING_AGG(s.SkillName, ', ') AS Skills, COUNT(s.SkillName) as SkillCount
                FROM Profiles p
                JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID
                JOIN Skills s ON psm.SkillID = s.SkillID
                WHERE p.CompanyName = %s
                AND s.SkillName IN ({skill_placeholders})
                GROUP BY p.ProfileID, p.FullName
                HAVING COUNT(DISTINCT s.SkillName) >= 2
                ORDER BY SkillCount DESC
                LIMIT 10; 
            """
            
            # Parameters for the query
            params = (company_name,) + tuple(target_skills)
            
            internal_employees_df = pd.read_sql_query(sql, self.pg_engine, params=params)
            
            return internal_employees_df

        except Exception as e:
            print(f"Error in internal skill audit SQL: {e}")
            return pd.DataFrame() # Return empty dataframe on error
                
    def get_external_market_analysis(self, target_job_title):
        """
        Analyzes the external job market (MySQL) for hiring data.
        """
        try:
            sql = "SELECT COUNT(DISTINCT j.JobID) AS JobCount, GROUP_CONCAT(DISTINCT j.Location SEPARATOR ', ') AS Locations FROM Jobs j WHERE j.JobTitle LIKE %s"
            params = (f"%{target_job_title}%",)
            df = pd.read_sql_query(sql, self.mysql_engine, params=params)
            
            if df.empty or df.iloc[0]['JobCount'] == 0:
                return {"open_jobs_count": 0, "common_locations": "N/A"}
            
            # This logic assumes GROUP_CONCAT returns a comma-separated string
            all_locations = df.iloc[0]['Locations'].split(', ')
            common_locations = pd.Series(all_locations).value_counts().head(3).index.tolist()

            return {
                "open_jobs_count": int(df.iloc[0]['JobCount']),
                "common_locations": ", ".join(common_locations)
            }
        except Exception as e:
            print(f"Error in external market analysis: {e}")
            return {"open_jobs_count": 0, "common_locations": "N/A"}

    def run_skill_forecast(self, company_name, target_job_title):
        """
        Runs the full "Build vs. Buy" analysis and generates a report.
        """
        if not company_name or not target_job_title:
            return "To generate a forecast, please specify your company and the target job title (e.g., 'Data Scientist')."
        
        # --- ROBUST LOGIC ---
        # 1. Get Target Skills from MySQL
        try:
            # First, try to get MANDATORY skills
            sql_mandatory = "SELECT DISTINCT s.SkillName FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID JOIN Jobs j ON jsm.JobID = j.JobID WHERE j.JobTitle LIKE %s AND jsm.Importance = 'Mandatory'"
            params = (f"%{target_job_title}%",)
            target_skills_df = pd.read_sql_query(sql_mandatory, self.mysql_engine, params=params)
            
            target_skills = target_skills_df['SkillName'].tolist()

            # 2. FALLBACK: If no mandatory skills, get ALL skills
            if not target_skills:
                print("No mandatory skills found, falling back to all skills.")
                sql_all = "SELECT DISTINCT s.SkillName FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID JOIN Jobs j ON jsm.JobID = j.JobID WHERE j.JobTitle LIKE %s"
                target_skills_df = pd.read_sql_query(sql_all, self.mysql_engine, params=params)
                target_skills = target_skills_df['SkillName'].tolist()

        except Exception as e:
            print(f"Error getting target skills for forecast: {e}")
            target_skills = []
        
        # 3. Federate: Call your helper functions
        internal_candidates_df = self.get_internal_skill_audit(company_name, target_skills)
        external_data = self.get_external_market_analysis(target_job_title)
        
        # 4. Convert internal data to a compact JSON for the prompt
        if not internal_candidates_df.empty:
            internal_data_json = internal_candidates_df.to_json(orient="records")
            internal_total_count = len(internal_candidates_df)
        else:
            internal_data_json = "[]"
            internal_total_count = 0
        
        # 5. Integrate & Generate ONE Report (The single LLM call)
        prompt = f"""
        You are a top-tier HR Strategy consultant for {company_name}.
        I need a "Build vs. Buy" strategic report for the role of '{target_job_title}'.
        Use the following federated data to write your recommendation.
        Use markdown for formatting. Be concise and professional.

        **Internal Data (Our Company - {company_name}):**
        - Total Pre-filtered Employees (with >=2 relevant skills): {internal_total_count}
        - Top 10 High-Potential Employees (JSON): {internal_data_json}

        **External Market Data (Hiring):**
        - Current open jobs for '{target_job_title}': {external_data['open_jobs_count']}
        - Top 3 hiring locations: {external_data['common_locations']}

        **Your Report:**
        (Start with a '## Strategic Recommendation' section offering a clear "Build", "Buy", or "Hybrid" suggestion. 
        Then create two sections: '### Build (Internal Upskilling)' and '### Buy (External Hiring)'.
        In the 'Build' section, **explicitly mention the names** of the high-potential employees from the JSON, if any.)
        """
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating final forecast report: {e}")
            return "Sorry, an error occurred while generating the strategic report."

    # --- Profile CRUD Operations ---
    def register_new_user(self, name, headline, experience, skills, company_name):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    # Added CompanyName to INSERT
                    cur.execute("INSERT INTO Profiles (FullName, Headline, YearsOfExperience, CompanyName) VALUES (%s, %s, %s, %s) RETURNING ProfileID;", (name, headline, experience, company_name))
                    profile_id = cur.fetchone()[0]
                    if skills:
                        skill_ids_query = f"SELECT SkillID FROM Skills WHERE SkillName IN ({','.join(['%s'] * len(skills))})"
                        cur.execute(skill_ids_query, tuple(skills))
                        skill_ids = [row[0] for row in cur.fetchall()]
                        for skill_id in skill_ids:
                            cur.execute("INSERT INTO Profile_Skills_Mapping (ProfileID, SkillID) VALUES (%s, %s);", (profile_id, skill_id))
            return True
        except Exception as e:
            print(f"Error in register_new_user: {e}")
            return False

    def update_profile(self, profile_id, name, headline, experience, skills, company_name):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    # Added CompanyName to UPDATE
                    cur.execute("UPDATE Profiles SET FullName = %s, Headline = %s, YearsOfExperience = %s, CompanyName = %s WHERE ProfileID = %s;", (name, headline, experience, company_name, profile_id))
                    cur.execute("DELETE FROM Profile_Skills_Mapping WHERE ProfileID = %s;", (profile_id,))
                    if skills:
                        skill_ids_query = f"SELECT SkillID FROM Skills WHERE SkillName IN ({','.join(['%s'] * len(skills))})"
                        cur.execute(skill_ids_query, tuple(skills))
                        skill_ids = [row[0] for row in cur.fetchall()]
                        for skill_id in skill_ids:
                            cur.execute("INSERT INTO Profile_Skills_Mapping (ProfileID, SkillID) VALUES (%s, %s);", (profile_id, skill_id))
            return True
        except Exception as e:
            print(f"Error in update_profile: {e}")
            return False
    
    def get_all_skills(self):
        """Fetches a complete, sorted list of all unique skills from both databases."""
        try:
            sql_pg = "SELECT DISTINCT SkillName FROM Skills;"
            sql_mysql = "SELECT DISTINCT SkillName FROM Skills;"
            
            df_pg = pd.read_sql_query(sql_pg, self.pg_engine)
            df_mysql = pd.read_sql_query(sql_mysql, self.mysql_engine)
            
            # Combine, drop any null/NaN values, convert all to string, get unique
            all_skills_series = pd.concat([df_pg, df_mysql])['SkillName'].dropna().astype(str)
            
            # Sort the unique list
            all_skills_list = sorted(all_skills_series.unique())
            
            return all_skills_list
        except Exception as e:
            print(f"Error fetching all skills: {e}")
            return [] # Fallback

    def delete_profile(self, profile_id):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM Profiles WHERE ProfileID = %s;", (profile_id,))
            return True
        except Exception as e:
            print(f"Error in delete_profile: {e}")
            return False

    # --- Job CRUD Operations ---
    def add_job(self, title, company, location, skills, importance_map):
        try:
            with mysql.connector.connect(**self.mysql_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO Jobs (JobTitle, CompanyName, Location) VALUES (%s, %s, %s);", (title, company, location))
                    job_id = cur.lastrowid
                    if skills:
                        skill_ids_query = f"SELECT SkillID, SkillName FROM Skills WHERE SkillName IN ({','.join(['%s'] * len(skills))})"
                        cur.execute(skill_ids_query, tuple(skills))
                        skill_id_map = {name: sid for sid, name in cur.fetchall()}
                        for skill_name in skills:
                            skill_id = skill_id_map.get(skill_name)
                            importance = importance_map.get(skill_name, 'Preferred')
                            if skill_id:
                                cur.execute("INSERT INTO Job_Skills_Mapping (JobID, SkillID, Importance) VALUES (%s, %s, %s);", (job_id, skill_id, importance))
                conn.commit() 
            return True
        except Exception as e:
            print(f"Error in add_job: {e}")
            return False

    def update_job(self, job_id, title, company, location, skills, importance_map):
        try:
            with mysql.connector.connect(**self.mysql_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE Jobs SET JobTitle = %s, CompanyName = %s, Location = %s WHERE JobID = %s;", (title, company, location, job_id))
                    cur.execute("DELETE FROM Job_Skills_Mapping WHERE JobID = %s;", (job_id,))
                    if skills:
                        skill_ids_query = f"SELECT SkillID, SkillName FROM Skills WHERE SkillName IN ({','.join(['%s'] * len(skills))})"
                        cur.execute(skill_ids_query, tuple(skills))
                        skill_id_map = {name: sid for sid, name in cur.fetchall()}
                        for skill_name in skills:
                            skill_id = skill_id_map.get(skill_name)
                            importance = importance_map.get(skill_name, 'Preferred')
                            if skill_id:
                                cur.execute("INSERT INTO Job_Skills_Mapping (JobID, SkillID, Importance) VALUES (%s, %s, %s);", (job_id, skill_id, importance))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error in update_job: {e}")
            return False

    def delete_job(self, job_id):
        try:
            with mysql.connector.connect(**self.mysql_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET FOREIGN_KEY_CHECKS=0;")
                    cur.execute("DELETE FROM Jobs WHERE JobID = %s;", (job_id,))
                    cur.execute("DELETE FROM Job_Skills_Mapping WHERE JobID = %s;", (job_id,))
                    cur.execute("SET FOREIGN_KEY_CHECKS=1;")
                conn.commit()
            return True
        except Exception as e:
            print(f"Error in delete_job: {e}")
            return False
                
    # --- Data Fetching for Display ---
    def get_all_profiles_data(self):
        try:
            # Postgres uses STRING_AGG
            sql = "SELECT p.ProfileID, p.FullName, p.Headline, p.YearsOfExperience, p.CompanyName, STRING_AGG(s.SkillName, ', ') AS Skills FROM Profiles p LEFT JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID LEFT JOIN Skills s ON psm.SkillID = s.SkillID GROUP BY p.ProfileID, p.FullName, p.Headline, p.YearsOfExperience, p.CompanyName ORDER BY p.FullName;"
            return pd.read_sql_query(sql, self.pg_engine)
        except Exception as e:
            print(f"Error fetching profiles data: {e}")
            return pd.DataFrame()
        
    def get_all_jobs_data_for_crud(self):
        try:
            # MySQL uses GROUP_CONCAT
            sql = "SELECT j.JobID, j.JobTitle, j.CompanyName, j.Location, GROUP_CONCAT(s.SkillName SEPARATOR ', ') AS RequiredSkills FROM Jobs j LEFT JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID LEFT JOIN Skills s ON jsm.SkillID = s.SkillID GROUP BY j.JobID ORDER BY j.CompanyName, j.JobTitle;"
            return pd.read_sql_query(sql, self.mysql_engine)
        except Exception as e:
            print(f"Error fetching jobs data for CRUD: {e}")
            return pd.DataFrame()