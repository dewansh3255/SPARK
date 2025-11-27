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
        # --- Connection Details ---
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
        
        # SQLAlchemy Engines
        pg_uri = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        mysql_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/{mysql_db}"
        self.pg_engine = create_engine(pg_uri)
        self.mysql_engine = create_engine(mysql_uri)
        
        # Raw Params
        self.pg_conn_params = { 'dbname': pg_db, 'user': pg_user, 'password': pg_pass, 'host': pg_host, 'port': pg_port }
        self.mysql_conn_params = { 'host': mysql_host, 'database': mysql_db, 'user': mysql_user, 'password': mysql_pass, 'port': mysql_port }

        # LLM Setup
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("Google API Key not found.")
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        print("Career Navigator initialized.")

    def execute_general_query(self, query):
        """
        Master function. Handles standard queries, ANALYTICS, and complex filters.
        """
        intent_prompt = f"""
        You are the query orchestrator for SPARK.
        Analyze the user's query to classify intent and extract detailed entities.
        
        **Goal:** Break down the query into a LIST of actionable tasks.
        
        **CRITICAL DECOMPOSITION RULES:**
        1. **"AND" Logic:** If the user asks distinct questions separated by "AND", split them into separate objects (e.g., "Find Jobs AND Find People" -> 2 objects).
           - Distinguish "Python AND SQL" (One task, 2 skills) vs "Find Jobs AND Find People" (Two separate tasks).
        2. **"OR" Logic (Job Titles):** If the user searches for multiple job titles (e.g., "Software Engineer OR Backend Engineer"), create a **SEPARATE** 'FIND_JOBS' task for each title.
           - *Reason:* The database can only search one title at a time.
           - *Example:* "Jobs for Data Scientist or Analyst" -> [{{ "intent": "FIND_JOBS", "target_job": "Data Scientist" }}, {{ "intent": "FIND_JOBS", "target_job": "Analyst" }}]

        **Intents:**
        1. 'CAREER_PATH': Personalized skill-gap analysis.
        2. 'FIND_JOBS': Search job listings.
        3. 'ELIGIBLE_JOBS': Find jobs matching a user's profile.
        4. 'CANDIDATE_SEARCH': Find candidates eligible for a specific job.
        5. 'FIND_PEOPLE': Search candidates with filters (Skills, Location, Experience).
        6. 'ANALYTICS': Questions involving Counts, Rankings, Averages, or Comparisons.
        7. 'PROFILE_AGGREGATION': Simple counts of people.
        8. 'SKILL_FORECAST': "Build vs. Buy" analysis.
        9. 'SKILL_LOOKUP': List skills for a job.
        10. 'USER_SKILL_LOOKUP': List skills of a person.

        **Entities:**
        - 'user_name': Person's name.
        - 'company_name': Target company.
        - 'exclude_company': Company to EXCLUDE (e.g. "NOT at Google").
        - 'target_job': **SINGLE** Job title string (e.g. "Manager"). Never a list.
        - 'location': City or list of cities.
        - 'target_skill': Skill or list of strings.
        - 'min_experience': Integer (years).
        
        **Analytics Specifics:**
        - 'metric': "COUNT", "AVG", "RANK", "COMPARE"
        - 'target_table': "PROFILES" (for candidates/employees) or "JOBS" (for openings).
        - 'group_by': "Company", "Location", "Headline", "Skill"
        - 'limit': Integer (e.g. Top 5 -> 5)

        **Normalization:**
        - "ML" -> "Machine Learning", "DL" -> "Deep Learning"
        - "Frontend Engineer" -> "Frontend Developer"

        **Instructions:**
        - Return a JSON LIST of objects (e.g. [{{...}}, {{...}}]).
        - Do not include markdown formatting.

        **Few-Shot Examples:**

        *User:* "I know Python, Docker, and Go. Any jobs for Software Engineer or Backend Engineer?"
        *JSON:* [
            {{ "intent": "FIND_JOBS", "target_job": "Software Engineer", "target_skill": ["Python", "Docker", "Go"] }},
            {{ "intent": "FIND_JOBS", "target_job": "Backend Engineer", "target_skill": ["Python", "Docker", "Go"] }}
        ]

        *User:* "Find candidates with Python AND Machine Learning in Bangalore."
        *JSON:* [{{ "intent": "FIND_PEOPLE", "target_skill": ["Python", "Machine Learning"], "location": ["Bangalore"] }}]

        *User:* "List jobs at Google AND find candidates for Data Scientist."
        *JSON:* [{{ "intent": "FIND_JOBS", "company_name": "Google" }}, {{ "intent": "CANDIDATE_SEARCH", "target_job": "Data Scientist" }}]

        *User:* "What is the average years of experience for Python devs? AND List top 5 companies with most employees > 8 years exp. AND Show me candidates with ML but NOT at Google."
        *JSON:* [
            {{ "intent": "ANALYTICS", "metric": "AVG", "target_table": "PROFILES", "target_skill": ["Python"] }},
            {{ "intent": "ANALYTICS", "metric": "RANK", "target_table": "PROFILES", "group_by": "Company", "min_experience": 8, "limit": 5 }},
            {{ "intent": "FIND_PEOPLE", "target_skill": ["Machine Learning"], "exclude_company": "Google" }}
        ]

        **User Query:** "{query}"
        **JSON:**
        """
        try:
            response = self.llm.generate_content(intent_prompt)
            json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            
            # DEBUG: See how the LLM split the tasks
            print(f"\n[DEBUG] Decomposed Tasks:\n{json_text}\n")
            
            data = json.loads(json_text)
            analysis_list = [data] if isinstance(data, dict) else data
                
            final_results = []

            for analysis in analysis_list:
                intent = analysis.get("intent")
                result = None
                
                if intent == "CAREER_PATH": result = self.run_dynamic_query(analysis)
                elif intent == "FIND_JOBS": 
                    result = self.find_jobs_by_criteria(
                        analysis.get("company_name"), 
                        analysis.get("target_job"), 
                        analysis.get("location"),
                        analysis.get("target_skill")
                    )
                elif intent == "ELIGIBLE_JOBS": result = self.find_eligible_jobs_for_user(analysis.get("user_name"))
                elif intent == "CANDIDATE_SEARCH": result = self.find_eligible_candidates_for_job(analysis.get("target_job"))
                elif intent == "FIND_PEOPLE": 
                    result = self.find_people(
                        analysis.get("target_skill"), 
                        analysis.get("company_name"), 
                        analysis.get("min_experience"),
                        analysis.get("location"),
                        analysis.get("exclude_company")
                    )
                elif intent == "ANALYTICS": result = self.run_analytics_query(analysis)
                elif intent == "PROFILE_AGGREGATION": result = self.aggregate_profile_query(analysis.get("target_skill"), analysis.get("company_name"))
                elif intent == "SKILL_LOOKUP": result = self.get_skills_for_job(analysis.get("target_job"), analysis.get("company_name"))
                elif intent == "USER_SKILL_LOOKUP": result = self.get_skills_for_user(analysis.get("user_name"))
                elif intent == "SKILL_FORECAST": result = self.run_skill_forecast(analysis.get("company_name"), analysis.get("target_job"))
                else: result = "Sorry, I can only answer questions about career paths, job eligibility, job listings, skills, analytics, or corporate skill forecasts."
                
                if result:
                    final_results.append(result)
            
            return final_results
        
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return ["Sorry, I encountered an error."]
        
    # --- ANALYTICS ENGINE (Handles Complex Queries) ---
    def run_analytics_query(self, params):
        metric = params.get("metric")
        target_table = params.get("target_table")
        group_by = params.get("group_by")
        target_skill = params.get("target_skill")
        min_exp = params.get("min_experience")
        limit = params.get("limit", 10)
        target_job = params.get("target_job") 
        
        try:
            # 1. PROFILE ANALYTICS (Postgres)
            if target_table == "PROFILES":
                if metric == "AVG" and target_skill: 
                     sql = """
                        SELECT AVG(p.YearsOfExperience) 
                        FROM Profiles p 
                        JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID 
                        JOIN Skills s ON psm.SkillID = s.SkillID 
                        WHERE s.SkillName ILIKE %s
                     """
                     val = pd.read_sql_query(sql, self.pg_engine, params=(target_skill[0],)).iloc[0][0]
                     return f"The average years of experience for candidates with **{target_skill[0]}** is **{round(val, 1)} years**."

                if (metric == "RANK" or metric == "COUNT") and group_by == "Company": 
                    where_clause = "WHERE p.YearsOfExperience > %s" if min_exp else ""
                    p = (min_exp,) if min_exp else ()
                    sql = f"SELECT CompanyName, COUNT(*) as Count FROM Profiles p {where_clause} GROUP BY CompanyName ORDER BY Count DESC LIMIT {limit}"
                    df = pd.read_sql_query(sql, self.pg_engine, params=p)
                    return f"Top companies with employees having > {min_exp} years experience:", df

                if (metric == "RANK" or metric == "COUNT") and group_by == "Headline": 
                    if target_job:
                        sql = "SELECT Headline, COUNT(*) as Count FROM Profiles WHERE Headline ILIKE %s GROUP BY Headline ORDER BY Count DESC"
                        params = (f"%{target_job}%",)
                        label = f"containing '{target_job}'"
                    else:
                        sql = "SELECT Headline, COUNT(*) as Count FROM Profiles GROUP BY Headline ORDER BY Count DESC LIMIT 10"
                        params = ()
                        label = "(Top 10)"

                    df = pd.read_sql_query(sql, self.pg_engine, params=params) 
                    return f"Profile counts for titles {label}:", df

            # 2. JOB ANALYTICS (MySQL)
            if target_table == "JOBS":
                # Handle Specific Rankings
                if metric == "RANK" and group_by == "Location": 
                    title = params.get("target_job", "")
                    sql = "SELECT Location, COUNT(*) as JobOpenings FROM Jobs WHERE JobTitle LIKE %s GROUP BY Location ORDER BY JobOpenings DESC LIMIT 5"
                    df = pd.read_sql_query(sql, self.mysql_engine, params=(f"%{title}%",))
                    return f"Locations with most openings for **{title}**:", df
                
                if metric == "RANK" and group_by == "Skill": 
                    title = params.get("target_job", "")
                    sql = """
                        SELECT s.SkillName, COUNT(jsm.JobID) as DemandCount 
                        FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID 
                        JOIN Jobs j ON jsm.JobID = j.JobID
                        WHERE j.JobTitle LIKE %s AND jsm.Importance = 'Mandatory'
                        GROUP BY s.SkillName ORDER BY DemandCount DESC LIMIT 3
                    """
                    df = pd.read_sql_query(sql, self.mysql_engine, params=(f"%{title}%",))
                    return f"Top demanded mandatory skills for **{title}**:", df

                # --- FIX START: Catch-all for "List Companies/Jobs" queries ---
                # If we are here, it's a JOB query that isn't a specific RANK. 
                # Redirect it to the universal job search engine.
                return self.find_jobs_by_criteria(
                    company_name=params.get("company_name"), 
                    location=params.get("location"), 
                    skills=target_skill
                )
                # --- FIX END ---

            # 3. COMPARISON (Cross-DB)
            if metric == "COMPARE":
                skill = target_skill[0] if target_skill else ""
                p_sql = "SELECT COUNT(DISTINCT p.ProfileID) FROM Profiles p JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID JOIN Skills s ON psm.SkillID = s.SkillID WHERE s.SkillName ILIKE %s"
                p_count = pd.read_sql_query(p_sql, self.pg_engine, params=(skill,)).iloc[0][0]
                
                j_sql = "SELECT COUNT(DISTINCT j.JobID) FROM Jobs j JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID JOIN Skills s ON jsm.SkillID = s.SkillID WHERE s.SkillName LIKE %s"
                j_count = pd.read_sql_query(j_sql, self.mysql_engine, params=(skill,)).iloc[0][0]
                
                ratio = f"{p_count}:{j_count}"
                analysis = "Surplus" if p_count > j_count else "Shortage"
                return f"**Supply vs Demand Analysis for {skill}:**\n\n- **Profiles Available:** {p_count}\n- **Active Jobs:** {j_count}\n- **Market Status:** {analysis} (Ratio {ratio})"

            return "I understood the analysis request, but I don't have a specific calculation module for that combination yet."

        except Exception as e:
            print(f"Analytics Error: {e}")
            traceback.print_exc()
            return "Sorry, could not perform that analysis."

    # --- UNIVERSAL SEARCH (People) ---
    def find_people(self, skills=None, company_name=None, min_experience=None, location=None, exclude_company=None):
        if not any([skills, company_name, min_experience, location]):
            return "Please specify a skill, company, location, or experience level."
            
        try:
            params = []
            conditions = []
            having_clause = ""
            
            sql = """
                SELECT p.FullName, p.Headline, p.CompanyName, p.Location, p.YearsOfExperience, STRING_AGG(s.SkillName, ', ') as Skills
                FROM Profiles p
                LEFT JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID
                LEFT JOIN Skills s ON psm.SkillID = s.SkillID
            """
            
            if company_name:
                conditions.append("p.CompanyName ILIKE %s")
                params.append(f"%{company_name}%")
            
            if exclude_company:
                conditions.append("p.CompanyName NOT ILIKE %s")
                params.append(f"%{exclude_company}%")
                
            if location:
                if isinstance(location, str): location = [location]
                loc_placeholders = ' OR '.join(['p.Location ILIKE %s' for _ in location])
                conditions.append(f"({loc_placeholders})")
                params.extend([f"%{l}%" for l in location])

            if min_experience is not None:
                conditions.append("p.YearsOfExperience >= %s")
                params.append(min_experience)
            
            if skills:
                if isinstance(skills, str): skills = [skills]
                lower_skills = [s.lower() for s in skills]
                placeholders = ', '.join(['%s'] * len(skills))
                conditions.append(f"LOWER(s.SkillName) IN ({placeholders})")
                params.extend(lower_skills)
                having_clause = "HAVING COUNT(DISTINCT LOWER(s.SkillName)) >= %s"
            
            if conditions: sql += " WHERE " + " AND ".join(conditions)
            
            sql += " GROUP BY p.ProfileID, p.FullName, p.Headline, p.CompanyName, p.Location, p.YearsOfExperience"
            
            if having_clause and skills:
                sql += " " + having_clause
                params.append(len(skills))
            
            df = pd.read_sql_query(sql, self.pg_engine, params=tuple(params))
            
            if df.empty: return "No candidates found matching your specific criteria."
            return f"Found **{len(df)}** candidates.", df

        except Exception as e:
            print(f"Error in find_people: {e}")
            traceback.print_exc()
            return "Error searching for people."

    # --- UNIVERSAL SEARCH (Jobs) ---
# --- UNIVERSAL SEARCH (Jobs) ---
    def find_jobs_by_criteria(self, company_name=None, job_title=None, location=None, skills=None):
        """
        Fixed to handle LOCATION LISTS and show ALL skills in output.
        """
        if not any([company_name, job_title, location, skills]): return "Please specify criteria."
        try:
            final_params = []
            where_sql_part = ""
            
            # 1. Build WHERE
            where_conditions = []
            if company_name: 
                where_conditions.append("j.CompanyName LIKE %s")
                final_params.append(f"%{company_name}%")
            
            if job_title: 
                where_conditions.append("j.JobTitle LIKE %s")
                final_params.append(f"%{job_title}%")
            
            # --- FIX: Handle Location List or String ---
            if location:
                if isinstance(location, str): location = [location]
                # Create multiple OR conditions for locations to handle lists like ['Noida']
                loc_placeholders = ' OR '.join(['j.Location LIKE %s' for _ in location])
                where_conditions.append(f"({loc_placeholders})")
                final_params.extend([f"%{l}%" for l in location])
            
            if where_conditions: where_sql_part = " WHERE " + " AND ".join(where_conditions)
            
            # 2. Build HAVING
            having_sql_part = ""
            if skills:
                if isinstance(skills, str): skills = [skills]
                h_parts = []
                for skill in skills:
                    # SUM trick: filters jobs matching the skill without hiding other skills in the results
                    h_parts.append("SUM(CASE WHEN s.SkillName LIKE %s THEN 1 ELSE 0 END) > 0")
                    final_params.append(skill)
                having_sql_part = " HAVING " + " AND ".join(h_parts)

            # 3. Assemble
            final_sql = """
                SELECT j.JobTitle, j.CompanyName, j.Location, GROUP_CONCAT(DISTINCT s.SkillName SEPARATOR ', ') AS RequiredSkills 
                FROM Jobs j 
                LEFT JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID 
                LEFT JOIN Skills s ON jsm.SkillID = s.SkillID
            """ + where_sql_part + " GROUP BY j.JobID" + having_sql_part + " ORDER BY j.CompanyName, j.JobTitle;"

            df = pd.read_sql_query(final_sql, self.mysql_engine, params=tuple(final_params))
            if df.empty: return "No job openings found matching your criteria."
            return f"Found **{len(df)}** job openings:", df
        except Exception as e:
            print(f"Error in find_jobs: {e}")
            return "Error retrieving jobs."

    # --- HELPER FUNCTIONS ---
    def aggregate_profile_query(self, target_skill=None, company_name=None):
        """
        Fixed to handle multiple skills correctly via GROUP BY / HAVING.
        """
        if not target_skill and not company_name: return "Please specify a skill or company name."
        try:
            # We use a subquery to first find the matching Profiles, then count them.
            # This is cleaner for multi-skill AND logic.
            
            sql = """
                SELECT COUNT(*) FROM (
                    SELECT p.ProfileID
                    FROM Profiles p
                    JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID 
                    JOIN Skills s ON psm.SkillID = s.SkillID
            """
            
            params = []
            conditions = []
            having_clause = ""

            if target_skill:
                if isinstance(target_skill, str): target_skill = [target_skill]
                # Lowercase for case-insensitivity
                placeholders = ', '.join(['%s'] * len(target_skill))
                conditions.append(f"LOWER(s.SkillName) IN ({placeholders})")
                params.extend([ts.lower() for ts in target_skill])
                having_clause = "HAVING COUNT(DISTINCT LOWER(s.SkillName)) >= %s"

            if company_name: 
                conditions.append("p.CompanyName ILIKE %s")
                params.append(f"%{company_name}%")
                
            if conditions: sql += " WHERE " + " AND ".join(conditions)
            
            sql += " GROUP BY p.ProfileID"
            
            if having_clause: 
                sql += " " + having_clause
                params.append(len(target_skill))
            
            sql += ") as subquery"
            
            # Execute
            df = pd.read_sql_query(sql, self.pg_engine, params=tuple(params))
            count = df.iloc[0][0]
            
            criteria = []
            if target_skill: criteria.append(f"skills **{', '.join(target_skill)}**")
            if company_name: criteria.append(f"company **{company_name}**")
            return f"Found **{count}** candidates matching: {' and '.join(criteria)}."
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return "Error calculating aggregation."

    def get_closest_job_title_from_db(self, user_input_title):
        if not user_input_title: return None
        try:
            sql = "SELECT DISTINCT JobTitle FROM Jobs;"
            df = pd.read_sql_query(sql, self.mysql_engine)
            all_titles = df['JobTitle'].tolist()
            match = process.extractOne(user_input_title, all_titles, scorer=fuzz.token_sort_ratio, score_cutoff=50)
            if match: return match[0]
            return user_input_title
        except Exception: return user_input_title

    def find_eligible_candidates_for_job(self, target_job_title):
        if not target_job_title: return "Please specify a job title."
        real_job_title = self.get_closest_job_title_from_db(target_job_title)
        target_skills = self.get_all_skills_for_job_title(real_job_title)
        if not target_skills: return f"No jobs found for title '{target_job_title}'."
        matching_candidates_df = self.find_matching_candidates(target_skills, threshold=60)
        if matching_candidates_df.empty: return f"No candidates match 60% of skills for {target_job_title}."
        
        total = len(matching_candidates_df)
        prompt = f"""
        You are an HR recruiter. User asked for eligible candidates for '{target_job_title}'.
        Found {total} candidates.
        Summarize the finding (e.g. "Found {total} candidates...").
        Mention top 3 companies they work at.
        IMPORTANT: Do NOT generate a table.
        """
        try:
            summary = self.llm.generate_content(prompt).text
            return summary, matching_candidates_df
        except: return f"Found {total} candidates.", matching_candidates_df

    def find_matching_candidates(self, required_skills, threshold=0):
        if not required_skills: return pd.DataFrame()
        try:
            skill_placeholders = ', '.join(['%s'] * len(required_skills))
            # Fixed "Match %" syntax error
            sql = f"""
                SELECT p.FullName as "Candidate Name", p.Headline, p.YearsOfExperience, p.CompanyName, 
                STRING_AGG(s.SkillName, ', ') AS "Matched Skills", 
                (COUNT(DISTINCT s.SkillName) * 100.0 / %s) AS "Match Percentage"
                FROM Profiles p JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID
                JOIN Skills s ON psm.SkillID = s.SkillID
                WHERE s.SkillName IN ({skill_placeholders})
                GROUP BY p.ProfileID, p.FullName, p.Headline, p.YearsOfExperience, p.CompanyName
                HAVING (COUNT(DISTINCT s.SkillName) * 100.0 / %s) >= %s
                ORDER BY "Match Percentage" DESC LIMIT 50;
            """
            params = (len(required_skills),) + tuple(required_skills) + (len(required_skills), threshold)
            return pd.read_sql_query(sql, self.pg_engine, params=params)
        except: return pd.DataFrame()

    def get_all_skills_for_job_title(self, job_title):
        try:
            sql = "SELECT DISTINCT s.SkillName FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID JOIN Jobs j ON jsm.JobID = j.JobID WHERE j.JobTitle LIKE %s"
            df = pd.read_sql_query(sql, self.mysql_engine, params=(f"%{job_title}%",))
            return df['SkillName'].tolist()
        except: return []

    def find_eligible_jobs_for_user(self, user_name):
        if not user_name: return "Please specify your name."
        status, profile_data = self.get_user_profile(user_name)
        if status != "UNIQUE_MATCH": return f"Profile for '{user_name}' not found or ambiguous."
        # Unpack 6 elements including Location
        profile_id, full_name, headline, experience, company_name, location = profile_data
        
        user_skills = self.get_user_skills(profile_id)
        if not user_skills: return "You have no skills listed."
        matching_jobs = self.find_matching_jobs(user_skills, threshold=40)
        if matching_jobs.empty: return "No matching jobs found."
        return f"Found **{len(matching_jobs)}** jobs matching your skills:", matching_jobs

    def find_matching_jobs(self, user_skills, threshold=0):
        if not user_skills: return pd.DataFrame()
        try:
            skill_placeholders = ', '.join(['%s'] * len(user_skills))
            sql = f"""
                SELECT j.JobTitle, j.CompanyName, j.Location, 
                ROUND((SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) / COUNT(DISTINCT jsm.SkillID)) * 100, 2) AS MatchPercentage
                FROM Jobs j LEFT JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID LEFT JOIN Skills s ON jsm.SkillID = s.SkillID 
                GROUP BY j.JobID, j.JobTitle, j.CompanyName, j.Location
                HAVING SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) > 0 
                AND (SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) / COUNT(DISTINCT jsm.SkillID)) * 100 >= %s
                ORDER BY MatchPercentage DESC LIMIT 50;
            """
            params = tuple(user_skills) * 3 + (threshold,)
            return pd.read_sql_query(sql, self.mysql_engine, params=params)
        except: return pd.DataFrame()

    def get_user_profile(self, full_name):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ProfileID, FullName FROM Profiles;")
                    all_profiles = cur.fetchall()
            profile_choices = {name: pid for pid, name in all_profiles}
            best_matches = process.extract(full_name, profile_choices.keys(), scorer=fuzz.token_sort_ratio, limit=1)
            if not best_matches or best_matches[0][1] < 80: return "NO_MATCH", None
            pid = profile_choices[best_matches[0][0]]
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ProfileID, FullName, Headline, YearsOfExperience, CompanyName, Location FROM Profiles WHERE ProfileID = %s;", (pid,))
                    return "UNIQUE_MATCH", cur.fetchone()
        except: return "ERROR", None

    def get_user_skills(self, profile_id):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT s.SkillName FROM Skills s JOIN Profile_Skills_Mapping psm ON s.SkillID = psm.SkillID WHERE psm.ProfileID = %s;", (profile_id,))
                    return [row[0] for row in cur.fetchall()]
        except: return []
    
    def get_skills_for_user(self, user_name):
        if not user_name: return "Please specify a name."
        status, data = self.get_user_profile(user_name)
        if status != "UNIQUE_MATCH": return f"Profile for '{user_name}' not found."
        user_skills = self.get_user_skills(data[0])
        return f"Skills for **{data[1]}**:\n\n- {'\n- '.join(user_skills)}"

    def get_skills_for_job(self, job_title, company_name):
        if not job_title or not company_name: return "Specify job and company."
        skills = self.get_target_job_skills(job_title, company_name)
        if not skills: return "No data found."
        unique = set(s[0] for s in skills)
        mandatory = {s for s, i in skills if i == 'Mandatory'}
        preferred = unique - mandatory
        res = f"Skills for **{job_title}** at **{company_name}**:\n\n"
        if mandatory: res += f"**Mandatory:**\n- {'\n- '.join(sorted(list(mandatory)))}\n\n"
        if preferred: res += f"**Preferred:**\n- {'\n- '.join(sorted(list(preferred)))}"
        return res

    def get_target_job_skills(self, job_title, company_name):
        try:
            with mysql.connector.connect(**self.mysql_conn_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT s.SkillName, jsm.Importance FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID JOIN Jobs j ON jsm.JobID = j.JobID WHERE j.JobTitle LIKE %s AND j.CompanyName LIKE %s;", (f"%{job_title}%", f"%{company_name}%"))
                    return cursor.fetchall()
        except: return None
        
    def run_dynamic_query(self, analysis):
        user_name = analysis.get('user_name')
        target_job = analysis.get('target_job')
        company = analysis.get('company_name')
        if not all([user_name, target_job, company]): return "Need name, job, and company."
        status, data = self.get_user_profile(user_name)
        if status != "UNIQUE_MATCH": return f"Profile '{user_name}' not found."
        
        my_skills = self.get_user_skills(data[0])
        req_skills = self.get_target_job_skills(target_job, company)
        if not req_skills: return f"No data for '{target_job}' at '{company}'."
        
        mand_gap, pref_gap = self.analyze_skill_gap(my_skills, req_skills)
        return self.generate_learning_path(data, mand_gap, pref_gap)
        
    def analyze_skill_gap(self, current_skills, required_skills):
        current = set(current_skills)
        req_unique = {s[0] for s in required_skills}
        mand = {s[0] for s in required_skills if s[1] == 'Mandatory'}
        return sorted(list(mand - current)), sorted(list((req_unique - mand) - current))

    def generate_learning_path(self, profile_data, mandatory_gap, preferred_gap):
        if not mandatory_gap and not preferred_gap: return "You have all required skills!"
        prompt = f"""Career coach for {profile_data[1]}. Gaps: Mandatory: {mandatory_gap}; Preferred: {preferred_gap}. Recommend resources."""
        try: return self.llm.generate_content(prompt).text
        except: return "Error generating path."

    def get_top_skills(self, limit=5):
        try:
            sql = f"SELECT s.SkillName, COUNT(jsm.SkillID) as SkillCount FROM Job_Skills_Mapping jsm JOIN Skills s ON jsm.SkillID = s.SkillID GROUP BY s.SkillName ORDER BY SkillCount DESC LIMIT {limit};"
            return pd.read_sql_query(sql, self.mysql_engine)['SkillName'].tolist()
        except: return []

    def run_skill_forecast(self, company_name, target_job_title):
        if not company_name or not target_job_title: return "Need company and job title."
        try:
            check = pd.read_sql_query("SELECT COUNT(*) FROM Profiles WHERE CompanyName ILIKE %s", self.pg_engine, params=(f"%{company_name}%",))
            if check.iloc[0][0] == 0: return f"Company '{company_name}' not found."
            
            skills = self.get_target_job_skills(target_job_title, "%")
            if not skills: return f"No market data for '{target_job_title}'."
            target_skills = list(set([s[0] for s in skills]))
            
            internal = self.get_internal_skill_audit(company_name, target_skills)
            internal_json = internal.to_json(orient="records") if not internal.empty else "[]"
            
            external = self.get_external_market_analysis(target_job_title)
            
            prompt = f"""
            HR Strategy for {company_name}. Role: {target_job_title}.
            Internal Talent (JSON): {internal_json}
            External Market: {external['open_jobs_count']} open jobs, Top locs: {external['common_locations']}
            Write a 'Build vs Buy' recommendation.
            """
            return self.llm.generate_content(prompt).text
        except Exception as e: return f"Error: {e}"

    def get_internal_skill_audit(self, company_name, target_skills):
        try:
            placeholders = ', '.join(['%s'] * len(target_skills))
            sql = f"""
                SELECT p.FullName, STRING_AGG(s.SkillName, ', ') AS Skills, COUNT(s.SkillName) as SkillCount
                FROM Profiles p JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID JOIN Skills s ON psm.SkillID = s.SkillID
                WHERE p.CompanyName ILIKE %s AND s.SkillName IN ({placeholders})
                GROUP BY p.ProfileID, p.FullName HAVING COUNT(DISTINCT s.SkillName) >= 2
                ORDER BY SkillCount DESC LIMIT 10; 
            """
            return pd.read_sql_query(sql, self.pg_engine, params=(company_name,) + tuple(target_skills))
        except: return pd.DataFrame()

    def get_external_market_analysis(self, target_job_title):
        try:
            sql = "SELECT COUNT(DISTINCT j.JobID) AS JobCount, GROUP_CONCAT(DISTINCT j.Location SEPARATOR ', ') AS Locations FROM Jobs j WHERE j.JobTitle LIKE %s"
            df = pd.read_sql_query(sql, self.mysql_engine, params=(f"%{target_job_title}%",))
            if df.empty or df.iloc[0]['JobCount'] == 0: return {"open_jobs_count": 0, "common_locations": "N/A"}
            locs = df.iloc[0]['Locations'].split(', ') if df.iloc[0]['Locations'] else []
            top_locs = pd.Series(locs).value_counts().head(3).index.tolist() if locs else []
            return {"open_jobs_count": int(df.iloc[0]['JobCount']), "common_locations": ", ".join(top_locs)}
        except: return {"open_jobs_count": 0, "common_locations": "N/A"}

    # --- CRUD OPERATIONS ---
    def register_new_user(self, name, headline, experience, skills, company_name, location='Bangalore'):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO Profiles (FullName, Headline, YearsOfExperience, CompanyName, Location) VALUES (%s, %s, %s, %s, %s) RETURNING ProfileID;", (name, headline, experience, company_name, location))
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

    def update_profile(self, profile_id, name, headline, experience, skills, company_name, location=None):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    sql = "UPDATE Profiles SET FullName = %s, Headline = %s, YearsOfExperience = %s, CompanyName = %s"
                    params = [name, headline, experience, company_name]
                    if location:
                        sql += ", Location = %s"
                        params.append(location)
                    sql += " WHERE ProfileID = %s;"
                    params.append(profile_id)
                    
                    cur.execute(sql, tuple(params))
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

    def delete_profile(self, profile_id):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM Profiles WHERE ProfileID = %s;", (profile_id,))
            return True
        except Exception as e:
            print(f"Error in delete_profile: {e}")
            return False

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

    def get_all_profiles_data(self):
        try:
            sql = "SELECT p.ProfileID, p.FullName, p.Headline, p.YearsOfExperience, p.CompanyName, p.Location, STRING_AGG(s.SkillName, ', ') AS Skills FROM Profiles p LEFT JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID LEFT JOIN Skills s ON psm.SkillID = s.SkillID GROUP BY p.ProfileID, p.FullName, p.Headline, p.YearsOfExperience, p.CompanyName, p.Location ORDER BY p.FullName;"
            return pd.read_sql_query(sql, self.pg_engine)
        except Exception as e:
            print(f"Error fetching profiles data: {e}")
            return pd.DataFrame()
        
    def get_all_jobs_data_for_crud(self):
        try:
            sql = "SELECT j.JobID, j.JobTitle, j.CompanyName, j.Location, GROUP_CONCAT(s.SkillName SEPARATOR ', ') AS RequiredSkills FROM Jobs j LEFT JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID LEFT JOIN Skills s ON jsm.SkillID = s.SkillID GROUP BY j.JobID ORDER BY j.CompanyName, j.JobTitle;"
            return pd.read_sql_query(sql, self.mysql_engine)
        except Exception as e:
            print(f"Error fetching jobs data for CRUD: {e}")
            return pd.DataFrame()
    
    def get_all_skills(self):
        try:
             # Added alias to preserve Case Sensitivity in Pandas
             sql_pg = 'SELECT DISTINCT SkillName AS "SkillName" FROM Skills;'
             df = pd.read_sql_query(sql_pg, self.pg_engine)
             return sorted(df['SkillName'].tolist())
        except Exception as e: 
            print(f"Error getting skills: {e}")
            return []