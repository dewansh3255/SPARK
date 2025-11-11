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

class CareerNavigator:
    def __init__(self):
        # --- Connection Details for Distributed Setup ---
        pg_user = "dewansh"
        pg_pass = "password123"
        pg_host = "192.168.52.137"
        pg_db = "profile_db"

        mysql_user = "remote_user"
        mysql_pass = "password321"
        mysql_host = "192.168.52.116"
        mysql_db = "jobsdb"
        
        # Create SQLAlchemy Engines for Pandas
        pg_uri = f"postgresql://{pg_user}:{pg_pass}@{pg_host}/{pg_db}"
        mysql_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}"
        self.pg_engine = create_engine(pg_uri)
        self.mysql_engine = create_engine(mysql_uri)
        
        # Keep raw connection parameters for transactional queries
        self.pg_conn_params = { 'dbname': pg_db, 'user': pg_user, 'password': pg_pass, 'host': pg_host, 'port': '5432' }
        self.mysql_conn_params = { 'host': mysql_host, 'database': mysql_db, 'user': mysql_user, 'password': mysql_pass }

        # Configure LLM
        load_dotenv()
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
        - 'CAREER_PATH': User wants a skill gap analysis for a specific job. This is the intent if a user provides their name and asks "what skills do I need for...".
        - 'FIND_JOBS': User wants to find job listings with optional filters like company, location, or job title.
        - 'ELIGIBLE_JOBS': User wants to find jobs that match their own profile's skills.
        - 'SKILL_LOOKUP': User wants a simple list of skills for a job, without personalization. This is the intent for queries like "list skills for...".
        - 'USER_SKILL_LOOKUP': User wants to list the skills of a specific person.
        - 'UNKNOWN': The query is unclear.

        **Entities to Extract:**
        - 'user_name': The full name of a person mentioned.
        - 'company_name': The name of a company.
        - 'target_job': A specific job title.
        - 'location': A city or geographical area.

        **Instructions:**
        - Return ONLY a raw JSON object.
        - If an entity is not mentioned, its value must be null.

        **Examples:**
        - Query: "My name is Suhani and I want to get job in Google as data analyst. What skills do I need?" -> {{"intent": "CAREER_PATH", "user_name": "Suhani", "company_name": "Google", "target_job": "data analyst", "location": null}}
        - Query: "list all skills needed for data scientist in Google" -> {{"intent": "SKILL_LOOKUP", "user_name": null, "company_name": "Google", "target_job": "data scientist", "location": null}}
        - Query: "My name is Dewansh, what jobs am I eligible for?" -> {{"intent": "ELIGIBLE_JOBS", "user_name": "Dewansh", "company_name": null, "target_job": null, "location": null}}
        
        **User Query:** "{query}"

        **JSON Output:**
        """
        try:
            response = self.llm.generate_content(intent_prompt)
            analysis = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
            intent = analysis.get("intent")
            
            if intent == "CAREER_PATH": return self.run_dynamic_query(analysis)
            elif intent == "FIND_JOBS": return self.find_jobs_by_criteria(analysis.get("company_name"), analysis.get("target_job"), analysis.get("location"))
            elif intent == "ELIGIBLE_JOBS": return self.find_eligible_jobs_for_user(analysis.get("user_name"))
            elif intent == "SKILL_LOOKUP": return self.get_skills_for_job(analysis.get("target_job"), analysis.get("company_name"))
            elif intent == "USER_SKILL_LOOKUP": return self.get_skills_for_user(analysis.get("user_name"))
            else: return "Sorry, I can only answer questions about career paths, job eligibility, job listings, or skills for specific roles/users."
        except Exception as e:
            print(f"Error in general query execution: {e}")
            return "Sorry, I had trouble understanding your request."

    def find_eligible_jobs_for_user(self, user_name):
        """Handles ELIGIBLE_JOBS intent with improved logic."""
        if not user_name:
            return "Please specify your name so I can find your profile."
        
        status, profile_data = self.get_user_profile(user_name)
        if status == "AMBIGUOUS":
            return f"Multiple profiles found matching '{user_name}'. Please be more specific with the full name."
        if status != "UNIQUE_MATCH":
            return f"Could not find a unique profile for '{user_name}'. Please try again with a full name."

        user_skills = self.get_user_skills(profile_data[0])
        if not user_skills:
            return f"**{user_name}** has no skills listed. Cannot find eligible jobs."

        # Find jobs with at least a 40% skill match
        matching_jobs = self.find_matching_jobs(user_skills, threshold=40)
        
        if not matching_jobs.empty:
            response_text = f"Found **{len(matching_jobs)}** jobs that could be a good fit for **{user_name}** based on their skills:"
            return response_text, matching_jobs
        else:
            # If no jobs match, give a direct and helpful suggestion
            top_skills = self.get_top_skills(limit=3)
            return f"Sorry {user_name}, no jobs were found with a significant match to your current skills. The top 3 most in-demand skills right now are **{', '.join(top_skills)}**. Focusing on one of these could greatly improve your job prospects!"

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
            skill_placeholders = ', '.join(['%s'] * len(user_skills))
            sql = f"""SELECT j.JobTitle, j.CompanyName, j.Location, (SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) / COUNT(DISTINCT jsm.SkillID)) * 100 AS MatchPercentage FROM Jobs j LEFT JOIN Job_Skills_Mapping jsm ON j.JobID = jsm.JobID LEFT JOIN Skills s ON jsm.SkillID = s.SkillID GROUP BY j.JobID HAVING SUM(CASE WHEN s.SkillName IN ({skill_placeholders}) THEN 1 ELSE 0 END) > 0 ORDER BY MatchPercentage DESC LIMIT 50;"""
            params = user_skills * 2
            df = pd.read_sql_query(sql, self.mysql_engine, params=params)
            return df[df['MatchPercentage'] >= threshold]
        except Exception as e:
            print(f"Error finding matching jobs: {e}")
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
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ProfileID, FullName FROM Profiles;")
                    all_profiles = cur.fetchall()
            profile_choices = {name: pid for pid, name in all_profiles}
            best_matches = process.extract(full_name, profile_choices.keys(), scorer=fuzz.WRatio, limit=5, score_cutoff=80)
            if not best_matches: return "NO_MATCH", None
            top_match_name, top_match_score, _ = best_matches[0]
            if top_match_score > 95: matched_profile_ids = [profile_choices[top_match_name]]
            else: matched_profile_ids = [profile_choices[match[0]] for match in best_matches]
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    query_placeholders = ','.join(['%s'] * len(matched_profile_ids))
                    cur.execute(f"SELECT * FROM Profiles WHERE ProfileID IN ({query_placeholders});", tuple(matched_profile_ids))
                    matched_profiles_data = cur.fetchall()
            if len(matched_profiles_data) == 1: return "UNIQUE_MATCH", matched_profiles_data[0]
            else: return "AMBIGUOUS", matched_profiles_data
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
                    cursor.execute("SELECT s.SkillName, jsm.Importance FROM Skills s JOIN Job_Skills_Mapping jsm ON s.SkillID = jsm.SkillID JOIN Jobs j ON jsm.JobID = j.JobID WHERE j.JobTitle = %s AND j.CompanyName = %s;", (job_title, company_name))
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

    # --- Profile CRUD Operations ---
    def register_new_user(self, name, headline, experience, skills):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO Profiles (FullName, Headline, YearsOfExperience) VALUES (%s, %s, %s) RETURNING ProfileID;", (name, headline, experience))
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

    def update_profile(self, profile_id, name, headline, experience, skills):
        try:
            with psycopg2.connect(**self.pg_conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE Profiles SET FullName = %s, Headline = %s, YearsOfExperience = %s WHERE ProfileID = %s;", (name, headline, experience, profile_id))
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
            return True
        except Exception as e:
            print(f"Error in delete_job: {e}")
            return False
                
    # --- Data Fetching for Display ---
    def get_all_profiles_data(self):
        try:
            sql = "SELECT p.ProfileID, p.FullName, p.Headline, p.YearsOfExperience, STRING_AGG(s.SkillName, ', ') AS Skills FROM Profiles p LEFT JOIN Profile_Skills_Mapping psm ON p.ProfileID = psm.ProfileID LEFT JOIN Skills s ON psm.SkillID = s.SkillID GROUP BY p.ProfileID ORDER BY p.FullName;"
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