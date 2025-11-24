# db_setup.py
import psycopg2
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from faker import Faker
import random

# --- CONFIGURATION ---
NUM_PROFILES = 200
NUM_JOBS = 200

fake = Faker('en_IN') # Indian locale

# --- DATA CORPUS ---
SKILLS_LIST = [
    'Python', 'Java', 'Go', 'JavaScript', 'C++', 'C#', 'TypeScript', 'PHP', 'Ruby', 'Swift', 'Kotlin', 'R', 'Scala',
    'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot', 'ASP.NET', 'HTML5', 'CSS3',
    'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'SQL Server', 'Oracle', 'Redis', 'Cassandra', 'Elasticsearch',
    'Machine Learning', 'Statistics', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy', 'NLP', 'Computer Vision',
    'Data Warehousing', 'ETL', 'Cloud (AWS)', 'Cloud (Azure)', 'Cloud (GCP)', 'Docker', 'Kubernetes', 'CI/CD', 'Terraform', 'Ansible', 'Jenkins',
    'Git', 'PowerBI', 'Tableau', 'Qlik Sense', 'Looker', 'Data Visualization', 'Agile', 'Scrum', 'JIRA', 'Communication', 'Project Management',
    'Stakeholder Management', 'Kanban', 'Microservices', 'REST APIs', 'System Design', 'Object-Oriented Programming (OOP)', 'Data Structures & Algorithms'
]

COMPANIES_LIST = [
    'Google', 'Microsoft', 'Amazon', 'Salesforce', 'Adobe', 'Oracle', 'SAP', 'VMware', 'Intel', 'IBM',
    'TCS', 'Infosys', 'Wipro', 'HCL Tech', 'Tech Mahindra', 'Capgemini', 'Accenture', 'Deloitte',
    'Flipkart', 'Swiggy', 'Zomato', 'Paytm', 'Ola Cabs', 'BYJU\'S', 'Zerodha', 'PhonePe', 'Razorpay',
    'Freshworks', 'Zoho', 'Myntra', 'Publicis Sapient', 'IIIT-D', 'DTU', 'IIT Delhi', 'NSUT'
]

CITIES_LIST = ['Bangalore', 'Pune', 'Hyderabad', 'Mumbai', 'Chennai', 'Delhi', 'Gurgaon', 'Noida', 'Kolkata']

ARCHETYPES = {
    "data_scientist": {
        "titles": ["Data Scientist", "AI/ML Engineer", "Machine Learning Scientist"],
        "core": ["Python", "SQL", "Machine Learning", "Pandas", "Scikit-learn", "Statistics"],
        "secondary": ["TensorFlow", "PyTorch", "NLP", "Deep Learning", "Tableau", "PowerBI"]
    },
    "backend_dev": {
        "titles": ["Backend Engineer", "Software Engineer (Backend)", "API Developer"],
        "core": ["Python", "Java", "Go", "Node.js", "SQL", "PostgreSQL", "REST APIs", "System Design"],
        "secondary": ["Django", "Flask", "Spring Boot", "Microservices", "Docker", "Kubernetes", "Redis", "NoSQL"]
    },
    "frontend_dev": {
        "titles": ["Frontend Developer", "UI Engineer", "Web Developer"],
        "core": ["JavaScript", "React", "HTML5", "CSS3", "Git"],
        "secondary": ["Angular", "Vue.js", "TypeScript", "Node.js", "REST APIs", "CI/CD"]
    },
    "cloud_engineer": {
        "titles": ["Cloud Engineer", "DevOps Engineer", "SRE"],
        "core": ["Cloud (AWS)", "Docker", "Kubernetes", "CI/CD", "Terraform", "Jenkins", "Ansible"],
        "secondary": ["Cloud (Azure)", "Cloud (GCP)", "Go", "Python", "System Design", "Microservices"]
    },
    "data_analyst": {
        "titles": ["Data Analyst", "Business Intelligence Analyst", "BI Developer"],
        "core": ["SQL", "Tableau", "PowerBI", "Data Visualization", "Pandas", "Communication"],
        "secondary": ["Python", "R", "Statistics", "ETL", "Data Warehousing", "Scikit-learn"]
    },
    "manager": {
        "titles": ["Product Manager", "Engineering Manager", "Project Manager"],
        "core": ["Agile", "Scrum", "JIRA", "Project Management", "Communication", "Stakeholder Management"],
        "secondary": ["System Design", "Kanban", "CI/CD", "Git"]
    }
}

def get_random_skills(archetype_key, num_core, num_secondary, num_random):
    archetype = ARCHETYPES[archetype_key]
    skills = set()
    if len(archetype["core"]) >= num_core:
        skills.update(random.sample(archetype["core"], num_core))
    if len(archetype["secondary"]) >= num_secondary:
        skills.update(random.sample(archetype["secondary"], num_secondary))
    remaining_skills = [s for s in SKILLS_LIST if s not in skills]
    if len(remaining_skills) >= num_random:
        skills.update(random.sample(remaining_skills, num_random))
    return list(skills)

def setup_postgres_db():
    conn = None
    try:
        load_dotenv()
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASS"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        conn.autocommit = True
        cur = conn.cursor()
        print("Connected to PostgreSQL.")

        cur.execute("DROP TABLE IF EXISTS Profile_Skills_Mapping CASCADE;")
        cur.execute("DROP TABLE IF EXISTS Profiles CASCADE;")
        cur.execute("DROP TABLE IF EXISTS Skills CASCADE;")
        print("Dropped old PostgreSQL tables.")

        cur.execute("""
            CREATE TABLE Skills (
                SkillID SERIAL PRIMARY KEY,
                SkillName VARCHAR(100) NOT NULL UNIQUE
            );
        """)
        # --- UPDATE: Added Location Column ---
        cur.execute("""
            CREATE TABLE Profiles (
                ProfileID SERIAL PRIMARY KEY,
                FullName VARCHAR(100) NOT NULL,
                Headline VARCHAR(255),
                YearsOfExperience INT,
                CompanyName VARCHAR(100),
                Location VARCHAR(100)
            );
        """)
        cur.execute("""
            CREATE TABLE Profile_Skills_Mapping (
                ProfileID INT REFERENCES Profiles(ProfileID) ON DELETE CASCADE,
                SkillID INT REFERENCES Skills(SkillID) ON DELETE CASCADE,
                PRIMARY KEY (ProfileID, SkillID)
            );
        """)
        print("Created new PostgreSQL tables.")

        skill_name_to_id = {}
        for skill in SKILLS_LIST:
            cur.execute("INSERT INTO Skills (SkillName) VALUES (%s) RETURNING SkillID;", (skill,))
            skill_id = cur.fetchone()[0]
            skill_name_to_id[skill] = skill_id

        print(f"Generating {NUM_PROFILES} profiles...")
        for i in range(NUM_PROFILES):
            full_name = fake.name()
            archetype_key = random.choice(list(ARCHETYPES.keys()))
            headline = random.choice(ARCHETYPES[archetype_key]["titles"])
            experience = random.randint(0, 15)
            company = random.choice(COMPANIES_LIST)
            location = random.choice(CITIES_LIST) # --- Random City ---

            # --- UPDATE: Inserting Location ---
            cur.execute(
                "INSERT INTO Profiles (FullName, Headline, YearsOfExperience, CompanyName, Location) VALUES (%s, %s, %s, %s, %s) RETURNING ProfileID;",
                (full_name, headline, experience, company, location)
            )
            profile_id = cur.fetchone()[0]

            num_core = random.randint(3, len(ARCHETYPES[archetype_key]["core"]))
            num_secondary = random.randint(2, len(ARCHETYPES[archetype_key]["secondary"]))
            num_random = random.randint(0, 3)
            profile_skills = get_random_skills(archetype_key, num_core, num_secondary, num_random)

            for skill_name in profile_skills:
                if skill_name in skill_name_to_id:
                    skill_id = skill_name_to_id[skill_name]
                    cur.execute(
                        "INSERT INTO Profile_Skills_Mapping (ProfileID, SkillID) VALUES (%s, %s);",
                        (profile_id, skill_id)
                    )
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{NUM_PROFILES} profiles created.")

        print("PostgreSQL setup successful. ✅")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error in PostgreSQL setup: {error}")
    finally:
        if conn is not None:
            conn.close()

def setup_mysql_db():
    conn = None
    try:
        load_dotenv()
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            database=os.getenv("MYSQL_DB"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASS"),
            port=os.getenv("MYSQL_PORT")
        )
        cur = conn.cursor()
        print("\nConnected to MySQL.")

        cur.execute("SET FOREIGN_KEY_CHECKS=0;")
        cur.execute("DROP TABLE IF EXISTS Job_Skills_Mapping;")
        cur.execute("DROP TABLE IF EXISTS Jobs;")
        cur.execute("DROP TABLE IF EXISTS Skills;")
        cur.execute("SET FOREIGN_KEY_CHECKS=1;")
        print("Dropped old MySQL tables.")

        cur.execute("""
            CREATE TABLE Skills (
                SkillID INT AUTO_INCREMENT PRIMARY KEY,
                SkillName VARCHAR(100) NOT NULL UNIQUE
            ) ENGINE=InnoDB;
        """)
        cur.execute("""
            CREATE TABLE Jobs (
                JobID INT AUTO_INCREMENT PRIMARY KEY,
                JobTitle VARCHAR(100) NOT NULL,
                CompanyName VARCHAR(100),
                Location VARCHAR(100)
            ) ENGINE=InnoDB;
        """)
        cur.execute("""
            CREATE TABLE Job_Skills_Mapping (
                MapID INT AUTO_INCREMENT PRIMARY KEY,
                JobID INT,
                SkillID INT,
                Importance ENUM('Preferred', 'Mandatory') NOT NULL DEFAULT 'Preferred',
                FOREIGN KEY (JobID) REFERENCES Jobs(JobID) ON DELETE CASCADE,
                FOREIGN KEY (SkillID) REFERENCES Skills(SkillID) ON DELETE CASCADE
            ) ENGINE=InnoDB;
        """)
        print("Created new MySQL tables.")

        skill_name_to_id = {}
        for skill in SKILLS_LIST:
            cur.execute("INSERT INTO Skills (SkillName) VALUES (%s);", (skill,))
            skill_id = cur.lastrowid
            skill_name_to_id[skill] = skill_id
        conn.commit()

        print(f"Generating {NUM_JOBS} jobs...")
        for i in range(NUM_JOBS):
            archetype_key = random.choice(list(ARCHETYPES.keys()))
            title = random.choice(ARCHETYPES[archetype_key]["titles"])
            company = random.choice(COMPANIES_LIST)
            location = random.choice(CITIES_LIST)

            cur.execute(
                "INSERT INTO Jobs (JobTitle, CompanyName, Location) VALUES (%s, %s, %s);",
                (title, company, location)
            )
            job_id = cur.lastrowid

            archetype = ARCHETYPES[archetype_key]
            job_skills = set()
            num_mandatory = random.randint(2, len(archetype["core"]))
            mandatory_skills = random.sample(archetype["core"], num_mandatory)
            job_skills.update(mandatory_skills)
            num_preferred = random.randint(1, len(archetype["secondary"]))
            preferred_skills = random.sample(archetype["secondary"], num_preferred)
            job_skills.update(preferred_skills)

            for skill_name in job_skills:
                if skill_name in skill_name_to_id:
                    skill_id = skill_name_to_id[skill_name]
                    importance = "Mandatory" if skill_name in mandatory_skills else "Preferred"
                    cur.execute(
                        "INSERT INTO Job_Skills_Mapping (JobID, SkillID, Importance) VALUES (%s, %s, %s);",
                        (job_id, skill_id, importance)
                    )
            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  ... {i+1}/{NUM_JOBS} jobs created.")

        conn.commit()
        print("MySQL setup successful. ✅")

    except Error as error:
        print(f"Error in MySQL setup: {error}")
    finally:
        if conn is not None and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    print("Starting database setup...")
    setup_postgres_db()
    setup_mysql_db()
    print("\n✅ All databases are set up with Location support!")