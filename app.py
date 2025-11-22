import streamlit as st
import pandas as pd
from main import CareerNavigator

st.set_page_config(layout="wide")

@st.cache_resource
def load_navigator():
    return CareerNavigator()
navigator = load_navigator()

st.title("🚀 SPARK: Your AI Career Navigator (Generalized System)")

tab1, tab2 = st.tabs(["**AI Navigator**", "**Data Management**"])

# ==============================================================================
# --- TAB 1: The AI Conversational Interface (Now Fully Generalized) ---
# ==============================================================================
with tab1:
    st.header("👤 General Query Chat")
    st.write("Ask any question about profiles or jobs, and the AI will generate and execute the necessary database query dynamically.")
    st.write("Examples: 'What jobs are available at Google in Pune?' or 'List the skills of Hritik Shanker.'")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], tuple):
                st.markdown(message["content"][0])
                st.dataframe(message["content"][1], width='stretch')
            else:
                st.markdown(str(message["content"]))

    # User input chat bar
    if prompt := st.chat_input("What would you like to know?"):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = navigator.execute_general_query(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                if isinstance(response, tuple):
                    st.markdown(response[0])
                    st.dataframe(response[1], width='stretch')
                else:
                    st.markdown(str(response))
        
        # Rerun to ensure layout updates correctly after response
        st.rerun()

# ==============================================================================
# --- TAB 2: The Data Management Dashboard ---
# ==============================================================================
with tab2:
    st.header("🗂️ Data Management Dashboard")
    st.write("Perform CRUD (Create, Read, Update, Delete) operations on your databases.")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    @st.cache_data
    def load_data():
        # Updated to get all_skills_list dynamically from the navigator
        return navigator.get_all_profiles_data(), navigator.get_all_jobs_data_for_crud(), navigator.get_all_skills()
    
    profiles_df, jobs_df, all_skills_list = load_data()
        
    st.markdown("---")
    
    st.subheader("👤 Profiles Database (PostgreSQL)")
    # This dataframe will now correctly show the 'companyname' column
    st.dataframe(profiles_df, width='stretch')
    
    col1, col2, col3 = st.columns(3)

    with col1.expander("➕ Add New Profile"):
        with st.form("add_profile_form", clear_on_submit=True):
            p_name = st.text_input("Full Name*")
            p_headline = st.text_input("Headline")
            p_company = st.text_input("Company Name*")
            p_exp = st.number_input("Years of Experience", 0, 50)
            p_skills = st.multiselect("Skills*", options=all_skills_list, key="add_p_skills")
            if st.form_submit_button("Add Profile"):
                if p_name and p_skills and p_company:
                    if navigator.register_new_user(p_name, p_headline, p_exp, p_skills, p_company): 
                        st.success("Profile added!")
                    else: st.error("Failed to add profile.")
                else: st.warning("Name, Company, and Skills are required.")

    with col2.expander("✏️ Edit Profile"):
        if not profiles_df.empty:
            # Use 'fullname' from the dataframe
            profile_to_edit_name = st.selectbox("Select Profile to Edit", options=profiles_df['fullname'])
            profile_details = profiles_df[profiles_df['fullname'] == profile_to_edit_name].iloc[0]
            with st.form("edit_profile_form"):
                up_name = st.text_input("Full Name*", value=profile_details['fullname'])
                up_headline = st.text_input("Headline", value=profile_details['headline'])
                # Handle companyname, which might be None or not exist in old data
                company_val = str(profile_details.get('companyname', ''))
                up_company = st.text_input("Company Name*", value=company_val) # <-- ADDED
                up_exp = st.number_input("Years of Experience", 0, 50, value=int(profile_details['yearsofexperience']))
                current_skills = [s.strip() for s in profile_details['skills'].split(',') if s.strip()] if pd.notna(profile_details['skills']) else []
                up_skills = st.multiselect("Skills*", options=all_skills_list, default=current_skills, key="edit_p_skills")
                if st.form_submit_button("Update Profile"):
                    if up_name and up_skills and up_company: # <-- ADDED up_company check
                        # Added up_company to the function call
                        if navigator.update_profile(int(profile_details['profileid']), up_name, up_headline, up_exp, up_skills, up_company): 
                            st.success("Profile updated!")
                        else: st.error("Failed to update profile.")
                    else: st.warning("Name, Company, and Skills are required.")

    with col3.expander("🗑️ Delete Profile"):
        if not profiles_df.empty:
            # Use 'fullname' from the dataframe
            profile_to_delete_name = st.selectbox("Select Profile to Delete", options=profiles_df['fullname'], key="delete_p_select")
            if st.button("Delete Profile"):
                profile_id_to_delete = int(profiles_df[profiles_df['fullname'] == profile_to_delete_name]['profileid'].iloc[0])
                if navigator.delete_profile(profile_id_to_delete): st.success("Profile deleted!")
                else: st.error("Failed to delete profile.")

    st.markdown("---")

    st.subheader("💼 Jobs Database (MySQL)")
    st.dataframe(jobs_df, width='stretch')
    
    col4, col5, col6 = st.columns(3)

    with col4.expander("➕ Add New Job"):
        with st.form("add_job_form", clear_on_submit=True):
            j_title = st.text_input("Job Title*")
            j_company = st.text_input("Company Name*")
            j_location = st.text_input("Location")
            j_skills = st.multiselect("Required Skills*", options=all_skills_list, key="add_j_skills")
            importance_map = {}
            if j_skills:
                st.write("**Skill Importance:**")
                for skill in j_skills:
                    importance_map[skill] = st.radio(f"{skill}:", ('Preferred', 'Mandatory'), horizontal=True, key=f"imp_{skill}")
            if st.form_submit_button("Add Job"):
                if j_title and j_company and j_skills:
                    if navigator.add_job(j_title, j_company, j_location, j_skills, importance_map): st.success("Job added!")
                    else: st.error("Failed to add job.")
                else: st.warning("Title, company, and skills are required.")

    with col5.expander("✏️ Edit Job"):
        if not jobs_df.empty:
            job_to_edit_id = st.selectbox("Select Job to Edit (by ID)", options=jobs_df['JobID'])
            job_details = jobs_df[jobs_df['JobID'] == job_to_edit_id].iloc[0]
            with st.form("edit_job_form"):
                uj_title = st.text_input("Job Title*", value=job_details['JobTitle'])
                uj_company = st.text_input("Company Name*", value=job_details['CompanyName'])
                uj_location = st.text_input("Location", value=job_details['Location'])
                current_skills = [s.strip() for s in job_details['RequiredSkills'].split(',') if s.strip()] if pd.notna(job_details['RequiredSkills']) else []
                uj_skills = st.multiselect("Skills*", options=all_skills_list, default=current_skills, key="edit_j_skills")
                uj_importance_map = {}
                if uj_skills:
                    st.write("**Skill Importance:**")
                    for skill in uj_skills:
                        uj_importance_map[skill] = st.radio(f"{skill}:", ('Preferred', 'Mandatory'), horizontal=True, key=f"edit_imp_{skill}")
                if st.form_submit_button("Update Job"):
                    if uj_title and uj_company and uj_skills:
                        if navigator.update_job(int(job_details['JobID']), uj_title, uj_company, uj_location, uj_skills, uj_importance_map): st.success("Job updated!")
                        else: st.error("Failed to update job.")
                    else: st.warning("Title, company, and skills are required.")

    with col6.expander("🗑️ Delete Job"):
        if not jobs_df.empty:
            job_to_delete_id = st.selectbox("Select Job to Delete (by ID)", options=jobs_df['JobID'], key="delete_j_select")
            if st.button("Delete Job"):
                if navigator.delete_job(int(job_to_delete_id)): st.success("Job deleted!")
                else: st.error("Failed to delete job.")