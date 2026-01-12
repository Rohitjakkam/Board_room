import streamlit as st
import google.generativeai as genai
import json
import PyPDF2
import os
from datetime import datetime
from typing import Dict, Optional

# Configure Gemini
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))

# Data storage directory
DATA_DIR = "extracted_data"

def ensure_data_dir():
    """Ensure the data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def extract_pdf_with_gemini(pdf_file) -> str:
    """Use Gemini's native PDF processing"""
    try:
        pdf_file.seek(0)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        pdf_file.seek(0)
        uploaded_file = genai.upload_file(pdf_file, mime_type='application/pdf')

        prompt = """Extract ALL text content from this PDF document.
        Include everything: headers, body text, tables, numbers, charts, footnotes.
        Be extremely thorough."""

        response = model.generate_content([uploaded_file, prompt])
        return response.text
    except Exception as e:
        st.error(f"Gemini extraction error: {e}")
        return extract_pdf_with_pypdf2(pdf_file)

def extract_pdf_with_pypdf2(pdf_file) -> str:
    """Fallback PDF extraction"""
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text += f"\n{'='*50}\nPAGE {i+1}\n{'='*50}\n{page_text}\n"
        return text if text.strip() else ""
    except Exception as e:
        st.error(f"PyPDF2 error: {e}")
        return ""

def parse_module_content(pdf_text: str) -> Dict:
    """Parse module/course content"""
    if not pdf_text or len(pdf_text.strip()) < 100:
        raise ValueError("Module PDF text is empty or too short")

    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    prompt = f"""Analyze this course/module document thoroughly.

DOCUMENT ({len(pdf_text)} chars):
{pdf_text}

Extract ALL educational content:

1. Module name and subject area
2. Learning objectives
3. Key topics (10-30) with descriptions, principles, formulas, examples
4. Frameworks and models with components and applications
5. Key terms (20-50) with definitions
6. Assessment criteria

Return ONLY valid JSON:
{{
    "module_name": "Exact course name",
    "subject_area": "Finance/Marketing/Operations/Strategy/HR/Economics",
    "learning_objectives": ["Objective 1", "Objective 2"],
    "overview": "2-3 sentence overview",
    "topics": [
        {{
            "name": "Topic name",
            "description": "What this covers",
            "key_principles": ["Principle 1", "Principle 2"],
            "formulas": ["Formula 1"],
            "application": "When/how to use",
            "examples": ["Example 1"]
        }}
    ],
    "frameworks": [
        {{
            "name": "Framework name",
            "description": "What it does",
            "components": ["Component 1", "Component 2"],
            "application_scenario": "When to use"
        }}
    ],
    "key_terms": {{"term1": "definition", "term2": "definition"}},
    "assessment_criteria": ["Criterion 1", "Criterion 2"]
}}

Extract minimum 10 topics, 20 terms. Return ONLY JSON, no markdown."""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip().replace('```json', '').replace('```', '').strip()

        if not result_text.startswith('{'):
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result_text = result_text[json_start:json_end]

        return json.loads(result_text)
    except Exception as e:
        st.error(f"Module parsing error: {e}")
        raise

def parse_company_data(pdf_text: str) -> Dict:
    """Parse company data"""
    if not pdf_text or len(pdf_text.strip()) < 100:
        raise ValueError("Company PDF text is empty or too short")

    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    prompt = f"""Analyze this company document thoroughly.

DOCUMENT ({len(pdf_text)} chars):
{pdf_text}

Extract ALL company information:

1. Company name (exact)
2. Company overview (4-5 sentences)
3. Metrics (20-50): financial, operational, employee, market metrics
4. Leadership team (5-15 people with names, roles, personality traits)
5. Current problems/challenges (5-10)
6. Initial business situation

Return ONLY valid JSON:
{{
    "company_name": "Exact company name",
    "company_overview": "Detailed overview",
    "metrics": {{
        "revenue_total": {{"value": 500, "unit": "$M", "description": "Total revenue"}},
        "profit_margin": {{"value": 15, "unit": "%", "description": "Net profit margin"}},
        "employee_count": {{"value": 1200, "unit": "employees", "description": "Total workforce"}}
    }},
    "board_members": [
        {{"name": "Full Name", "role": "Complete Title", "personality": "Detailed personality"}},
        {{"name": "Full Name", "role": "Complete Title", "personality": "Detailed personality"}}
    ],
    "current_problems": ["Specific problem 1", "Specific problem 2"],
    "initial_scenario": "Current business situation"
}}

Extract 20-50 metrics, 5-15 board members, 5-10 problems. Return ONLY JSON."""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip().replace('```json', '').replace('```', '').strip()

        if not result_text.startswith('{'):
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result_text = result_text[json_start:json_end]

        return json.loads(result_text)
    except Exception as e:
        st.error(f"Company parsing error: {e}")
        raise

def save_extracted_data(company_data: Dict, module_data: Dict, session_name: str) -> str:
    """Save extracted data to JSON file for persistence"""
    ensure_data_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_session_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_name)
    filename = f"{safe_session_name}_{timestamp}.json"
    filepath = os.path.join(DATA_DIR, filename)

    data = {
        "session_name": session_name,
        "created_at": datetime.now().isoformat(),
        "company_data": company_data,
        "module_data": module_data,
        "status": "ready_for_simulation"
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath

def load_extracted_data(filepath: str) -> Optional[Dict]:
    """Load previously extracted data"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def list_saved_sessions() -> list:
    """List all saved session files"""
    ensure_data_dir()
    sessions = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        "filename": filename,
                        "filepath": filepath,
                        "session_name": data.get("session_name", "Unknown"),
                        "created_at": data.get("created_at", "Unknown"),
                        "company_name": data.get("company_data", {}).get("company_name", "Unknown"),
                        "module_name": data.get("module_data", {}).get("module_name", "Unknown")
                    })
            except:
                continue
    return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)

def delete_session(filepath: str) -> bool:
    """Delete a saved session"""
    try:
        os.remove(filepath)
        return True
    except Exception as e:
        st.error(f"Error deleting session: {e}")
        return False

# Streamlit App UI
def main():
    st.set_page_config(
        page_title="Board Meeting Simulation - Data Collection",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Board Meeting Simulation - Data Collection")
    st.markdown("Upload your company and module PDFs to prepare for the simulation.")

    # Initialize session state
    if 'company_data' not in st.session_state:
        st.session_state.company_data = None
    if 'module_data' not in st.session_state:
        st.session_state.module_data = None
    if 'company_text' not in st.session_state:
        st.session_state.company_text = None
    if 'module_text' not in st.session_state:
        st.session_state.module_text = None
    if 'extraction_complete' not in st.session_state:
        st.session_state.extraction_complete = False

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Extract", "💾 Saved Sessions", "🔍 Audit Data", "ℹ️ Help"])

    with tab1:
        st.header("Step 1: Upload PDF Documents")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏢 Company Document")
            company_file = st.file_uploader(
                "Upload company PDF (annual report, case study, etc.)",
                type=['pdf'],
                key="company_upload"
            )

            if company_file:
                st.success(f"Uploaded: {company_file.name}")

                if st.button("Extract Company Data", key="extract_company"):
                    with st.spinner("Extracting company information..."):
                        company_text = extract_pdf_with_gemini(company_file)
                        if company_text:
                            st.session_state.company_text = company_text
                            st.info(f"Extracted {len(company_text)} characters from PDF")

                            with st.spinner("Parsing company data with AI..."):
                                try:
                                    company_data = parse_company_data(company_text)
                                    st.session_state.company_data = company_data
                                    st.success("Company data parsed successfully!")
                                except Exception as e:
                                    st.error(f"Failed to parse company data: {e}")

        with col2:
            st.subheader("📚 Module Document")
            module_file = st.file_uploader(
                "Upload module/course PDF",
                type=['pdf'],
                key="module_upload"
            )

            if module_file:
                st.success(f"Uploaded: {module_file.name}")

                if st.button("Extract Module Data", key="extract_module"):
                    with st.spinner("Extracting module content..."):
                        module_text = extract_pdf_with_gemini(module_file)
                        if module_text:
                            st.session_state.module_text = module_text
                            st.info(f"Extracted {len(module_text)} characters from PDF")

                            with st.spinner("Parsing module content with AI..."):
                                try:
                                    module_data = parse_module_content(module_text)
                                    st.session_state.module_data = module_data
                                    st.success("Module data parsed successfully!")
                                except Exception as e:
                                    st.error(f"Failed to parse module data: {e}")

        st.divider()

        # Preview extracted data
        st.header("Step 2: Review Extracted Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.company_data:
                st.subheader("🏢 Company Data Preview")
                data = st.session_state.company_data

                st.markdown(f"**Company Name:** {data.get('company_name', 'N/A')}")
                st.markdown(f"**Overview:** {data.get('company_overview', 'N/A')[:200]}...")

                with st.expander("View Metrics"):
                    metrics = data.get('metrics', {})
                    for name, info in list(metrics.items())[:10]:
                        if isinstance(info, dict):
                            st.write(f"- {name}: {info.get('value', 'N/A')} {info.get('unit', '')}")
                        else:
                            st.write(f"- {name}: {info}")
                    if len(metrics) > 10:
                        st.info(f"... and {len(metrics) - 10} more metrics")

                with st.expander("View Board Members"):
                    for member in data.get('board_members', [])[:5]:
                        st.write(f"- **{member.get('name', 'N/A')}**: {member.get('role', 'N/A')}")
                    if len(data.get('board_members', [])) > 5:
                        st.info(f"... and {len(data.get('board_members', [])) - 5} more members")

                with st.expander("View Current Problems"):
                    for problem in data.get('current_problems', []):
                        st.write(f"- {problem}")
            else:
                st.info("Upload and extract company PDF to see preview")

        with col2:
            if st.session_state.module_data:
                st.subheader("📚 Module Data Preview")
                data = st.session_state.module_data

                st.markdown(f"**Module Name:** {data.get('module_name', 'N/A')}")
                st.markdown(f"**Subject Area:** {data.get('subject_area', 'N/A')}")
                st.markdown(f"**Overview:** {data.get('overview', 'N/A')[:200]}...")

                with st.expander("View Topics"):
                    for topic in data.get('topics', [])[:5]:
                        st.write(f"- **{topic.get('name', 'N/A')}**: {topic.get('description', 'N/A')[:100]}...")
                    if len(data.get('topics', [])) > 5:
                        st.info(f"... and {len(data.get('topics', [])) - 5} more topics")

                with st.expander("View Frameworks"):
                    for framework in data.get('frameworks', [])[:5]:
                        st.write(f"- **{framework.get('name', 'N/A')}**: {framework.get('description', 'N/A')[:100]}...")

                with st.expander("View Key Terms"):
                    terms = data.get('key_terms', {})
                    for term, definition in list(terms.items())[:10]:
                        st.write(f"- **{term}**: {definition[:80]}...")
                    if len(terms) > 10:
                        st.info(f"... and {len(terms) - 10} more terms")
            else:
                st.info("Upload and extract module PDF to see preview")

        st.divider()

        # Save data
        st.header("Step 3: Save Data for Simulation")

        if st.session_state.company_data and st.session_state.module_data:
            session_name = st.text_input(
                "Session Name",
                value=f"{st.session_state.company_data.get('company_name', 'Session')} - {st.session_state.module_data.get('module_name', 'Module')}",
                help="Give your session a memorable name"
            )

            if st.button("💾 Save Data for Simulation", type="primary"):
                with st.spinner("Saving data..."):
                    filepath = save_extracted_data(
                        st.session_state.company_data,
                        st.session_state.module_data,
                        session_name
                    )
                    st.session_state.extraction_complete = True
                    st.success(f"Data saved successfully!")
                    st.info(f"File location: `{filepath}`")
                    st.balloons()

                    st.markdown("---")
                    st.markdown("### Ready for Simulation!")
                    st.markdown("You can now start the simulation using the saved data.")
        else:
            missing = []
            if not st.session_state.company_data:
                missing.append("Company data")
            if not st.session_state.module_data:
                missing.append("Module data")
            st.warning(f"Please extract both documents first. Missing: {', '.join(missing)}")

    with tab2:
        st.header("💾 Saved Sessions")

        sessions = list_saved_sessions()

        if sessions:
            for session in sessions:
                with st.expander(f"📁 {session['session_name']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.markdown(f"**Company:** {session['company_name']}")
                        st.markdown(f"**Module:** {session['module_name']}")

                    with col2:
                        st.markdown(f"**Created:** {session['created_at'][:19].replace('T', ' ')}")
                        st.markdown(f"**File:** `{session['filename']}`")

                    with col3:
                        if st.button("🗑️ Delete", key=f"del_{session['filename']}"):
                            if delete_session(session['filepath']):
                                st.success("Deleted!")
                                st.rerun()

                        if st.button("📂 Load", key=f"load_{session['filename']}"):
                            data = load_extracted_data(session['filepath'])
                            if data:
                                st.session_state.company_data = data.get('company_data')
                                st.session_state.module_data = data.get('module_data')
                                st.success("Data loaded! Switch to 'Upload & Extract' tab to view.")
                                st.rerun()
        else:
            st.info("No saved sessions found. Upload and extract documents to create one.")

    with tab3:
        st.header("🔍 Audit Extracted Data")
        st.markdown("Review, edit, add, or remove extracted information before simulation.")

        # Initialize audit session state
        if 'audit_loaded_file' not in st.session_state:
            st.session_state.audit_loaded_file = None
        if 'audit_data' not in st.session_state:
            st.session_state.audit_data = None
        if 'audit_modified' not in st.session_state:
            st.session_state.audit_modified = False

        # Load session for auditing
        st.subheader("📂 Select Session to Audit")
        sessions = list_saved_sessions()

        if not sessions:
            st.warning("No saved sessions found. Please extract and save data first.")
        else:
            session_options = {s['session_name']: s['filepath'] for s in sessions}
            selected_session = st.selectbox(
                "Choose a session to audit",
                options=list(session_options.keys()),
                key="audit_session_select"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔄 Load for Audit", type="primary"):
                    filepath = session_options[selected_session]
                    data = load_extracted_data(filepath)
                    if data:
                        st.session_state.audit_data = data
                        st.session_state.audit_loaded_file = filepath
                        st.session_state.audit_modified = False
                        st.success("Session loaded for auditing!")
                        st.rerun()

            with col2:
                if st.session_state.audit_modified:
                    st.warning("⚠️ You have unsaved changes!")

        if st.session_state.audit_data:
            st.divider()

            # Audit sub-tabs
            audit_tab1, audit_tab2 = st.tabs(["🏢 Company Data", "📚 Module Data"])

            # ============ COMPANY DATA AUDIT ============
            with audit_tab1:
                company_data = st.session_state.audit_data.get('company_data', {})

                # Basic Info Section
                st.subheader("📋 Basic Information")
                col1, col2 = st.columns(2)

                with col1:
                    new_company_name = st.text_input(
                        "Company Name",
                        value=company_data.get('company_name', ''),
                        key="audit_company_name"
                    )
                    if new_company_name != company_data.get('company_name', ''):
                        st.session_state.audit_data['company_data']['company_name'] = new_company_name
                        st.session_state.audit_modified = True

                with col2:
                    new_scenario = st.text_input(
                        "Initial Scenario",
                        value=company_data.get('initial_scenario', ''),
                        key="audit_initial_scenario"
                    )
                    if new_scenario != company_data.get('initial_scenario', ''):
                        st.session_state.audit_data['company_data']['initial_scenario'] = new_scenario
                        st.session_state.audit_modified = True

                new_overview = st.text_area(
                    "Company Overview",
                    value=company_data.get('company_overview', ''),
                    height=100,
                    key="audit_company_overview"
                )
                if new_overview != company_data.get('company_overview', ''):
                    st.session_state.audit_data['company_data']['company_overview'] = new_overview
                    st.session_state.audit_modified = True

                st.divider()

                # Metrics Section
                st.subheader("📊 Metrics")
                metrics = company_data.get('metrics', {})

                # Priority options for metrics
                priority_options = ["General", "High", "Medium"]

                # Add new metric
                with st.expander("➕ Add New Metric", expanded=False):
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                    with col1:
                        new_metric_name = st.text_input("Metric Name (key)", key="new_metric_name", placeholder="e.g., market_share")
                    with col2:
                        new_metric_value = st.number_input("Value", key="new_metric_value", value=0.0)
                    with col3:
                        new_metric_unit = st.text_input("Unit", key="new_metric_unit", placeholder="e.g., %")
                    with col4:
                        new_metric_desc = st.text_input("Description", key="new_metric_desc", placeholder="e.g., Market Share")
                    with col5:
                        new_metric_priority = st.selectbox("Priority", options=priority_options, key="new_metric_priority", index=0)

                    if st.button("Add Metric", key="add_metric_btn"):
                        if new_metric_name and new_metric_name not in metrics:
                            st.session_state.audit_data['company_data']['metrics'][new_metric_name] = {
                                "value": new_metric_value,
                                "unit": new_metric_unit,
                                "description": new_metric_desc,
                                "priority": new_metric_priority
                            }
                            st.session_state.audit_modified = True
                            st.success(f"Added metric: {new_metric_name}")
                            st.rerun()
                        elif new_metric_name in metrics:
                            st.error("Metric with this name already exists!")
                        else:
                            st.error("Please enter a metric name")

                # Edit/Remove existing metrics
                with st.expander(f"📝 Edit Metrics ({len(metrics)} total)", expanded=True):
                    metrics_to_remove = []
                    for metric_key, metric_info in metrics.items():
                        st.markdown(f"**{metric_key}**")
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 0.5])

                        if isinstance(metric_info, dict):
                            with col1:
                                new_val = st.number_input(
                                    "Value",
                                    value=float(metric_info.get('value', 0)),
                                    key=f"metric_val_{metric_key}",
                                    label_visibility="collapsed"
                                )
                                if new_val != metric_info.get('value'):
                                    st.session_state.audit_data['company_data']['metrics'][metric_key]['value'] = new_val
                                    st.session_state.audit_modified = True

                            with col2:
                                new_unit = st.text_input(
                                    "Unit",
                                    value=metric_info.get('unit', ''),
                                    key=f"metric_unit_{metric_key}",
                                    label_visibility="collapsed"
                                )
                                if new_unit != metric_info.get('unit'):
                                    st.session_state.audit_data['company_data']['metrics'][metric_key]['unit'] = new_unit
                                    st.session_state.audit_modified = True

                            with col3:
                                new_desc = st.text_input(
                                    "Description",
                                    value=metric_info.get('description', ''),
                                    key=f"metric_desc_{metric_key}",
                                    label_visibility="collapsed"
                                )
                                if new_desc != metric_info.get('description'):
                                    st.session_state.audit_data['company_data']['metrics'][metric_key]['description'] = new_desc
                                    st.session_state.audit_modified = True

                            with col4:
                                current_priority = metric_info.get('priority', 'General')
                                priority_idx = priority_options.index(current_priority) if current_priority in priority_options else 0
                                new_priority = st.selectbox(
                                    "Priority",
                                    options=priority_options,
                                    index=priority_idx,
                                    key=f"metric_priority_{metric_key}",
                                    label_visibility="collapsed"
                                )
                                if new_priority != current_priority:
                                    st.session_state.audit_data['company_data']['metrics'][metric_key]['priority'] = new_priority
                                    st.session_state.audit_modified = True

                            with col5:
                                if st.button("🗑️", key=f"del_metric_{metric_key}", help="Remove this metric"):
                                    metrics_to_remove.append(metric_key)

                        st.markdown("---")

                    # Process removals
                    for key in metrics_to_remove:
                        del st.session_state.audit_data['company_data']['metrics'][key]
                        st.session_state.audit_modified = True
                    if metrics_to_remove:
                        st.rerun()

                st.divider()

                # Board Members Section
                st.subheader("👥 Board Members")
                board_members = company_data.get('board_members', [])

                # Director type options
                director_types = ["Independent Director", "Executive Director", "Non-Executive Director", "Nominee Director", "Promoter Director"]

                # Add new board member
                with st.expander("➕ Add New Board Member", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_member_name = st.text_input("Name", key="new_member_name")
                    with col2:
                        new_member_role = st.text_input("Role/Title", key="new_member_role")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        new_member_expertise = st.text_input("Expertise/Domain", key="new_member_expertise", placeholder="e.g., Finance, Technology, Legal")
                    with col2:
                        new_member_tenure = st.number_input("Tenure (years)", key="new_member_tenure", min_value=0, max_value=50, value=0)
                    with col3:
                        new_member_type = st.selectbox("Director Type", options=director_types, key="new_member_type")

                    new_member_personality = st.text_area("Personality Description", key="new_member_personality", height=80)

                    if st.button("Add Board Member", key="add_member_btn"):
                        if new_member_name and new_member_role:
                            st.session_state.audit_data['company_data']['board_members'].append({
                                "name": new_member_name,
                                "role": new_member_role,
                                "expertise": new_member_expertise,
                                "tenure_years": new_member_tenure,
                                "director_type": new_member_type,
                                "personality": new_member_personality
                            })
                            st.session_state.audit_modified = True
                            st.success(f"Added board member: {new_member_name}")
                            st.rerun()
                        else:
                            st.error("Please enter name and role")

                # Edit/Remove board members
                with st.expander(f"📝 Edit Board Members ({len(board_members)} total)", expanded=True):
                    members_to_remove = []
                    for i, member in enumerate(board_members):
                        st.markdown(f"**👤 {member.get('name', f'Member {i+1}')}**")
                        col1, col2, col3 = st.columns([2, 2, 0.5])

                        with col1:
                            new_name = st.text_input(
                                "Name",
                                value=member.get('name', ''),
                                key=f"member_name_{i}"
                            )
                            if new_name != member.get('name'):
                                st.session_state.audit_data['company_data']['board_members'][i]['name'] = new_name
                                st.session_state.audit_modified = True

                        with col2:
                            new_role = st.text_input(
                                "Role",
                                value=member.get('role', ''),
                                key=f"member_role_{i}"
                            )
                            if new_role != member.get('role'):
                                st.session_state.audit_data['company_data']['board_members'][i]['role'] = new_role
                                st.session_state.audit_modified = True

                        with col3:
                            if st.button("🗑️", key=f"del_member_{i}", help="Remove this member"):
                                members_to_remove.append(i)

                        # Additional fields row
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            new_expertise = st.text_input(
                                "Expertise/Domain",
                                value=member.get('expertise', ''),
                                key=f"member_expertise_{i}",
                                placeholder="e.g., Finance, Technology"
                            )
                            if new_expertise != member.get('expertise', ''):
                                st.session_state.audit_data['company_data']['board_members'][i]['expertise'] = new_expertise
                                st.session_state.audit_modified = True

                        with col2:
                            new_tenure = st.number_input(
                                "Tenure (years)",
                                value=int(member.get('tenure_years', 0)),
                                min_value=0,
                                max_value=50,
                                key=f"member_tenure_{i}"
                            )
                            if new_tenure != member.get('tenure_years', 0):
                                st.session_state.audit_data['company_data']['board_members'][i]['tenure_years'] = new_tenure
                                st.session_state.audit_modified = True

                        with col3:
                            current_type = member.get('director_type', 'Independent Director')
                            type_index = director_types.index(current_type) if current_type in director_types else 0
                            new_type = st.selectbox(
                                "Director Type",
                                options=director_types,
                                index=type_index,
                                key=f"member_type_{i}"
                            )
                            if new_type != current_type:
                                st.session_state.audit_data['company_data']['board_members'][i]['director_type'] = new_type
                                st.session_state.audit_modified = True

                        new_personality = st.text_area(
                            "Personality",
                            value=member.get('personality', ''),
                            key=f"member_personality_{i}",
                            height=60
                        )
                        if new_personality != member.get('personality'):
                            st.session_state.audit_data['company_data']['board_members'][i]['personality'] = new_personality
                            st.session_state.audit_modified = True

                        st.markdown("---")

                    # Process removals (in reverse to maintain indices)
                    for idx in sorted(members_to_remove, reverse=True):
                        del st.session_state.audit_data['company_data']['board_members'][idx]
                        st.session_state.audit_modified = True
                    if members_to_remove:
                        st.rerun()

                st.divider()

                # ============ COMMITTEES SECTION ============
                st.subheader("🏛️ Board Committees")

                # Initialize committees if not exists
                if 'committees' not in st.session_state.audit_data['company_data']:
                    st.session_state.audit_data['company_data']['committees'] = []

                committees = st.session_state.audit_data['company_data'].get('committees', [])

                # Get list of board member names for selection
                member_names = [m.get('name', f"Member {i+1}") for i, m in enumerate(board_members)]

                # Predefined committee types
                committee_types = [
                    "Audit Committee",
                    "Risk Management Committee",
                    "Nomination & Remuneration Committee",
                    "Corporate Social Responsibility Committee",
                    "Stakeholders Relationship Committee",
                    "Strategy Committee",
                    "Finance Committee",
                    "Technology Committee",
                    "Compliance Committee",
                    "Executive Committee",
                    "Governance Committee",
                    "Custom"
                ]

                # Add new committee
                with st.expander("➕ Create New Committee", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        committee_type = st.selectbox(
                            "Committee Type",
                            options=committee_types,
                            key="new_committee_type"
                        )

                    with col2:
                        if committee_type == "Custom":
                            new_committee_name = st.text_input(
                                "Custom Committee Name",
                                key="new_custom_committee_name",
                                placeholder="Enter committee name"
                            )
                        else:
                            new_committee_name = committee_type
                            st.text_input(
                                "Committee Name",
                                value=committee_type,
                                key="new_committee_name_display",
                                disabled=True
                            )

                    new_committee_purpose = st.text_area(
                        "Committee Purpose/Mandate",
                        key="new_committee_purpose",
                        height=60,
                        placeholder="Describe the committee's purpose and responsibilities..."
                    )

                    st.markdown("**Select Committee Members:**")
                    if member_names:
                        selected_members = st.multiselect(
                            "Choose board members for this committee",
                            options=member_names,
                            key="new_committee_members",
                            help="Select one or more board members"
                        )

                        # Select chairperson from selected members
                        if selected_members:
                            committee_chair = st.selectbox(
                                "Committee Chairperson",
                                options=["None"] + selected_members,
                                key="new_committee_chair"
                            )
                        else:
                            committee_chair = "None"
                    else:
                        st.warning("No board members available. Please add board members first.")
                        selected_members = []
                        committee_chair = "None"

                    if st.button("Create Committee", key="create_committee_btn"):
                        final_name = new_committee_name if committee_type != "Custom" else new_committee_name
                        if final_name and selected_members:
                            # Check for duplicate committee names
                            existing_names = [c.get('name', '').lower() for c in committees]
                            if final_name.lower() in existing_names:
                                st.error(f"A committee named '{final_name}' already exists!")
                            else:
                                new_committee = {
                                    "name": final_name,
                                    "type": committee_type if committee_type != "Custom" else "Custom",
                                    "purpose": new_committee_purpose,
                                    "members": selected_members,
                                    "chairperson": committee_chair if committee_chair != "None" else None,
                                    "created_at": datetime.now().isoformat()
                                }
                                st.session_state.audit_data['company_data']['committees'].append(new_committee)
                                st.session_state.audit_modified = True
                                st.success(f"Created committee: {final_name} with {len(selected_members)} members")
                                st.rerun()
                        elif not final_name:
                            st.error("Please enter a committee name")
                        else:
                            st.error("Please select at least one member for the committee")

                # Display and Edit existing committees
                if committees:
                    with st.expander(f"📝 Manage Committees ({len(committees)} total)", expanded=True):
                        committees_to_remove = []

                        for i, committee in enumerate(committees):
                            st.markdown(f"### 🏛️ {committee.get('name', 'Unnamed Committee')}")

                            col1, col2, col3 = st.columns([3, 3, 1])

                            with col1:
                                edited_name = st.text_input(
                                    "Committee Name",
                                    value=committee.get('name', ''),
                                    key=f"committee_name_{i}"
                                )
                                if edited_name != committee.get('name'):
                                    st.session_state.audit_data['company_data']['committees'][i]['name'] = edited_name
                                    st.session_state.audit_modified = True

                            with col2:
                                edited_type = st.selectbox(
                                    "Type",
                                    options=committee_types,
                                    index=committee_types.index(committee.get('type', 'Custom')) if committee.get('type', 'Custom') in committee_types else committee_types.index('Custom'),
                                    key=f"committee_type_{i}"
                                )
                                if edited_type != committee.get('type'):
                                    st.session_state.audit_data['company_data']['committees'][i]['type'] = edited_type
                                    st.session_state.audit_modified = True

                            with col3:
                                st.markdown("<br>", unsafe_allow_html=True)
                                if st.button("🗑️ Delete", key=f"del_committee_{i}", help="Delete this committee"):
                                    committees_to_remove.append(i)

                            edited_purpose = st.text_area(
                                "Purpose/Mandate",
                                value=committee.get('purpose', ''),
                                key=f"committee_purpose_{i}",
                                height=60
                            )
                            if edited_purpose != committee.get('purpose'):
                                st.session_state.audit_data['company_data']['committees'][i]['purpose'] = edited_purpose
                                st.session_state.audit_modified = True

                            # Edit committee members
                            current_members = committee.get('members', [])
                            # Filter to only show members that still exist
                            valid_current_members = [m for m in current_members if m in member_names]

                            edited_members = st.multiselect(
                                "Committee Members",
                                options=member_names,
                                default=valid_current_members,
                                key=f"committee_members_{i}"
                            )
                            if set(edited_members) != set(current_members):
                                st.session_state.audit_data['company_data']['committees'][i]['members'] = edited_members
                                st.session_state.audit_modified = True

                            # Edit chairperson
                            current_chair = committee.get('chairperson')
                            chair_options = ["None"] + edited_members
                            current_chair_index = 0
                            if current_chair and current_chair in edited_members:
                                current_chair_index = chair_options.index(current_chair)

                            edited_chair = st.selectbox(
                                "Chairperson",
                                options=chair_options,
                                index=current_chair_index,
                                key=f"committee_chair_{i}"
                            )
                            new_chair_value = edited_chair if edited_chair != "None" else None
                            if new_chair_value != current_chair:
                                st.session_state.audit_data['company_data']['committees'][i]['chairperson'] = new_chair_value
                                st.session_state.audit_modified = True

                            # Display member count and chair
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"👥 {len(edited_members)} member(s)")
                            with col2:
                                if new_chair_value:
                                    st.info(f"👤 Chair: {new_chair_value}")

                            st.markdown("---")

                        # Process committee removals
                        for idx in sorted(committees_to_remove, reverse=True):
                            del st.session_state.audit_data['company_data']['committees'][idx]
                            st.session_state.audit_modified = True
                        if committees_to_remove:
                            st.rerun()
                else:
                    st.info("No committees created yet. Use the form above to create a committee.")

                # Committee Summary View
                if committees:
                    with st.expander("📊 Committee Summary", expanded=False):
                        st.markdown("### Committee Overview")

                        for committee in committees:
                            st.markdown(f"**{committee.get('name', 'Unnamed')}**")
                            chair = committee.get('chairperson', 'Not assigned')
                            members = committee.get('members', [])
                            st.markdown(f"- **Chairperson:** {chair if chair else 'Not assigned'}")
                            st.markdown(f"- **Members ({len(members)}):** {', '.join(members) if members else 'None'}")
                            if committee.get('purpose'):
                                st.markdown(f"- **Purpose:** {committee.get('purpose')[:100]}{'...' if len(committee.get('purpose', '')) > 100 else ''}")
                            st.markdown("")

                        # Member participation matrix
                        st.markdown("### Member Participation")
                        member_committees = {}
                        for member in member_names:
                            member_committees[member] = []
                            for committee in committees:
                                if member in committee.get('members', []):
                                    role = "Chair" if committee.get('chairperson') == member else "Member"
                                    member_committees[member].append(f"{committee.get('name')} ({role})")

                        for member, comms in member_committees.items():
                            if comms:
                                st.markdown(f"**{member}:** {', '.join(comms)}")
                            else:
                                st.markdown(f"**{member}:** _No committee assignments_")

                st.divider()

                # Current Problems Section
                st.subheader("⚠️ Current Problems")
                problems = company_data.get('current_problems', [])

                # Add new problem
                with st.expander("➕ Add New Problem", expanded=False):
                    new_problem = st.text_area("Problem Description", key="new_problem_text", height=60)

                    if st.button("Add Problem", key="add_problem_btn"):
                        if new_problem:
                            st.session_state.audit_data['company_data']['current_problems'].append(new_problem)
                            st.session_state.audit_modified = True
                            st.success("Added new problem!")
                            st.rerun()
                        else:
                            st.error("Please enter a problem description")

                # Edit/Remove problems
                with st.expander(f"📝 Edit Problems ({len(problems)} total)", expanded=True):
                    problems_to_remove = []
                    for i, problem in enumerate(problems):
                        col1, col2 = st.columns([10, 1])

                        with col1:
                            new_problem_text = st.text_area(
                                f"Problem {i+1}",
                                value=problem,
                                key=f"problem_{i}",
                                height=60,
                                label_visibility="collapsed"
                            )
                            if new_problem_text != problem:
                                st.session_state.audit_data['company_data']['current_problems'][i] = new_problem_text
                                st.session_state.audit_modified = True

                        with col2:
                            if st.button("🗑️", key=f"del_problem_{i}", help="Remove this problem"):
                                problems_to_remove.append(i)

                        st.markdown("---")

                    for idx in sorted(problems_to_remove, reverse=True):
                        del st.session_state.audit_data['company_data']['current_problems'][idx]
                        st.session_state.audit_modified = True
                    if problems_to_remove:
                        st.rerun()

            # ============ MODULE DATA AUDIT ============
            with audit_tab2:
                module_data = st.session_state.audit_data.get('module_data', {})

                # Basic Info Section
                st.subheader("📋 Basic Information")
                col1, col2 = st.columns(2)

                with col1:
                    new_module_name = st.text_input(
                        "Module Name",
                        value=module_data.get('module_name', ''),
                        key="audit_module_name"
                    )
                    if new_module_name != module_data.get('module_name', ''):
                        st.session_state.audit_data['module_data']['module_name'] = new_module_name
                        st.session_state.audit_modified = True

                with col2:
                    new_subject = st.text_input(
                        "Subject Area",
                        value=module_data.get('subject_area', ''),
                        key="audit_subject_area"
                    )
                    if new_subject != module_data.get('subject_area', ''):
                        st.session_state.audit_data['module_data']['subject_area'] = new_subject
                        st.session_state.audit_modified = True

                new_module_overview = st.text_area(
                    "Overview",
                    value=module_data.get('overview', ''),
                    height=80,
                    key="audit_module_overview"
                )
                if new_module_overview != module_data.get('overview', ''):
                    st.session_state.audit_data['module_data']['overview'] = new_module_overview
                    st.session_state.audit_modified = True

                st.divider()

                # Learning Objectives Section
                st.subheader("🎯 Learning Objectives")
                objectives = module_data.get('learning_objectives', [])

                with st.expander("➕ Add New Objective", expanded=False):
                    new_objective = st.text_input("Objective", key="new_objective_text")
                    if st.button("Add Objective", key="add_objective_btn"):
                        if new_objective:
                            if 'learning_objectives' not in st.session_state.audit_data['module_data']:
                                st.session_state.audit_data['module_data']['learning_objectives'] = []
                            st.session_state.audit_data['module_data']['learning_objectives'].append(new_objective)
                            st.session_state.audit_modified = True
                            st.success("Added objective!")
                            st.rerun()

                with st.expander(f"📝 Edit Objectives ({len(objectives)} total)", expanded=True):
                    obj_to_remove = []
                    for i, obj in enumerate(objectives):
                        col1, col2 = st.columns([10, 1])
                        with col1:
                            new_obj = st.text_input(f"Objective {i+1}", value=obj, key=f"obj_{i}", label_visibility="collapsed")
                            if new_obj != obj:
                                st.session_state.audit_data['module_data']['learning_objectives'][i] = new_obj
                                st.session_state.audit_modified = True
                        with col2:
                            if st.button("🗑️", key=f"del_obj_{i}"):
                                obj_to_remove.append(i)

                    for idx in sorted(obj_to_remove, reverse=True):
                        del st.session_state.audit_data['module_data']['learning_objectives'][idx]
                        st.session_state.audit_modified = True
                    if obj_to_remove:
                        st.rerun()

                st.divider()

                # Topics Section
                st.subheader("📚 Topics")
                topics = module_data.get('topics', [])

                with st.expander("➕ Add New Topic", expanded=False):
                    new_topic_name = st.text_input("Topic Name", key="new_topic_name")
                    new_topic_desc = st.text_area("Description", key="new_topic_desc", height=60)
                    new_topic_principles = st.text_input("Key Principles (comma-separated)", key="new_topic_principles")
                    new_topic_application = st.text_input("Application", key="new_topic_application")

                    if st.button("Add Topic", key="add_topic_btn"):
                        if new_topic_name:
                            if 'topics' not in st.session_state.audit_data['module_data']:
                                st.session_state.audit_data['module_data']['topics'] = []
                            st.session_state.audit_data['module_data']['topics'].append({
                                "name": new_topic_name,
                                "description": new_topic_desc,
                                "key_principles": [p.strip() for p in new_topic_principles.split(',') if p.strip()],
                                "formulas": [],
                                "application": new_topic_application,
                                "examples": []
                            })
                            st.session_state.audit_modified = True
                            st.success(f"Added topic: {new_topic_name}")
                            st.rerun()

                with st.expander(f"📝 Edit Topics ({len(topics)} total)", expanded=True):
                    topics_to_remove = []
                    for i, topic in enumerate(topics):
                        st.markdown(f"**Topic {i+1}: {topic.get('name', 'Unnamed')}**")
                        col1, col2 = st.columns([10, 1])

                        with col1:
                            new_name = st.text_input("Name", value=topic.get('name', ''), key=f"topic_name_{i}")
                            if new_name != topic.get('name'):
                                st.session_state.audit_data['module_data']['topics'][i]['name'] = new_name
                                st.session_state.audit_modified = True

                            new_desc = st.text_area("Description", value=topic.get('description', ''), key=f"topic_desc_{i}", height=60)
                            if new_desc != topic.get('description'):
                                st.session_state.audit_data['module_data']['topics'][i]['description'] = new_desc
                                st.session_state.audit_modified = True

                            new_app = st.text_input("Application", value=topic.get('application', ''), key=f"topic_app_{i}")
                            if new_app != topic.get('application'):
                                st.session_state.audit_data['module_data']['topics'][i]['application'] = new_app
                                st.session_state.audit_modified = True

                        with col2:
                            if st.button("🗑️", key=f"del_topic_{i}"):
                                topics_to_remove.append(i)

                        st.markdown("---")

                    for idx in sorted(topics_to_remove, reverse=True):
                        del st.session_state.audit_data['module_data']['topics'][idx]
                        st.session_state.audit_modified = True
                    if topics_to_remove:
                        st.rerun()

                st.divider()

                # Key Terms Section
                st.subheader("📖 Key Terms")
                terms = module_data.get('key_terms', {})

                with st.expander("➕ Add New Term", expanded=False):
                    new_term = st.text_input("Term", key="new_term_name")
                    new_definition = st.text_area("Definition", key="new_term_def", height=60)

                    if st.button("Add Term", key="add_term_btn"):
                        if new_term and new_term not in terms:
                            if 'key_terms' not in st.session_state.audit_data['module_data']:
                                st.session_state.audit_data['module_data']['key_terms'] = {}
                            st.session_state.audit_data['module_data']['key_terms'][new_term] = new_definition
                            st.session_state.audit_modified = True
                            st.success(f"Added term: {new_term}")
                            st.rerun()
                        elif new_term in terms:
                            st.error("Term already exists!")

                with st.expander(f"📝 Edit Terms ({len(terms)} total)", expanded=True):
                    terms_to_remove = []
                    for term, definition in terms.items():
                        col1, col2 = st.columns([10, 1])

                        with col1:
                            st.markdown(f"**{term}**")
                            new_def = st.text_area(
                                "Definition",
                                value=definition,
                                key=f"term_def_{term}",
                                height=60,
                                label_visibility="collapsed"
                            )
                            if new_def != definition:
                                st.session_state.audit_data['module_data']['key_terms'][term] = new_def
                                st.session_state.audit_modified = True

                        with col2:
                            if st.button("🗑️", key=f"del_term_{term}"):
                                terms_to_remove.append(term)

                        st.markdown("---")

                    for term_key in terms_to_remove:
                        del st.session_state.audit_data['module_data']['key_terms'][term_key]
                        st.session_state.audit_modified = True
                    if terms_to_remove:
                        st.rerun()

                st.divider()

                # Frameworks Section
                st.subheader("🔧 Frameworks")
                frameworks = module_data.get('frameworks', [])

                with st.expander("➕ Add New Framework", expanded=False):
                    new_fw_name = st.text_input("Framework Name", key="new_fw_name")
                    new_fw_desc = st.text_area("Description", key="new_fw_desc", height=60)
                    new_fw_components = st.text_input("Components (comma-separated)", key="new_fw_components")
                    new_fw_scenario = st.text_input("Application Scenario", key="new_fw_scenario")

                    if st.button("Add Framework", key="add_fw_btn"):
                        if new_fw_name:
                            if 'frameworks' not in st.session_state.audit_data['module_data']:
                                st.session_state.audit_data['module_data']['frameworks'] = []
                            st.session_state.audit_data['module_data']['frameworks'].append({
                                "name": new_fw_name,
                                "description": new_fw_desc,
                                "components": [c.strip() for c in new_fw_components.split(',') if c.strip()],
                                "application_scenario": new_fw_scenario
                            })
                            st.session_state.audit_modified = True
                            st.success(f"Added framework: {new_fw_name}")
                            st.rerun()

                with st.expander(f"📝 Edit Frameworks ({len(frameworks)} total)", expanded=True):
                    fw_to_remove = []
                    for i, fw in enumerate(frameworks):
                        st.markdown(f"**Framework {i+1}: {fw.get('name', 'Unnamed')}**")
                        col1, col2 = st.columns([10, 1])

                        with col1:
                            new_name = st.text_input("Name", value=fw.get('name', ''), key=f"fw_name_{i}")
                            if new_name != fw.get('name'):
                                st.session_state.audit_data['module_data']['frameworks'][i]['name'] = new_name
                                st.session_state.audit_modified = True

                            new_desc = st.text_area("Description", value=fw.get('description', ''), key=f"fw_desc_{i}", height=60)
                            if new_desc != fw.get('description'):
                                st.session_state.audit_data['module_data']['frameworks'][i]['description'] = new_desc
                                st.session_state.audit_modified = True

                            new_scenario = st.text_input("Application Scenario", value=fw.get('application_scenario', ''), key=f"fw_scenario_{i}")
                            if new_scenario != fw.get('application_scenario'):
                                st.session_state.audit_data['module_data']['frameworks'][i]['application_scenario'] = new_scenario
                                st.session_state.audit_modified = True

                        with col2:
                            if st.button("🗑️", key=f"del_fw_{i}"):
                                fw_to_remove.append(i)

                        st.markdown("---")

                    for idx in sorted(fw_to_remove, reverse=True):
                        del st.session_state.audit_data['module_data']['frameworks'][idx]
                        st.session_state.audit_modified = True
                    if fw_to_remove:
                        st.rerun()

                st.divider()

                # Assessment Criteria Section
                st.subheader("✅ Assessment Criteria")
                criteria = module_data.get('assessment_criteria', [])

                with st.expander("➕ Add New Criterion", expanded=False):
                    new_criterion = st.text_input("Criterion", key="new_criterion_text")
                    if st.button("Add Criterion", key="add_criterion_btn"):
                        if new_criterion:
                            if 'assessment_criteria' not in st.session_state.audit_data['module_data']:
                                st.session_state.audit_data['module_data']['assessment_criteria'] = []
                            st.session_state.audit_data['module_data']['assessment_criteria'].append(new_criterion)
                            st.session_state.audit_modified = True
                            st.success("Added criterion!")
                            st.rerun()

                with st.expander(f"📝 Edit Criteria ({len(criteria)} total)", expanded=True):
                    criteria_to_remove = []
                    for i, criterion in enumerate(criteria):
                        col1, col2 = st.columns([10, 1])
                        with col1:
                            new_crit = st.text_input(f"Criterion {i+1}", value=criterion, key=f"crit_{i}", label_visibility="collapsed")
                            if new_crit != criterion:
                                st.session_state.audit_data['module_data']['assessment_criteria'][i] = new_crit
                                st.session_state.audit_modified = True
                        with col2:
                            if st.button("🗑️", key=f"del_crit_{i}"):
                                criteria_to_remove.append(i)

                    for idx in sorted(criteria_to_remove, reverse=True):
                        del st.session_state.audit_data['module_data']['assessment_criteria'][idx]
                        st.session_state.audit_modified = True
                    if criteria_to_remove:
                        st.rerun()

            # Save Changes Section
            st.divider()
            st.subheader("💾 Save Changes")

            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                if st.button("💾 Save to Current File", type="primary", disabled=not st.session_state.audit_modified):
                    if st.session_state.audit_loaded_file:
                        # Update modified timestamp
                        st.session_state.audit_data['modified_at'] = datetime.now().isoformat()

                        with open(st.session_state.audit_loaded_file, 'w', encoding='utf-8') as f:
                            json.dump(st.session_state.audit_data, f, indent=2, ensure_ascii=False)

                        st.session_state.audit_modified = False
                        st.success("Changes saved successfully!")
                        st.rerun()

            with col2:
                new_session_name = st.text_input("New Session Name", key="audit_new_session_name", placeholder="Enter name to save as new")

            with col3:
                if st.button("💾 Save as New Session"):
                    if new_session_name:
                        st.session_state.audit_data['session_name'] = new_session_name
                        st.session_state.audit_data['created_at'] = datetime.now().isoformat()

                        filepath = save_extracted_data(
                            st.session_state.audit_data.get('company_data', {}),
                            st.session_state.audit_data.get('module_data', {}),
                            new_session_name
                        )
                        st.session_state.audit_modified = False
                        st.success(f"Saved as new session: {filepath}")
                    else:
                        st.error("Please enter a session name")

            # View and Export JSON
            st.divider()
            st.subheader("📄 View & Download JSON")

            # View JSON options
            view_tab1, view_tab2, view_tab3 = st.tabs(["📋 Full Data", "🏢 Company Only", "📚 Module Only"])

            with view_tab1:
                st.markdown("**Complete Session Data**")
                full_json = json.dumps(st.session_state.audit_data, indent=2, ensure_ascii=False)

                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Size", f"{len(full_json):,} chars")
                with col2:
                    metrics_count = len(st.session_state.audit_data.get('company_data', {}).get('metrics', {}))
                    st.metric("Metrics", metrics_count)
                with col3:
                    members_count = len(st.session_state.audit_data.get('company_data', {}).get('board_members', []))
                    st.metric("Board Members", members_count)

                # View JSON
                with st.expander("👁️ View Full JSON", expanded=False):
                    st.code(full_json, language="json")

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ Download Full JSON",
                        data=full_json,
                        file_name=f"full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_full_json"
                    )
                with col2:
                    # Minified version
                    minified_json = json.dumps(st.session_state.audit_data, ensure_ascii=False)
                    st.download_button(
                        label="⬇️ Download Minified JSON",
                        data=minified_json,
                        file_name=f"full_export_minified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_full_minified"
                    )

            with view_tab2:
                st.markdown("**Company Data Only**")
                company_json = json.dumps(st.session_state.audit_data.get('company_data', {}), indent=2, ensure_ascii=False)

                # Statistics
                company_data_view = st.session_state.audit_data.get('company_data', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Size", f"{len(company_json):,} chars")
                with col2:
                    st.metric("Metrics", len(company_data_view.get('metrics', {})))
                with col3:
                    st.metric("Board Members", len(company_data_view.get('board_members', [])))
                with col4:
                    st.metric("Committees", len(company_data_view.get('committees', [])))

                # View JSON
                with st.expander("👁️ View Company JSON", expanded=False):
                    st.code(company_json, language="json")

                # Download
                st.download_button(
                    label="⬇️ Download Company JSON",
                    data=company_json,
                    file_name=f"company_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_company_json"
                )

            with view_tab3:
                st.markdown("**Module Data Only**")
                module_json = json.dumps(st.session_state.audit_data.get('module_data', {}), indent=2, ensure_ascii=False)

                # Statistics
                module_data_view = st.session_state.audit_data.get('module_data', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Size", f"{len(module_json):,} chars")
                with col2:
                    st.metric("Topics", len(module_data_view.get('topics', [])))
                with col3:
                    st.metric("Key Terms", len(module_data_view.get('key_terms', {})))
                with col4:
                    st.metric("Frameworks", len(module_data_view.get('frameworks', [])))

                # View JSON
                with st.expander("👁️ View Module JSON", expanded=False):
                    st.code(module_json, language="json")

                # Download
                st.download_button(
                    label="⬇️ Download Module JSON",
                    data=module_json,
                    file_name=f"module_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_module_json"
                )

            # Copy to clipboard helper
            st.divider()
            with st.expander("📋 Copy JSON to Clipboard"):
                st.markdown("Select which data to copy:")
                copy_option = st.radio(
                    "Select data",
                    options=["Full Data", "Company Only", "Module Only"],
                    horizontal=True,
                    key="copy_option",
                    label_visibility="collapsed"
                )

                if copy_option == "Full Data":
                    copy_data = json.dumps(st.session_state.audit_data, indent=2, ensure_ascii=False)
                elif copy_option == "Company Only":
                    copy_data = json.dumps(st.session_state.audit_data.get('company_data', {}), indent=2, ensure_ascii=False)
                else:
                    copy_data = json.dumps(st.session_state.audit_data.get('module_data', {}), indent=2, ensure_ascii=False)

                st.text_area(
                    "JSON Data (select all and copy)",
                    value=copy_data,
                    height=200,
                    key="copy_json_area"
                )

    with tab4:
        st.header("ℹ️ How to Use")

        st.markdown("""
        ### Overview
        This app helps you prepare data for the Board Meeting Simulation. You'll upload two PDF documents:

        1. **Company Document**: An annual report, case study, or company profile containing:
           - Company overview and background
           - Financial and operational metrics
           - Leadership team information
           - Current business challenges

        2. **Module Document**: A course or training material containing:
           - Learning objectives
           - Key topics and concepts
           - Frameworks and models
           - Assessment criteria

        ### Steps

        1. **Upload PDFs**: Upload both company and module PDF files
        2. **Extract Data**: Click the extract buttons to process each PDF with AI
        3. **Review**: Check the extracted data preview to ensure accuracy
        4. **Audit**: Use the Audit Data tab to check, edit, add, or remove information
        5. **Save**: Give your session a name and save for later simulation

        ### Audit Features

        The **Audit Data** tab allows you to:
        - **Check**: Review all extracted data in detail
        - **Edit**: Modify existing values (metrics, board members, topics, etc.)
        - **Add**: Add new items (metrics, board members, problems, topics, terms, etc.)
        - **Remove**: Delete items that are incorrect or unnecessary
        - **Committees**: Create and manage board committees with selected members
        - **Save**: Save changes to the current file or as a new session
        - **Export**: Download the data as a JSON file

        ### Board Committees

        You can create multiple committees from your board members:
        - **Predefined Types**: Audit, Risk Management, Nomination & Remuneration, CSR, Strategy, Finance, Technology, Compliance, Executive, Governance
        - **Custom Committees**: Create any custom committee type
        - **Member Assignment**: Select multiple board members for each committee
        - **Chairperson**: Designate a chairperson from committee members
        - **Purpose/Mandate**: Define each committee's responsibilities
        - **Summary View**: See an overview of all committees and member participation

        ### Tips

        - Use clear, text-based PDFs for best results
        - Larger documents may take longer to process
        - You can re-extract if the initial results aren't satisfactory
        - Saved sessions can be loaded later without re-uploading
        - Always audit extracted data before running simulations

        ### Technical Notes

        - Data is saved as JSON files in the `extracted_data/` folder
        - The simulation app can load these files directly
        - API key for Gemini is required in `.streamlit/secrets.toml`
        """)

if __name__ == "__main__":
    main()