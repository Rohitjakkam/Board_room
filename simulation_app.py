import streamlit as st
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import PyPDF2
import io
import base64
from PIL import Image
import pdf2image

# Configure Gemini
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))

# Constants
MAX_ROUNDS = 8

@dataclass
class BoardMember:
    name: str
    role: str
    personality: str

class BoardMeetingSimulation:
    def __init__(self):
        self.board_members = []
        self.company_context = ""
        self.metrics = {}  # Dynamic metrics
        self.metric_definitions = {}  # Store what each metric means
        
    def extract_pdf_with_gemini(self, pdf_file) -> str:
        """Use Gemini's native PDF processing"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Read PDF as bytes
            pdf_bytes = pdf_file.read()
            
            # Encode to base64
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Use Gemini with document support
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            # Upload file to Gemini
            pdf_file.seek(0)
            uploaded_file = genai.upload_file(pdf_file, mime_type='application/pdf')
            
            # Extract all text content
            prompt = """Please extract ALL text content from this PDF document. 
            Include everything: headers, body text, tables, numbers, charts data, footnotes, etc.
            Maintain the structure and organization of the content.
            Be extremely thorough - extract every piece of text you can find."""
            
            response = model.generate_content([uploaded_file, prompt])
            
            return response.text
            
        except Exception as e:
            st.error(f"Error with Gemini PDF extraction: {e}")
            # Fallback to PyPDF2
            return self.extract_pdf_with_pypdf2(pdf_file)
    
    def extract_pdf_with_pypdf2(self, pdf_file) -> str:
        """Fallback PDF extraction using PyPDF2"""
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            st.info(f"PDF has {len(pdf_reader.pages)} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n{'='*50}\nPAGE {i+1}\n{'='*50}\n{page_text}\n"
            
            if not text.strip():
                st.error("⚠️ PyPDF2 couldn't extract text. The PDF might be image-based or protected.")
                return ""
                
            return text
        except Exception as e:
            st.error(f"Error extracting PDF with PyPDF2: {e}")
            return ""
    
    def parse_company_data(self, pdf_text: str) -> Dict:
        """Use Gemini to parse company data from PDF with enhanced extraction"""
        if not pdf_text or len(pdf_text.strip()) < 100:
            st.error("❌ Insufficient text extracted from PDF. Please ensure the PDF contains readable text.")
            raise ValueError("PDF text is empty or too short")
        
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are an expert business analyst. Analyze this company document VERY CAREFULLY and extract ALL information.

DOCUMENT TEXT (Total length: {len(pdf_text)} characters):
{pdf_text}

YOUR TASK - EXTRACT EVERYTHING:

1. COMPANY BASICS:
   - Find the company name (look carefully - it's usually at the top or in headers)
   - Write a detailed 4-5 sentence overview covering: industry, business model, size, operations, current state

2. METRICS (MOST IMPORTANT - BE THOROUGH):
   Extract EVERY SINGLE number, percentage, or measurement you find:
   
   FINANCIAL METRICS:
   - Revenue (total, growth rate, by segment)
   - Profit (gross, net, operating, margins)
   - EBITDA, EBIT
   - Cash flow, cash reserves
   - Debt, equity, ratios
   - ROI, ROE, ROA
   - Expenses (operating, capital, R&D)
   
   OPERATIONAL METRICS:
   - Production capacity, utilization
   - Efficiency scores
   - Quality metrics, defect rates
   - Inventory levels, turnover
   - Supply chain metrics
   
   EMPLOYEE METRICS:
   - Total headcount
   - Satisfaction scores, engagement
   - Turnover rate, retention rate
   - Training hours, costs
   - Productivity metrics
   - Diversity percentages
   
   MARKET METRICS:
   - Market share (total and by segment)
   - Growth rates
   - Customer acquisition cost (CAC)
   - Customer lifetime value (CLV)
   - Retention rate, churn rate
   - Net Promoter Score (NPS)
   - Brand value, recognition
   
   PRODUCT/INNOVATION:
   - R&D spending
   - New product launches
   - Patent counts
   - Innovation index
   - Time to market
   
   RISK & COMPLIANCE:
   - Risk scores
   - Compliance rates
   - Incident counts
   - Audit findings
   
   Look in: tables, charts, financial statements, dashboards, KPI sections, executive summaries

3. LEADERSHIP TEAM (FIND EVERYONE):
   Look for ALL people mentioned with titles:
   - C-Suite: CEO, CFO, COO, CTO, CMO, etc.
   - VPs and SVPs
   - Directors
   - Board members
   - Department heads
   
   For each person extract:
   - Full name (exactly as written)
   - Complete title/role
   - Personality traits: Look for their quoted statements, concerns they raised, their focus areas, leadership style mentioned, or infer from their role

4. PROBLEMS & CHALLENGES:
   Find ALL issues mentioned:
   - Strategic challenges
   - Market threats
   - Operational problems
   - Financial concerns
   - Competitive pressures
   - Regulatory issues
   - Technology gaps
   - HR challenges

5. CURRENT SITUATION:
   Synthesize the overall business situation and key decisions needed

RETURN THIS JSON (BE EXTREMELY THOROUGH):

{{
    "company_name": "EXACT company name from document",
    "company_overview": "Detailed 4-5 sentences about the company",
    "metrics": {{
        "revenue_total": {{"value": 0, "unit": "$M", "description": "Total annual revenue"}},
        "revenue_growth": {{"value": 0, "unit": "%", "description": "Year over year revenue growth"}},
        "net_profit_margin": {{"value": 0, "unit": "%", "description": "Net profit margin"}},
        "operating_margin": {{"value": 0, "unit": "%", "description": "Operating profit margin"}},
        "ebitda": {{"value": 0, "unit": "$M", "description": "EBITDA"}},
        "cash_flow": {{"value": 0, "unit": "$M", "description": "Operating cash flow"}},
        "debt_to_equity": {{"value": 0, "unit": "ratio", "description": "Debt to equity ratio"}},
        "employee_count": {{"value": 0, "unit": "employees", "description": "Total workforce"}},
        "employee_satisfaction": {{"value": 0, "unit": "%", "description": "Employee satisfaction score"}},
        "employee_turnover": {{"value": 0, "unit": "%", "description": "Annual turnover rate"}},
        "market_share": {{"value": 0, "unit": "%", "description": "Market share percentage"}},
        "customer_satisfaction": {{"value": 0, "unit": "score", "description": "Customer satisfaction score"}},
        "nps": {{"value": 0, "unit": "score", "description": "Net Promoter Score"}},
        
        // ADD 20-40 MORE METRICS BASED ON WHAT YOU FIND IN THE DOCUMENT
        // Use format: "metric_name": {{"value": number, "unit": "unit", "description": "description"}}
    }},
    "board_members": [
        {{"name": "Full Name", "role": "Complete Title", "personality": "Detailed personality/style (3-4 sentences)"}},
        {{"name": "Full Name", "role": "Complete Title", "personality": "Detailed personality/style"}},
        // ADD ALL LEADERS/EXECUTIVES FOUND (aim for 5-15 people)
    ],
    "current_problems": [
        "Specific detailed problem 1",
        "Specific detailed problem 2",
        // ADD ALL PROBLEMS FOUND (aim for 5-10 problems)
    ],
    "initial_scenario": "Comprehensive description of current situation (3-4 sentences)"
}}

CRITICAL RULES:
- Extract MINIMUM 20 metrics, IDEAL 30-50 metrics
- Extract MINIMUM 5 board members/executives
- Extract MINIMUM 5 current problems
- Use EXACT names, numbers, and terminology from the document
- If a specific number isn't in the doc, you can estimate ONLY if there's strong context
- Return ONLY valid JSON, nothing else"""

        try:
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up the response
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            # Try to find JSON if there's extra text
            if not result_text.startswith('{'):
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    result_text = result_text[json_start:json_end]
            
            parsed_data = json.loads(result_text)
            
            # Validate extraction quality
            metrics_count = len(parsed_data.get('metrics', {}))
            board_count = len(parsed_data.get('board_members', []))
            problems_count = len(parsed_data.get('current_problems', []))
            
            if metrics_count < 5:
                st.warning(f"⚠️ Only {metrics_count} metrics extracted. This seems low. The document might not contain much data.")
            if board_count < 2:
                st.warning(f"⚠️ Only {board_count} board members found. The document might not list leadership.")
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse AI response as JSON. Error: {e}")
            with st.expander("🔍 View AI Response"):
                st.text(result_text)
            raise
        except Exception as e:
            st.error(f"❌ Error parsing company data: {e}")
            raise
    
    def generate_scenario(self, context: str, metrics: Dict, is_first: bool = False, round_num: int = 1) -> Dict:
        """Generate next scenario based on previous decision"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        metrics_summary = self._format_metrics_for_prompt(metrics)
        
        is_final_round = (round_num == MAX_ROUNDS)
        
        if is_first:
            prompt = f"""You are simulating a board meeting scenario based on this company context.

Company Context: {self.company_context}

Current Company Metrics:
{metrics_summary}

Generate the FIRST critical decision point for the CEO based on the company's current situation and metrics.
This is round 1 of {MAX_ROUNDS} total rounds.

Generate a JSON response with:
1. "scenario": A detailed business scenario/problem (2-3 sentences) based on the company context and current metrics
2. "question": The decision question for the CEO
3. "options": Array of 3-4 decision options as STRINGS (each should be a clear action described in 1-2 sentences)

Example format:
{{
    "scenario": "The company faces...",
    "question": "What should we do?",
    "options": [
        "Option 1 description as a string",
        "Option 2 description as a string",
        "Option 3 description as a string"
    ]
}}

Make it realistic and urgent. Use the company context and specific metrics to make it relevant to this company.
IMPORTANT: Options must be simple strings, not objects or dictionaries.

Return ONLY valid JSON, no other text."""
        elif is_final_round:
            prompt = f"""You are simulating the FINAL board meeting scenario. This is round {MAX_ROUNDS} of {MAX_ROUNDS}.

Company Background: {self.company_context}
Previous Decision Context: {context}

Current Company Metrics:
{metrics_summary}

Generate the FINAL critical decision that will determine the company's long-term future. This should feel like a culminating moment that will shape the company's trajectory for years to come.

Generate a JSON response with:
1. "scenario": A detailed, high-stakes scenario (2-3 sentences) that represents a defining moment for the company
2. "question": The ultimate strategic question for the CEO
3. "options": Array of 3-4 decision options as STRINGS (each representing a bold, consequential path forward)

Example format:
{{
    "scenario": "After all the previous decisions, the company now faces its most critical moment...",
    "question": "What is your final strategic direction?",
    "options": [
        "Option 1 description as a string",
        "Option 2 description as a string",
        "Option 3 description as a string"
    ]
}}

Make it feel epic and consequential. This is the CEO's defining decision.
IMPORTANT: Options must be simple strings, not objects or dictionaries.

Return ONLY valid JSON, no other text."""
        else:
            prompt = f"""You are simulating a board meeting scenario. This is round {round_num} of {MAX_ROUNDS} total rounds.

Company Background: {self.company_context}
Previous Decision Context: {context}

Current Company Metrics:
{metrics_summary}

Generate the NEXT critical decision point that naturally follows from the context.

Generate a JSON response with:
1. "scenario": A detailed business scenario/problem (2-3 sentences) that naturally follows from the context
2. "question": The decision question for the CEO
3. "options": Array of 3-4 decision options as STRINGS (each should be a clear action described in 1-2 sentences)

Example format:
{{
    "scenario": "Following the previous decision...",
    "question": "How should we proceed?",
    "options": [
        "Option 1 description as a string",
        "Option 2 description as a string",
        "Option 3 description as a string"
    ]
}}

Make it realistic and consequential. The scenario should feel like a natural consequence or new challenge.
Reference specific metrics in the scenario when relevant.
IMPORTANT: Options must be simple strings, not objects or dictionaries.

Return ONLY valid JSON, no other text."""

        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            
            # Ensure options are strings
            if 'options' in result and result['options']:
                if isinstance(result['options'][0], dict):
                    result['options'] = [str(opt.get('text', opt.get('value', opt))) for opt in result['options']]
                else:
                    result['options'] = [str(opt) for opt in result['options']]
            
            return result
        except Exception as e:
            st.error(f"Error generating scenario: {e}")
            raise
    
    def generate_board_opinions(self, scenario: str, question: str, options: List, metrics: Dict, selected_members: List[str] = None) -> Dict[str, str]:
        """Generate each board member's opinion on the decision"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        opinions = {}
        
        metrics_summary = self._format_metrics_for_prompt(metrics)
        
        # Ensure options are strings - be defensive
        options_str = []
        for opt in options:
            if isinstance(opt, dict):
                options_str.append(str(opt.get('text', opt.get('value', str(opt)))))
            elif opt is None:
                options_str.append("None")
            else:
                options_str.append(str(opt))
        
        # Filter members if specific ones are selected
        members_to_ask = self.board_members
        if selected_members:
            members_to_ask = [m for m in self.board_members if m.name in selected_members]
        
        for member in members_to_ask:
            prompt = f"""You are {member.name}, {member.role} with personality: {member.personality}.

Company Context: {self.company_context}

Scenario: {scenario}
Question: {question}
Options: {', '.join(options_str)}

Current Metrics:
{metrics_summary}

Provide your opinion (2-3 sentences) on which option you prefer and why, staying true to your role and personality. 
Be specific and reference relevant metrics or company situation."""

            response = model.generate_content(prompt)
            opinions[member.name] = response.text.strip()
        
        return opinions
    
    def chat_with_board_member(self, member_name: str, user_question: str, scenario: str, question: str, options: List, metrics: Dict, conversation_history: List[Dict] = None) -> str:
        """Have a direct conversation with a specific board member"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Find the member
        member = next((m for m in self.board_members if m.name == member_name), None)
        if not member:
            return "Board member not found."
        
        metrics_summary = self._format_metrics_for_prompt(metrics)
        
        # Ensure options are strings
        options_str = []
        for opt in options:
            if isinstance(opt, dict):
                options_str.append(str(opt.get('text', opt.get('value', str(opt)))))
            elif opt is None:
                options_str.append("None")
            else:
                options_str.append(str(opt))
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            for msg in conversation_history:
                conversation_context += f"\nCEO: {msg['question']}\n{member_name}: {msg['answer']}\n"
        
        prompt = f"""You are {member.name}, {member.role} with personality: {member.personality}.

Company Context: {self.company_context}

Current Board Meeting Scenario:
{scenario}

Decision Question: {question}
Options: {', '.join(options_str)}

Current Metrics:
{metrics_summary}

Previous Conversation:
{conversation_context}

The CEO is now asking you: "{user_question}"

Respond naturally as {member.name}, staying true to your role and personality. Be helpful, specific, and reference relevant data or company context. Keep your response to 2-4 sentences unless the question requires more detail."""

        response = model.generate_content(prompt)
        return response.text.strip()
    
    def calculate_metric_impact(self, scenario: str, decision: str, metrics: Dict) -> Dict:
        """Calculate how the decision impacts company metrics"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        metrics_summary = self._format_metrics_for_prompt(metrics)
        metric_names = list(metrics.keys())
        
        prompt = f"""Analyze the business impact of this decision on company metrics.

Company Context: {self.company_context}
Scenario: {scenario}
Decision Made: {decision}

Current Metrics:
{metrics_summary}

Return a JSON object with changes to the metrics. Include ALL metrics that might be affected.
Use positive/negative numbers for change. Format:
{{
    "metric_name_1": {{
        "change": float (the change amount),
        "reason": "brief explanation of why this changed"
    }},
    "metric_name_2": {{
        "change": float,
        "reason": "explanation"
    }}
    // Include all metrics: {', '.join(metric_names)}
}}

Be realistic - most decisions have trade-offs and don't affect all metrics equally.
Changes should typically be between -15 to +15 depending on the metric scale.
Some metrics might not change at all (use 0).

Return ONLY valid JSON."""

        response = model.generate_content(prompt)
        changes = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
        
        # Apply changes to metrics - deep copy to avoid mutation
        new_metrics = {}
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                new_metrics[metric_name] = metric_data.copy()
            else:
                new_metrics[metric_name] = {'value': metric_data, 'unit': '', 'description': ''}
        
        for metric_name, change_data in changes.items():
            if metric_name in new_metrics:
                old_value = new_metrics[metric_name]['value']
                change = change_data.get('change', 0)
                new_value = old_value + change
                
                # Apply reasonable bounds (don't go below 0 for most metrics)
                if new_value < 0 and not any(x in metric_name.lower() for x in ['loss', 'debt', 'risk', 'turnover']):
                    new_value = 0
                
                new_metrics[metric_name]['value'] = new_value
        
        return new_metrics
    
    def generate_final_summary(self, history: List[Dict], final_metrics: Dict) -> str:
        """Generate a comprehensive summary of the CEO's performance"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Prepare history summary
        decisions_summary = "\n".join([
            f"Round {entry['round']}: {entry['decision']}" 
            for entry in history
        ])
        
        # Get initial and final metrics for comparison
        initial_metrics = history[0]['old_metrics'] if history else {}
        
        metrics_comparison = []
        for metric_name in final_metrics.keys():
            if metric_name in initial_metrics:
                initial = initial_metrics[metric_name].get('value', 0) if isinstance(initial_metrics[metric_name], dict) else initial_metrics[metric_name]
                final = final_metrics[metric_name].get('value', 0) if isinstance(final_metrics[metric_name], dict) else final_metrics[metric_name]
                change = final - initial
                unit = final_metrics[metric_name].get('unit', '') if isinstance(final_metrics[metric_name], dict) else ''
                metrics_comparison.append(f"{metric_name}: {initial}{unit} → {final}{unit} ({change:+.1f})")
        
        prompt = f"""You are a business performance analyst. The CEO has completed an 8-round board meeting simulation.

Company Context: {self.company_context}

Decisions Made:
{decisions_summary}

Metric Changes:
{chr(10).join(metrics_comparison[:15])}  

Provide a comprehensive performance evaluation (4-5 paragraphs) covering:
1. Overall leadership assessment
2. Key successes and wins
3. Areas of concern or missed opportunities
4. Strategic coherence across decisions
5. Final grade and outlook for the company

Be honest, insightful, and specific. Reference actual decisions and metrics."""

        response = model.generate_content(prompt)
        return response.text.strip()
    
    def _format_metrics_for_prompt(self, metrics: Dict) -> str:
        """Format metrics dictionary for AI prompts"""
        lines = []
        for name, data in metrics.items():
            # Handle both dict and non-dict values
            if isinstance(data, dict):
                value = data.get('value', 0)
                unit = data.get('unit', '')
                description = data.get('description', '')
            else:
                value = data
                unit = ''
                description = ''
            
            metric_line = f"- {name.replace('_', ' ').title()}: {value}{unit}"
            if description:
                metric_line += f" ({description})"
            lines.append(metric_line)
        return '\n'.join(lines)

# Streamlit UI
st.set_page_config(page_title="CEO Board Meeting Simulation", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.simulation = BoardMeetingSimulation()

# PDF Upload Section
if not st.session_state.initialized:
    st.title("🏢 CEO Board Meeting Simulation")
    st.markdown("### 📄 Upload Company Document")
    st.info(f"Upload a PDF containing company details, metrics, board members, and current challenges. You'll face {MAX_ROUNDS} rounds of critical decisions.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        try:
            with st.spinner("📖 Step 1/3: Extracting text from PDF..."):
                # Try Gemini's native PDF processing first
                pdf_text = st.session_state.simulation.extract_pdf_with_gemini(uploaded_file)
                
                # Show extraction results
                text_length = len(pdf_text.strip())
                st.success(f"✅ Extracted {text_length:,} characters from PDF")
                
                if text_length < 100:
                    st.error("❌ PDF appears to be empty or unreadable. Please check your file.")
                    st.stop()
            
            # Show extracted text preview
            with st.expander("🔍 Preview Extracted Text (first 2000 characters)"):
                st.text(pdf_text[:2000] + "..." if len(pdf_text) > 2000 else pdf_text)
            
            with st.spinner("🤖 Step 2/3: AI is analyzing document (30-60 seconds)..."):
                # Parse company data
                company_data = st.session_state.simulation.parse_company_data(pdf_text)
            
            with st.spinner("⚙️ Step 3/3: Initializing simulation..."):
                # Store parsed data
                st.session_state.company_name = company_data.get('company_name', 'Your Company')
                st.session_state.company_overview = company_data.get('company_overview', '')
                
                # Initialize dynamic metrics
                st.session_state.simulation.metrics = company_data.get('metrics', {})
                st.session_state.metrics_count = len(st.session_state.simulation.metrics)
                
                # Initialize board members
                board_data = company_data.get('board_members', [])
                st.session_state.simulation.board_members = [
                    BoardMember(
                        name=member.get('name', 'Unknown'),
                        role=member.get('role', 'Board Member'),
                        personality=member.get('personality', 'Professional')
                    ) for member in board_data
                ]
                
                # Store company context
                st.session_state.simulation.company_context = f"""
Company: {st.session_state.company_name}
Overview: {st.session_state.company_overview}
Current Problems: {', '.join(company_data.get('current_problems', []))}
Initial Situation: {company_data.get('initial_scenario', '')}
"""
                
                # Initialize other session state
                st.session_state.history = []
                st.session_state.current_scenario = None
                st.session_state.board_opinions = None
                st.session_state.round = 0
                st.session_state.simulation_complete = False
                st.session_state.final_summary = None
                st.session_state.chat_history = {}  # Store chat history per board member
                st.session_state.selected_chat_members = []  # Track selected members for chat
                st.session_state.initialized = True
            
            # Show extraction statistics
            st.success(f"""✅ **Analysis Complete!**

📊 **Extraction Summary:**
- **Metrics Found:** {st.session_state.metrics_count}
- **Board Members Found:** {len(st.session_state.simulation.board_members)}
- **Problems Identified:** {len(company_data.get('current_problems', []))}
- **Total Rounds:** {MAX_ROUNDS}
""")
            
            # Display parsed information in detail
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"🏢 {st.session_state.company_name}")
                st.write(st.session_state.company_overview)
                
                st.write("**Extracted Metrics:**")
                metrics = st.session_state.simulation.metrics
                
                # Display first 10 metrics as preview
                for i, (metric_name, metric_data) in enumerate(list(metrics.items())[:10]):
                    # Handle both dict and simple values
                    if isinstance(metric_data, dict):
                        value = metric_data.get('value', 0)
                        unit = metric_data.get('unit', '')
                        description = metric_data.get('description', '')
                    else:
                        value = metric_data
                        unit = ''
                        description = ''
                    
                    st.metric(
                        metric_name.replace('_', ' ').title(),
                        f"{value}{unit}",
                        help=description if description else None
                    )
                
                if len(metrics) > 10:
                    with st.expander(f"View all {len(metrics)} metrics"):
                        for metric_name, metric_data in list(metrics.items())[10:]:
                            if isinstance(metric_data, dict):
                                value = metric_data.get('value', 0)
                                unit = metric_data.get('unit', '')
                                description = metric_data.get('description', '')
                            else:
                                value = metric_data
                                unit = ''
                                description = ''
                            
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{value}{unit}",
                                help=description if description else None
                            )
            
            with col2:
                st.subheader("👔 Board Members & Leadership")
                for member in st.session_state.simulation.board_members:
                    with st.expander(f"**{member.name}** - {member.role}"):
                        st.write(member.personality)
                
                st.subheader("⚠️ Current Challenges")
                for i, problem in enumerate(company_data.get('current_problems', []), 1):
                    st.write(f"{i}. {problem}")
            
            if st.button("🚀 Start Board Meeting Simulation", type="primary", use_container_width=True):
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ **Error processing PDF:** {str(e)}")
            st.warning("**Troubleshooting tips:**")
            st.write("1. Ensure PDF contains readable text (not scanned images)")
            st.write("2. Try a different PDF viewer to verify content")
            st.write("3. Check if PDF is password protected")
            st.write("4. Verify PDF is not corrupted")
            st.stop()

else:
    # Check if simulation is complete
    if st.session_state.get('simulation_complete', False):
        # Final Summary Screen
        st.title(f"🎯 {st.session_state.company_name} - Simulation Complete!")
        st.success(f"You've completed all {MAX_ROUNDS} rounds of board meetings!")
        
        # Generate final summary if not already done
        if st.session_state.final_summary is None:
            with st.spinner("📊 Analyzing your performance..."):
                st.session_state.final_summary = st.session_state.simulation.generate_final_summary(
                    st.session_state.history,
                    st.session_state.simulation.metrics
                )
        
        # Display final summary
        st.subheader("📈 Executive Performance Review")
        st.write(st.session_state.final_summary)
        
        # Show final metrics comparison
        st.divider()
        st.subheader("📊 Final Metrics Dashboard")
        
        initial_metrics = st.session_state.history[0]['old_metrics']
        final_metrics = st.session_state.simulation.metrics
        
        # Display metrics in columns
        cols = st.columns(4)
        col_idx = 0
        
        for metric_name in final_metrics.keys():
            if metric_name in initial_metrics:
                # Handle both dict and simple values
                if isinstance(initial_metrics[metric_name], dict):
                    initial_val = initial_metrics[metric_name]['value']
                    unit = initial_metrics[metric_name].get('unit', '')
                else:
                    initial_val = initial_metrics[metric_name]
                    unit = ''
                
                if isinstance(final_metrics[metric_name], dict):
                    final_val = final_metrics[metric_name]['value']
                    unit = final_metrics[metric_name].get('unit', '')
                else:
                    final_val = final_metrics[metric_name]
                
                change = final_val - initial_val
                
                with cols[col_idx % 4]:
                    st.metric(
                        metric_name.replace('_', ' ').title(),
                        f"{final_val:.1f}{unit}",
                        f"{change:+.1f}",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
                col_idx += 1
        
        # Full decision history
        st.divider()
        st.subheader("📜 Complete Decision History")
        
        for entry in st.session_state.history:
            with st.expander(f"Round {entry['round']}: {entry['decision'][:60]}...", expanded=False):
                st.write("**Scenario:**", entry['scenario'])
                st.write("**Decision:**", entry['decision'])
                
                st.write("**Metric Changes:**")
                old_m = entry['old_metrics']
                new_m = entry['new_metrics']
                
                cols = st.columns(3)
                col_idx = 0
                for metric_name in list(old_m.keys())[:6]:
                    if isinstance(old_m[metric_name], dict):
                        old_val = old_m[metric_name]['value']
                        new_val = new_m[metric_name]['value']
                        unit = new_m[metric_name].get('unit', '')
                    else:
                        old_val = old_m[metric_name]
                        new_val = new_m[metric_name]
                        unit = ''
                    
                    change = new_val - old_val
                    
                    with cols[col_idx % 3]:
                        st.metric(
                            metric_name.replace('_', ' ').title(),
                            f"{new_val:.1f}{unit}",
                            f"{change:+.1f}",
                            delta_color="normal" if change >= 0 else "inverse"
                        )
                    col_idx += 1
        
        # Reset button
        st.divider()
        if st.button("🔄 Start New Simulation", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    else:
        # Main Simulation Interface
        st.title(f"🏢 {st.session_state.company_name} - Board Meeting")
        st.caption(st.session_state.company_overview)
        
        # Progress indicator
        progress = st.session_state.round / MAX_ROUNDS
        st.progress(progress, text=f"Round {st.session_state.round} of {MAX_ROUNDS}")
        
        # Sidebar - Company Metrics Dashboard
        with st.sidebar:
            st.header("📊 Company Metrics")
            st.caption(f"{st.session_state.metrics_count} metrics tracked")
            
            metrics = st.session_state.simulation.metrics
            
            # Display all metrics
            for metric_name, metric_data in metrics.items():
                # Handle both dict and simple values
                if isinstance(metric_data, dict):
                    value = metric_data.get('value', 0)
                    unit = metric_data.get('unit', '')
                else:
                    value = metric_data
                    unit = ''
                
                st.metric(
                    metric_name.replace('_', ' ').title(),
                    f"{float(value):.1f}{unit}"
                )
            
            st.divider()
            st.caption(f"Round: {st.session_state.round} / {MAX_ROUNDS}")
            
            if st.button("🔄 Reset Simulation"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Start or continue simulation
            if st.session_state.current_scenario is None:
                st.info(f"👋 Welcome, CEO! Ready to face your first board meeting challenge? You'll make {MAX_ROUNDS} critical decisions.")
                
                if st.button("Start Board Meeting", type="primary"):
                    with st.spinner("Preparing board meeting scenario..."):
                        st.session_state.current_scenario = st.session_state.simulation.generate_scenario(
                            "", st.session_state.simulation.metrics, is_first=True, round_num=1
                        )
                        st.session_state.round += 1
                        st.rerun()
            else:
                scenario_data = st.session_state.current_scenario
                
                # Show if this is the final round
                if st.session_state.round == MAX_ROUNDS:
                    st.warning(f"⚠️ **FINAL ROUND ({MAX_ROUNDS}/{MAX_ROUNDS})** - This decision will shape the company's future!")
                else:
                    st.subheader(f"📋 Round {st.session_state.round} of {MAX_ROUNDS}: Decision Required")
                
                st.write("**Scenario:**")
                st.info(scenario_data['scenario'])
                
                st.write("**Question:**")
                st.warning(scenario_data['question'])
                
                # Decision options
                st.write("**Your Options:**")
                
                # Ensure options are strings
                display_options = scenario_data['options']
                if display_options and isinstance(display_options[0], dict):
                    display_options = [str(opt.get('text', opt.get('value', opt))) for opt in display_options]
                else:
                    display_options = [str(opt) for opt in display_options]
                
                selected_option = st.radio(
                    "Select your decision:",
                    display_options,
                    key=f"decision_{st.session_state.round}"
                )
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("📊 Get All Board Input", type="secondary", use_container_width=True):
                        with st.spinner("Consulting all board members..."):
                            # Ensure options are strings before passing
                            options_for_board = scenario_data['options']
                            if options_for_board and isinstance(options_for_board[0], dict):
                                options_for_board = [str(opt.get('text', opt.get('value', opt))) for opt in options_for_board]
                            else:
                                options_for_board = [str(opt) for opt in options_for_board]
                            
                            st.session_state.board_opinions = st.session_state.simulation.generate_board_opinions(
                                scenario_data['scenario'],
                                scenario_data['question'],
                                options_for_board,
                                st.session_state.simulation.metrics
                            )
                        st.rerun()
                
                with col_b:
                    button_label = "✅ Make Final Decision" if st.session_state.round == MAX_ROUNDS else "✅ Make Decision"
                    if st.button(button_label, type="primary", use_container_width=True):
                        with st.spinner("Processing decision impact..."):
                            # Calculate impact
                            new_metrics = st.session_state.simulation.calculate_metric_impact(
                                scenario_data['scenario'],
                                selected_option,
                                st.session_state.simulation.metrics
                            )
                            
                            # Store history
                            st.session_state.history.append({
                                'round': st.session_state.round,
                                'scenario': scenario_data['scenario'],
                                'question': scenario_data['question'],
                                'decision': selected_option,
                                'old_metrics': st.session_state.simulation.metrics.copy(),
                                'new_metrics': new_metrics.copy()
                            })
                            
                            # Update metrics
                            st.session_state.simulation.metrics = new_metrics
                            
                            # Check if this was the final round
                            if st.session_state.round >= MAX_ROUNDS:
                                st.session_state.simulation_complete = True
                                st.session_state.current_scenario = None
                            else:
                                # Generate next scenario
                                context = f"Previous decision: {selected_option}. Impact on company is now reflected in metrics."
                                st.session_state.current_scenario = st.session_state.simulation.generate_scenario(
                                    context, new_metrics, is_first=False, round_num=st.session_state.round + 1
                                )
                                st.session_state.round += 1
                            
                            st.session_state.board_opinions = None
                            # Clear chat when moving to next round
                            st.session_state.chat_history = {}
                            st.session_state.selected_chat_members = []
                            
                        if st.session_state.simulation_complete:
                            st.success("🎉 Simulation complete! Analyzing your performance...")
                        else:
                            st.success("Decision recorded! Moving to next challenge...")
                        st.rerun()
                
                # Selective Board Member Consultation
                st.divider()
                st.subheader("💬 Consult Specific Board Members")
                
                board_member_names = [m.name for m in st.session_state.simulation.board_members]
                
                selected_for_opinion = st.multiselect(
                    "Select board members to get their opinion on the decision:",
                    board_member_names,
                    key="selected_opinion_members"
                )
                
                if selected_for_opinion:
                    if st.button("📊 Get Selected Members' Input", use_container_width=True):
                        with st.spinner(f"Consulting {len(selected_for_opinion)} board member(s)..."):
                            options_for_board = scenario_data['options']
                            if options_for_board and isinstance(options_for_board[0], dict):
                                options_for_board = [str(opt.get('text', opt.get('value', opt))) for opt in options_for_board]
                            else:
                                options_for_board = [str(opt) for opt in options_for_board]
                            
                            # Get opinions only from selected members
                            selected_opinions = st.session_state.simulation.generate_board_opinions(
                                scenario_data['scenario'],
                                scenario_data['question'],
                                options_for_board,
                                st.session_state.simulation.metrics,
                                selected_members=selected_for_opinion
                            )
                            
                            # Merge with existing opinions
                            if st.session_state.board_opinions is None:
                                st.session_state.board_opinions = {}
                            st.session_state.board_opinions.update(selected_opinions)
                        st.rerun()
                
                # Direct Chat with Board Members
                st.divider()
                st.subheader("💭 Chat with Board Members")
                
                selected_for_chat = st.selectbox(
                    "Select a board member to chat with:",
                    [""] + board_member_names,
                    key="selected_chat_member"
                )
                
                if selected_for_chat:
                    # Initialize chat history for this member if needed
                    if selected_for_chat not in st.session_state.chat_history:
                        st.session_state.chat_history[selected_for_chat] = []
                    
                    # Display chat history
                    if st.session_state.chat_history[selected_for_chat]:
                        st.write(f"**Conversation with {selected_for_chat}:**")
                        for msg in st.session_state.chat_history[selected_for_chat]:
                            with st.chat_message("user"):
                                st.write(msg['question'])
                            with st.chat_message("assistant"):
                                st.write(msg['answer'])
                    
                    # Chat input
                    user_question = st.text_input(
                        f"Ask {selected_for_chat} a question:",
                        key=f"chat_input_{selected_for_chat}",
                        placeholder="e.g., What are the risks of this approach?"
                    )
                    
                    if st.button("Send Question", key=f"send_{selected_for_chat}"):
                        if user_question.strip():
                            with st.spinner(f"Getting response from {selected_for_chat}..."):
                                options_for_board = scenario_data['options']
                                if options_for_board and isinstance(options_for_board[0], dict):
                                    options_for_board = [str(opt.get('text', opt.get('value', opt))) for opt in options_for_board]
                                else:
                                    options_for_board = [str(opt) for opt in options_for_board]
                                
                                answer = st.session_state.simulation.chat_with_board_member(
                                    selected_for_chat,
                                    user_question,
                                    scenario_data['scenario'],
                                    scenario_data['question'],
                                    options_for_board,
                                    st.session_state.simulation.metrics,
                                    st.session_state.chat_history[selected_for_chat]
                                )
                                
                                # Store in chat history
                                st.session_state.chat_history[selected_for_chat].append({
                                    'question': user_question,
                                    'answer': answer
                                })
                            st.rerun()
                        else:
                            st.warning("Please enter a question first.")
        
        with col2:
            st.subheader("👔 Board Members")
            
            if st.session_state.board_opinions:
                for member in st.session_state.simulation.board_members:
                    if member.name in st.session_state.board_opinions:
                        with st.expander(f"{member.name} - {member.role}", expanded=True):
                            st.caption(member.personality)
                            st.write(st.session_state.board_opinions[member.name])
                    else:
                        st.write(f"**{member.name}** - {member.role}")
            else:
                st.info("Use the options below to consult board members")
                for member in st.session_state.simulation.board_members:
                    st.write(f"**{member.name}** - {member.role}")
        
        # Decision History
        if st.session_state.history:
            st.divider()
            st.subheader("📜 Decision History")
            
            for entry in reversed(st.session_state.history[-3:]):  # Show last 3
                with st.expander(f"Round {entry['round']}: {entry['decision'][:50]}..."):
                    st.write("**Scenario:**", entry['scenario'])
                    st.write("**Decision:**", entry['decision'])
                    
                    st.write("**Metric Changes:**")
                    
                    # Show changes for key metrics (first 6)
                    old_m = entry['old_metrics']
                    new_m = entry['new_metrics']
                    
                    cols = st.columns(3)
                    col_idx = 0
                    for metric_name in list(old_m.keys())[:6]:  # Show up to 6 metrics
                        # Handle both dict and simple values
                        if isinstance(old_m[metric_name], dict):
                            old_val = old_m[metric_name]['value']
                            new_val = new_m[metric_name]['value']
                            unit = new_m[metric_name].get('unit', '')
                        else:
                            old_val = old_m[metric_name]
                            new_val = new_m[metric_name]
                            unit = ''
                        
                        change = new_val - old_val
                        
                        with cols[col_idx % 3]:
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{new_val:.1f}{unit}",
                                f"{change:+.1f}",
                                delta_color="normal" if change >= 0 else "inverse"
                            )
                        col_idx += 1