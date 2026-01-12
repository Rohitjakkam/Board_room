import streamlit as st
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import PyPDF2
import base64
import time
from enum import Enum

# Configure Gemini
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))

# Constants
MAX_ROUNDS = 8

class DifficultyLevel(Enum):
    """Difficulty levels for adaptive scenarios"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class BoardMember:
    name: str
    role: str
    personality: str

@dataclass
class PerformanceAnalytics:
    """Track user performance for adaptive difficulty"""
    business_scores: List[float] = field(default_factory=list)
    module_scores: List[float] = field(default_factory=list)
    overall_scores: List[float] = field(default_factory=list)
    weak_topics: List[str] = field(default_factory=list)
    strong_topics: List[str] = field(default_factory=list)
    current_difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    hint_count: int = 0
    consultation_count: int = 0

    # Intelligent scenario tracking
    topics_covered: List[str] = field(default_factory=list)  # All topics tested
    topic_scores: Dict[str, List[float]] = field(default_factory=dict)  # Scores per topic
    frameworks_tested: List[str] = field(default_factory=list)  # All frameworks used
    framework_mastery: Dict[str, float] = field(default_factory=dict)  # Mastery level per framework
    scenario_history: List[Dict[str, Any]] = field(default_factory=list)  # Full scenario history
    last_topics_used: List[str] = field(default_factory=list)  # Last 3 topics to avoid repetition

class BoardMeetingSimulation:
    def __init__(self):
        self.board_members = []
        self.company_context = ""
        self.module_content = ""
        self.module_topics = []
        self.metrics = {}
        self.performance = PerformanceAnalytics()
        self.available_hints = []
        
    def extract_pdf_with_gemini(self, pdf_file) -> str:
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
            return self.extract_pdf_with_pypdf2(pdf_file)
    
    def extract_pdf_with_pypdf2(self, pdf_file) -> str:
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
    
    def parse_module_content(self, pdf_text: str) -> Dict:
        """Parse module/course content"""
        if not pdf_text or len(pdf_text.strip()) < 100:
            raise ValueError("Module PDF text is empty")
        
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
    
    def parse_company_data(self, pdf_text: str) -> Dict:
        """Parse company data"""
        if not pdf_text or len(pdf_text.strip()) < 100:
            raise ValueError("Company PDF text is empty")
        
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
    
    def generate_scenario(self, context: str, metrics: Dict, is_first: bool = False, round_num: int = 1) -> Dict:
        """Generate scenario testing business AND module knowledge with adaptive difficulty"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        metrics_summary = self._format_metrics(metrics)
        is_final = (round_num == MAX_ROUNDS)

        # Get current difficulty level
        difficulty = self.performance.current_difficulty

        # Difficulty-specific instructions
        difficulty_instructions = self._get_difficulty_instructions(difficulty)

        # Intelligent topic targeting
        topic_targeting = ""
        if hasattr(self, 'module_topics') and self.module_topics and not is_first:
            target_info = self.identify_target_topics(self.module_topics)

            if target_info['primary_target']:
                topic_targeting = f"""
INTELLIGENT TOPIC TARGETING:
- PRIMARY FOCUS: {target_info['primary_target']}
- REASON: {target_info['reason']}
- PRIORITY: {target_info['priority']}
- AVOID RECENTLY USED: {', '.join(target_info['avoid_topics']) if target_info['avoid_topics'] else 'None'}

The scenario MUST focus on "{target_info['primary_target']}" to address this learning need.
Do NOT use topics: {', '.join(target_info['avoid_topics']) if target_info['avoid_topics'] else 'None'}"""

        module_context = ""
        if self.module_content:
            module_context = f"""
MODULE CONTENT TO TEST:
{self.module_content}

CRITICAL REQUIREMENTS:
- Scenario MUST require applying specific module concepts
- Reference frameworks, theories, or models from the module
- Use module terminology in the scenario
- Options should represent different module-based approaches
- CEO must demonstrate module understanding to succeed

ADAPTIVE DIFFICULTY LEVEL: {difficulty.value.upper()}
{difficulty_instructions}

{topic_targeting}"""
        
        if is_first:
            prompt = f"""Generate first board meeting scenario testing BOTH business acumen AND module knowledge.

Company Context: {self.company_context}

Current Metrics:
{metrics_summary}

{module_context}

This is Round 1 of {MAX_ROUNDS}.

Create a scenario that:
1. Is relevant to the company's actual situation
2. Requires applying specific concepts from the module
3. Tests understanding of module frameworks/theories
4. Includes module terminology
5. Has options based on different module approaches

Return ONLY valid JSON (no markdown):
{{
    "scenario": "Detailed business scenario requiring module knowledge (3-4 sentences)",
    "question": "Decision question that tests module understanding",
    "options": [
        "Option 1 description based on module concept A",
        "Option 2 description based on module concept B",
        "Option 3 description based on module concept C"
    ],
    "module_connection": "Specific module topic/framework being tested"
}}

Options must be STRINGS only. Return ONLY JSON."""

        elif is_final:
            prompt = f"""Generate FINAL ROUND scenario ({MAX_ROUNDS} of {MAX_ROUNDS}) - Ultimate test of business + module mastery.

Company Background: {self.company_context}
Previous Decision Context: {context}

Current Metrics:
{metrics_summary}

{module_context}

Create final scenario requiring:
1. Synthesis of multiple module concepts
2. Application to complex business problem
3. Demonstration of complete module mastery
4. High-stakes business decision

Return ONLY valid JSON (no markdown):
{{
    "scenario": "Epic final scenario (3-4 sentences)",
    "question": "Ultimate strategic question",
    "options": ["Option 1 string", "Option 2 string", "Option 3 string"],
    "module_connection": "Multiple module concepts being tested"
}}

Return ONLY JSON."""

        else:
            prompt = f"""Generate scenario for Round {round_num} of {MAX_ROUNDS}.

Company Background: {self.company_context}
Previous Decision: {context}

Current Metrics:
{metrics_summary}

{module_context}

Create scenario that:
1. Follows naturally from previous decisions
2. Tests different module concepts than previous rounds
3. Increases in complexity/difficulty
4. Requires specific framework application

Return ONLY valid JSON (no markdown):
{{
    "scenario": "Business scenario (3-4 sentences)",
    "question": "Decision question",
    "options": ["Option 1 string", "Option 2 string", "Option 3 string"],
    "module_connection": "Module topic being tested"
}}

Return ONLY JSON."""

        try:
            response = model.generate_content(prompt)
            result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            
            if not result_text.startswith('{'):
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    result_text = result_text[json_start:json_end]
            
            result = json.loads(result_text)
            
            # Ensure options are strings
            if 'options' in result and result['options']:
                result['options'] = [str(opt) if not isinstance(opt, dict) else str(opt.get('text', opt.get('value', opt))) for opt in result['options']]
            
            return result
        except Exception as e:
            st.error(f"Scenario generation error: {e}")
            raise
    
    def evaluate_decision(self, scenario: str, decision: str, scenario_data: Dict, metrics: Dict) -> Dict:
        """Evaluate decision on BOTH business impact AND module knowledge"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        metrics_summary = self._format_metrics(metrics)
        module_connection = scenario_data.get('module_connection', 'General business knowledge')
        
        prompt = f"""Evaluate this CEO decision from TWO perspectives:

1. BUSINESS IMPACT (practical outcomes)
2. MODULE KNOWLEDGE APPLICATION (theoretical understanding)

Company Context: {self.company_context}
Module Content Being Tested: {self.module_content}
Specific Module Topic: {module_connection}

Scenario: {scenario}
Decision Made: {decision}

Current Metrics:
{metrics_summary}

Provide comprehensive evaluation in JSON:

{{
    "business_evaluation": {{
        "score": 75,
        "strengths": ["Specific strength 1", "Specific strength 2"],
        "weaknesses": ["Specific weakness 1"],
        "reasoning": "2-3 sentences explaining business impact",
        "metric_impacts": {{
            "revenue_total": {{"change": 5.0, "reason": "Why revenue changed"}},
            "profit_margin": {{"change": -2.0, "reason": "Why profit changed"}},
            "employee_satisfaction": {{"change": 3.0, "reason": "Why satisfaction changed"}}
        }}
    }},
    "module_evaluation": {{
        "score": 80,
        "concepts_applied_correctly": ["Concept 1 correctly used", "Concept 2 applied well"],
        "concepts_missed": ["Missed concept or opportunity"],
        "framework_usage": "Assessment of how frameworks were applied",
        "theoretical_soundness": "Evaluation of theoretical understanding",
        "reasoning": "2-3 sentences on module knowledge application"
    }},
    "overall_score": 77,
    "feedback": "3-4 sentences of constructive feedback covering both business and module dimensions",
    "better_approach": "2-3 sentences describing the optimal approach"
}}

Be realistic with scores. Perfect (100) requires excellence in both dimensions.
Include metric impacts for relevant metrics from the current metrics list.

Return ONLY valid JSON (no markdown)."""

        try:
            response = model.generate_content(prompt)
            result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            
            if not result_text.startswith('{'):
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    result_text = result_text[json_start:json_end]
            
            result = json.loads(result_text)
            
            # Apply metric changes
            new_metrics = {}
            for name, data in metrics.items():
                if isinstance(data, dict):
                    new_metrics[name] = data.copy()
                else:
                    new_metrics[name] = {'value': data, 'unit': '', 'description': ''}
            
            impacts = result.get('business_evaluation', {}).get('metric_impacts', {})
            for name, change_data in impacts.items():
                if name in new_metrics:
                    old_val = new_metrics[name]['value']
                    change = change_data.get('change', 0)
                    new_val = old_val + change
                    
                    # Don't go below 0 for most metrics
                    if new_val < 0 and not any(x in name.lower() for x in ['loss', 'debt', 'risk', 'turnover']):
                        new_val = 0
                    
                    new_metrics[name]['value'] = new_val
            
            result['new_metrics'] = new_metrics
            return result
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            raise
    
    def generate_board_opinions(self, scenario: str, question: str, options: List, metrics: Dict, selected_members: List[str] = None) -> Dict[str, str]:
        """Board opinions referencing module concepts"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        opinions = {}
        
        metrics_summary = self._format_metrics(metrics)
        options_str = [str(opt) if not isinstance(opt, dict) else str(opt.get('text', str(opt))) for opt in options]
        
        members = self.board_members
        if selected_members:
            members = [m for m in self.board_members if m.name in selected_members]
        
        for member in members:
            prompt = f"""You are {member.name}, {member.role}.
Personality: {member.personality}

Company Context: {self.company_context}
Module Knowledge Base: {self.module_content}

Current Scenario: {scenario}
Question: {question}
Options: {', '.join(options_str)}

Current Metrics:
{metrics_summary}

Provide your opinion (2-3 sentences):
1. State which option you prefer and why
2. Reference relevant module concepts/frameworks in your reasoning
3. Cite specific business metrics that support your view

Stay true to your role and personality."""

            try:
                response = model.generate_content(prompt)
                opinions[member.name] = response.text.strip()
            except:
                opinions[member.name] = "I'm still analyzing the situation..."
        
        return opinions
    
    def chat_with_board_member(self, member_name: str, user_question: str, scenario: str, question: str, options: List, metrics: Dict, history: List[Dict] = None) -> str:
        """Chat with board member about module concepts"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        member = next((m for m in self.board_members if m.name == member_name), None)
        if not member:
            return "Board member not found."
        
        metrics_summary = self._format_metrics(metrics)
        options_str = [str(opt) if not isinstance(opt, dict) else str(opt.get('text', str(opt))) for opt in options]
        
        conv_context = ""
        if history:
            for msg in history:
                conv_context += f"\nCEO: {msg['question']}\n{member_name}: {msg['answer']}\n"
        
        prompt = f"""You are {member.name}, {member.role}.
Personality: {member.personality}

Company Context: {self.company_context}
Module Knowledge: {self.module_content}

Current Scenario: {scenario}
Decision Question: {question}
Options: {', '.join(options_str)}

Current Metrics:
{metrics_summary}

Previous Conversation:
{conv_context}

The CEO is asking you: "{user_question}"

Respond as {member.name}:
- Stay in character with your personality
- Reference module concepts when relevant
- Be specific with data and frameworks
- Keep response to 2-4 sentences unless the question requires more detail"""

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except:
            return "I need a moment to formulate my response..."
    
    def generate_final_summary(self, history: List[Dict], final_metrics: Dict) -> str:
        """Final evaluation of business AND module mastery"""
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        decisions = "\n".join([
            f"Round {e['round']}: {e['decision'][:80]}...\n  Business Score: {e.get('business_score', 0)}/100, Module Score: {e.get('module_score', 0)}/100" 
            for e in history
        ])
        
        initial = history[0]['old_metrics'] if history else {}
        
        metrics_comp = []
        for name in list(final_metrics.keys())[:15]:
            if name in initial:
                init_val = initial[name].get('value', 0) if isinstance(initial[name], dict) else initial[name]
                final_val = final_metrics[name].get('value', 0) if isinstance(final_metrics[name], dict) else final_metrics[name]
                change = final_val - init_val
                unit = final_metrics[name].get('unit', '') if isinstance(final_metrics[name], dict) else ''
                metrics_comp.append(f"{name}: {init_val}{unit} → {final_val}{unit} (change: {change:+.1f})")
        
        avg_biz = sum([e.get('business_score', 0) for e in history]) / len(history) if history else 0
        avg_mod = sum([e.get('module_score', 0) for e in history]) / len(history) if history else 0
        
        prompt = f"""Generate comprehensive CEO Performance Evaluation - {MAX_ROUNDS} rounds completed.

Company Context: {self.company_context}
Module Tested: {self.module_content}

DECISIONS MADE:
{decisions}

AVERAGE SCORES:
- Business Performance: {avg_biz:.1f}/100
- Module Knowledge: {avg_mod:.1f}/100

KEY METRIC CHANGES:
{chr(10).join(metrics_comp)}

Provide detailed evaluation (5-6 paragraphs):

**PARAGRAPH 1-2: BUSINESS PERFORMANCE**
- Analyze overall business outcomes and metric changes
- Evaluate strategic coherence across decisions
- Assess leadership effectiveness and practical judgment

**PARAGRAPH 3-4: MODULE KNOWLEDGE MASTERY**
- Evaluate how well module concepts were applied
- Identify frameworks used correctly vs incorrectly
- Assess theoretical understanding and gaps in knowledge
- Note which concepts were mastered vs missed

**PARAGRAPH 5-6: FINAL ASSESSMENT**
- Provide overall grade (A+ to F) with separate grades for:
  * Business Performance Grade
  * Module Mastery Grade
- List key strengths demonstrated
- Identify areas for improvement
- Give company outlook based on decisions made

Be honest, specific, and constructive. Reference actual decisions and metrics."""

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except:
            return "Unable to generate comprehensive summary at this time."
    
    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for AI prompts"""
        lines = []
        for name, data in metrics.items():
            if isinstance(data, dict):
                value = data.get('value', 0)
                unit = data.get('unit', '')
                desc = data.get('description', '')
            else:
                value = data
                unit = ''
                desc = ''

            line = f"- {name.replace('_', ' ').title()}: {value}{unit}"
            if desc:
                line += f" ({desc})"
            lines.append(line)
        return '\n'.join(lines)

    def update_performance_analytics(self, eval_result: Dict, module_topic: str):
        """Update performance analytics after each decision"""
        biz_score = eval_result.get('business_evaluation', {}).get('score', 0)
        mod_score = eval_result.get('module_evaluation', {}).get('score', 0)
        overall = eval_result.get('overall_score', 0)

        self.performance.business_scores.append(biz_score)
        self.performance.module_scores.append(mod_score)
        self.performance.overall_scores.append(overall)

        # Track weak vs strong topics
        if mod_score < 70:
            if module_topic not in self.performance.weak_topics:
                self.performance.weak_topics.append(module_topic)
        elif mod_score >= 85:
            if module_topic not in self.performance.strong_topics:
                self.performance.strong_topics.append(module_topic)

        # Adjust difficulty based on performance
        self.performance.current_difficulty = self._calculate_adaptive_difficulty()

    def _calculate_adaptive_difficulty(self) -> DifficultyLevel:
        """Calculate appropriate difficulty based on recent performance"""
        if len(self.performance.overall_scores) < 2:
            return DifficultyLevel.INTERMEDIATE

        # Look at last 3 rounds
        recent_scores = self.performance.overall_scores[-3:]
        avg_recent = sum(recent_scores) / len(recent_scores)

        # Look at overall average
        avg_overall = sum(self.performance.overall_scores) / len(self.performance.overall_scores)

        # Check trend
        if len(self.performance.overall_scores) >= 2:
            trend = self.performance.overall_scores[-1] - self.performance.overall_scores[-2]
        else:
            trend = 0

        # Adaptive logic
        if avg_recent >= 85 and avg_overall >= 80:
            return DifficultyLevel.EXPERT
        elif avg_recent >= 75 and trend >= 0:
            return DifficultyLevel.ADVANCED
        elif avg_recent >= 60 or (avg_recent >= 55 and trend > 5):
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.BEGINNER

    def should_provide_hints(self) -> bool:
        """Determine if user needs hints based on performance"""
        if len(self.performance.overall_scores) < 2:
            return False

        recent_avg = sum(self.performance.overall_scores[-2:]) / 2
        return recent_avg < 65 or self.performance.overall_scores[-1] < 60

    def generate_contextual_hints(self, scenario_data: Dict) -> List[str]:
        """Generate hints for struggling users"""
        hints = []
        module_topic = scenario_data.get('module_connection', '')

        # Check if this is a weak area
        if module_topic in self.performance.weak_topics:
            hints.append(f"💡 **Topic Review**: This scenario tests {module_topic}, which you've found challenging. Review the key frameworks related to this topic.")

        # Performance-based hints
        if len(self.performance.module_scores) >= 2:
            recent_mod_avg = sum(self.performance.module_scores[-2:]) / 2
            if recent_mod_avg < 65:
                hints.append("📚 **Module Focus**: Your theoretical application needs improvement. Try to identify which framework from the module best fits this scenario.")

        if len(self.performance.business_scores) >= 2:
            recent_biz_avg = sum(self.performance.business_scores[-2:]) / 2
            if recent_biz_avg < 65:
                hints.append("📊 **Business Metrics**: Consider which metrics will be most impacted by each option. Use data-driven reasoning.")

        # Consultation reminder
        if self.performance.consultation_count < len(self.performance.overall_scores):
            hints.append("👥 **Board Wisdom**: You haven't consulted your board much. They can provide valuable insights on frameworks and business impact.")

        return hints

    def get_difficulty_descriptor(self) -> Dict[str, str]:
        """Get user-friendly difficulty information"""
        difficulty_map = {
            DifficultyLevel.BEGINNER: {
                "name": "Foundation Builder",
                "description": "Scenarios focus on basic concepts with clear framework applications",
                "color": "🟢",
                "emoji": "🌱"
            },
            DifficultyLevel.INTERMEDIATE: {
                "name": "Skill Developer",
                "description": "Balanced scenarios testing understanding and application",
                "color": "🔵",
                "emoji": "📈"
            },
            DifficultyLevel.ADVANCED: {
                "name": "Strategic Thinker",
                "description": "Complex scenarios requiring synthesis of multiple concepts",
                "color": "🟠",
                "emoji": "🎯"
            },
            DifficultyLevel.EXPERT: {
                "name": "Master Strategist",
                "description": "Challenging scenarios with nuanced trade-offs and deep analysis",
                "color": "🔴",
                "emoji": "👑"
            }
        }
        return difficulty_map.get(self.performance.current_difficulty, difficulty_map[DifficultyLevel.INTERMEDIATE])

    def _get_difficulty_instructions(self, difficulty: DifficultyLevel) -> str:
        """Get AI prompt instructions based on difficulty level"""
        instructions = {
            DifficultyLevel.BEGINNER: """
BEGINNER LEVEL ADJUSTMENTS:
- Focus on ONE primary framework/concept (clearly stated)
- Scenario should have obvious clues about which framework applies
- Options should be clearly differentiated (good/better/best)
- Avoid ambiguous trade-offs
- Use straightforward language
- Provide clear cause-and-effect relationships""",

            DifficultyLevel.INTERMEDIATE: """
INTERMEDIATE LEVEL ADJUSTMENTS:
- Test 1-2 frameworks/concepts
- Some ambiguity in best approach is acceptable
- Options should have reasonable trade-offs
- Require moderate analysis
- Mix of obvious and subtle clues""",

            DifficultyLevel.ADVANCED: """
ADVANCED LEVEL ADJUSTMENTS:
- Require synthesis of 2-3 frameworks/concepts
- Multiple valid approaches with different trade-offs
- Include nuanced business considerations
- Require deep analysis of metrics and context
- Test ability to prioritize competing objectives""",

            DifficultyLevel.EXPERT: """
EXPERT LEVEL ADJUSTMENTS:
- Complex multi-framework scenarios
- No single "correct" answer - all options have merit
- Require weighing subtle trade-offs and long-term implications
- Test mastery of theory AND practical judgment
- Include edge cases and exceptional situations
- Demand strategic thinking beyond module content"""
        }
        return instructions.get(difficulty, instructions[DifficultyLevel.INTERMEDIATE])

    def get_personalized_learning_path(self) -> Dict[str, Any]:
        """Generate personalized recommendations based on performance"""
        recommendations = {
            "focus_areas": [],
            "review_topics": self.performance.weak_topics[:],
            "strengths": self.performance.strong_topics[:],
            "suggested_actions": [],
            "difficulty_adjustment": self.performance.current_difficulty.value
        }

        # Analyze performance gaps
        if len(self.performance.business_scores) >= 3:
            avg_biz = sum(self.performance.business_scores) / len(self.performance.business_scores)
            avg_mod = sum(self.performance.module_scores) / len(self.performance.module_scores)

            gap = abs(avg_biz - avg_mod)
            if gap > 15:
                if avg_biz < avg_mod:
                    recommendations["focus_areas"].append("Business Decision-Making")
                    recommendations["suggested_actions"].append("Focus on practical application and metric analysis")
                else:
                    recommendations["focus_areas"].append("Module Theory Application")
                    recommendations["suggested_actions"].append("Review course frameworks and strengthen theoretical foundations")

        # Performance trend analysis
        if len(self.performance.overall_scores) >= 3:
            recent_trend = self.performance.overall_scores[-1] - self.performance.overall_scores[-3]
            if recent_trend < -10:
                recommendations["suggested_actions"].append("⚠️ Performance declining - review recent feedback carefully")
            elif recent_trend > 10:
                recommendations["suggested_actions"].append("✅ Great improvement! Continue current approach")

        # Consultation behavior
        if self.performance.consultation_count < len(self.performance.overall_scores) * 0.5:
            recommendations["suggested_actions"].append("💡 Consult board members more frequently for better insights")

        return recommendations

    def update_scenario_analytics(self, scenario_data: Dict, score: float):
        """Track scenario usage and topic performance"""
        topic = scenario_data.get('module_connection', 'N/A')

        # Track topic coverage
        if topic not in self.performance.topics_covered:
            self.performance.topics_covered.append(topic)

        # Track topic scores
        if topic not in self.performance.topic_scores:
            self.performance.topic_scores[topic] = []
        self.performance.topic_scores[topic].append(score)

        # Update last topics (for repetition avoidance)
        self.performance.last_topics_used.append(topic)
        if len(self.performance.last_topics_used) > 3:
            self.performance.last_topics_used.pop(0)

        # Store scenario in history
        self.performance.scenario_history.append({
            'topic': topic,
            'score': score,
            'round': len(self.performance.scenario_history) + 1
        })

    def calculate_framework_mastery(self, topic: str) -> float:
        """Calculate mastery level for a specific topic/framework (0-100)"""
        if topic not in self.performance.topic_scores or not self.performance.topic_scores[topic]:
            return 0.0

        scores = self.performance.topic_scores[topic]
        # Weight recent scores more heavily
        if len(scores) == 1:
            return scores[0]
        elif len(scores) == 2:
            return (scores[0] * 0.4 + scores[1] * 0.6)
        else:
            # Recent scores weighted more
            recent = scores[-2:]
            older = scores[:-2]
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older) if older else recent_avg
            return older_avg * 0.3 + recent_avg * 0.7

    def identify_target_topics(self, all_available_topics: List[str]) -> Dict[str, Any]:
        """Intelligently identify which topics to target in next scenario"""
        target_info = {
            'primary_target': None,
            'reason': '',
            'avoid_topics': self.performance.last_topics_used[:],
            'priority': 'balanced'
        }

        # Not enough data yet - random selection
        if len(self.performance.overall_scores) < 1:
            target_info['reason'] = 'Initial exploration'
            target_info['priority'] = 'exploration'
            return target_info

        # Identify weak topics (mastery < 65)
        weak_untested = []
        weak_tested = []

        for topic_name in all_available_topics:
            # Skip recently used topics
            if topic_name in self.performance.last_topics_used:
                continue

            mastery = self.calculate_framework_mastery(topic_name)

            if topic_name not in self.performance.topics_covered:
                weak_untested.append(topic_name)
            elif mastery < 65:
                weak_tested.append((topic_name, mastery))

        # Priority 1: Target weak tested topics (need remediation)
        if weak_tested and len(self.performance.overall_scores) >= 2:
            # Sort by lowest mastery
            weak_tested.sort(key=lambda x: x[1])
            target_info['primary_target'] = weak_tested[0][0]
            target_info['reason'] = f'Remediation needed (mastery: {weak_tested[0][1]:.0f}%)'
            target_info['priority'] = 'remediation'
            return target_info

        # Priority 2: Explore untested topics (coverage)
        if weak_untested:
            target_info['primary_target'] = weak_untested[0]
            target_info['reason'] = 'Unexplored topic - assessing baseline'
            target_info['priority'] = 'exploration'
            return target_info

        # Priority 3: Reinforce borderline topics (65-80)
        borderline = []
        for topic_name in all_available_topics:
            if topic_name in self.performance.last_topics_used:
                continue
            mastery = self.calculate_framework_mastery(topic_name)
            if 65 <= mastery < 80:
                borderline.append((topic_name, mastery))

        if borderline:
            # Target lowest borderline topic
            borderline.sort(key=lambda x: x[1])
            target_info['primary_target'] = borderline[0][0]
            target_info['reason'] = f'Reinforcement (mastery: {borderline[0][1]:.0f}%)'
            target_info['priority'] = 'reinforcement'
            return target_info

        # Default: Balanced rotation (avoid last 3)
        available = [t for t in all_available_topics if t not in self.performance.last_topics_used]
        if available:
            target_info['primary_target'] = available[0]
            target_info['reason'] = 'Balanced rotation'
            target_info['priority'] = 'balanced'

        return target_info

    def get_scenario_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of intelligent scenario targeting"""
        summary = {
            'topics_covered': len(self.performance.topics_covered),
            'total_available': 0,  # Will be set by caller
            'coverage_percentage': 0,
            'weakest_topic': None,
            'strongest_topic': None,
            'next_target_reason': ''
        }

        # Find weakest topic
        if self.performance.topic_scores:
            topic_masteries = []
            for topic, scores in self.performance.topic_scores.items():
                mastery = self.calculate_framework_mastery(topic)
                topic_masteries.append((topic, mastery))

            if topic_masteries:
                topic_masteries.sort(key=lambda x: x[1])
                summary['weakest_topic'] = {
                    'name': topic_masteries[0][0],
                    'mastery': topic_masteries[0][1]
                }
                summary['strongest_topic'] = {
                    'name': topic_masteries[-1][0],
                    'mastery': topic_masteries[-1][1]
                }

        return summary

# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="CEO Board Meeting + Module Assessment",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.simulation = BoardMeetingSimulation()
    st.session_state.module_uploaded = False

# ==================== UPLOAD SECTION ====================

if not st.session_state.initialized:
    st.title("🏢 CEO Board Meeting Simulation + Module Assessment")
    
    st.markdown("""
    ### 📚 Dual-Dimension Business Simulation
    
    This simulation tests you on **TWO critical dimensions**:
    
    1. **🎯 Business Decision-Making** - Navigate real company challenges and drive results
    2. **📖 Module Knowledge Application** - Apply course concepts, theories, and frameworks
    
    **How it works:**
    - Upload company document (financials, strategy, metrics)
    - Upload module content (course material, textbooks, lectures)
    - Face 8 rounds of critical decisions
    - Get scored on both business impact AND theoretical understanding
    - Receive comprehensive performance evaluation
    
    **Upload both documents to begin:**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Step 1: Company Document")
        st.info("PDF containing company details, financial metrics, leadership team, and current challenges")
        company_file = st.file_uploader("Upload Company PDF", type=['pdf'], key="company_pdf")
        
        if company_file:
            st.success(f"✅ Uploaded: {company_file.name}")
    
    with col2:
        st.subheader("📚 Step 2: Module Content")
        st.info("PDF with course material, theories, frameworks, concepts, and learning objectives")
        module_file = st.file_uploader("Upload Module/Course PDF", type=['pdf'], key="module_pdf")
        
        if module_file:
            st.success(f"✅ Uploaded: {module_file.name}")
    
    if company_file is not None and module_file is not None:
        st.divider()
        
        if st.button("🚀 Process Documents & Start Simulation", type="primary", use_container_width=True):
            try:
                # Extract Company Data
                with st.spinner("📖 Step 1/5: Extracting company document..."):
                    company_text = st.session_state.simulation.extract_pdf_with_gemini(company_file)
                    st.success(f"✅ Extracted {len(company_text):,} characters")
                
                # Extract Module Data
                with st.spinner("📚 Step 2/5: Extracting module content..."):
                    module_text = st.session_state.simulation.extract_pdf_with_gemini(module_file)
                    st.success(f"✅ Extracted {len(module_text):,} characters")
                
                # Parse Company
                with st.spinner("🤖 Step 3/5: AI analyzing company data (30-60 sec)..."):
                    company_data = st.session_state.simulation.parse_company_data(company_text)
                    st.success("✅ Company analysis complete")
                
                # Parse Module
                with st.spinner("🧠 Step 4/5: AI analyzing module content (30-60 sec)..."):
                    module_data = st.session_state.simulation.parse_module_content(module_text)
                    st.success("✅ Module analysis complete")
                
                with st.spinner("⚙️ Step 5/5: Initializing simulation..."):
                    # Store company data
                    st.session_state.company_name = company_data.get('company_name', 'Your Company')
                    st.session_state.company_overview = company_data.get('company_overview', '')
                    st.session_state.simulation.metrics = company_data.get('metrics', {})
                    
                    st.session_state.simulation.board_members = [
                        BoardMember(
                            name=m.get('name', 'Unknown'),
                            role=m.get('role', 'Board Member'),
                            personality=m.get('personality', 'Professional')
                        ) for m in company_data.get('board_members', [])
                    ]
                    
                    st.session_state.simulation.company_context = f"""
Company: {st.session_state.company_name}
Overview: {st.session_state.company_overview}
Current Problems: {', '.join(company_data.get('current_problems', []))}
Initial Situation: {company_data.get('initial_scenario', '')}"""
                    
                    # Store module data
                    st.session_state.module_name = module_data.get('module_name', 'Course Module')
                    st.session_state.module_subject = module_data.get('subject_area', 'Business')
                    
                    # Format module content
                    topics_text = '\n'.join([f"- {t['name']}: {t['description']}" for t in module_data.get('topics', [])])
                    frameworks_text = '\n'.join([f"- {f['name']}: {f['description']}" for f in module_data.get('frameworks', [])])
                    terms_text = '\n'.join([f"- {k}: {v}" for k, v in list(module_data.get('key_terms', {}).items())[:30]])
                    
                    st.session_state.simulation.module_content = f"""
MODULE: {module_data.get('module_name')}
SUBJECT: {module_data.get('subject_area')}

LEARNING OBJECTIVES:
{chr(10).join(['- ' + o for o in module_data.get('learning_objectives', [])])}

KEY TOPICS:
{topics_text}

FRAMEWORKS & MODELS:
{frameworks_text}

KEY TERMINOLOGY:
{terms_text}

ASSESSMENT CRITERIA:
{chr(10).join(['- ' + c for c in module_data.get('assessment_criteria', [])])}"""

                    # Extract topic names for intelligent targeting
                    topic_names = [t['name'] for t in module_data.get('topics', [])]
                    framework_names = [f['name'] for f in module_data.get('frameworks', [])]
                    # Combine topics and frameworks as targets
                    st.session_state.simulation.module_topics = topic_names + framework_names

                    st.session_state.module_data = module_data
                    
                    # Initialize simulation state
                    st.session_state.history = []
                    st.session_state.current_scenario = None
                    st.session_state.board_opinions = None
                    st.session_state.round = 0
                    st.session_state.simulation_complete = False
                    st.session_state.chat_history = {}
                    st.session_state.selected_chat_members = []
                    st.session_state.initialized = True
                
                st.success("✅ Simulation ready!")
                time.sleep(1)
                st.rerun()
                    
            except Exception as e:
                st.error(f"❌ **Error processing documents:** {str(e)}")
                st.warning("**Troubleshooting tips:**")
                st.write("1. Ensure PDFs contain readable text (not scanned images)")
                st.write("2. Verify PDFs are not password protected")
                st.write("3. Check file size (very large PDFs may timeout)")
                st.write("4. Try re-uploading the files")

# ==================== SIMULATION COMPLETE SCREEN ====================

elif st.session_state.get('simulation_complete', False):
    st.title(f"🎯 {st.session_state.company_name} - Simulation Complete!")
    st.success(f"Congratulations! You've completed all {MAX_ROUNDS} rounds of board meetings!")
    
    # Generate final summary if not done
    if 'final_summary' not in st.session_state or st.session_state.final_summary is None:
        with st.spinner("📊 Generating comprehensive performance analysis..."):
            st.session_state.final_summary = st.session_state.simulation.generate_final_summary(
                st.session_state.history,
                st.session_state.simulation.metrics
            )
    
    # Display final summary
    st.subheader("📈 Executive Performance Review")
    st.write(st.session_state.final_summary)
    
    # Calculate and display average scores
    if st.session_state.history:
        avg_biz = sum([e.get('business_score', 0) for e in st.session_state.history]) / len(st.session_state.history)
        avg_mod = sum([e.get('module_score', 0) for e in st.session_state.history]) / len(st.session_state.history)
        avg_all = sum([e.get('overall_score', 0) for e in st.session_state.history]) / len(st.session_state.history)
        
        st.divider()
        st.subheader("📊 Overall Scores")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Business Performance", f"{avg_biz:.1f}/100")
            if avg_biz >= 80:
                st.success("Excellent business acumen!")
            elif avg_biz >= 70:
                st.info("Good business performance")
            else:
                st.warning("Room for improvement")
        
        with col2:
            st.metric("📚 Module Mastery", f"{avg_mod:.1f}/100")
            if avg_mod >= 80:
                st.success("Strong theoretical understanding!")
            elif avg_mod >= 70:
                st.info("Good knowledge application")
            else:
                st.warning("Review module concepts")
        
        with col3:
            st.metric("⭐ Overall Score", f"{avg_all:.1f}/100")
            if avg_all >= 85:
                st.success("Outstanding performance!")
            elif avg_all >= 75:
                st.info("Solid performance")
            else:
                st.warning("Keep practicing")
    
    # Final metrics comparison
    st.divider()
    st.subheader("📊 Final Metrics Dashboard")
    st.caption("Initial → Final (Change)")
    
    initial = st.session_state.history[0]['old_metrics']
    final = st.session_state.simulation.metrics
    
    cols = st.columns(4)
    idx = 0
    
    for name in final.keys():
        if name in initial:
            init_val = initial[name].get('value', 0) if isinstance(initial[name], dict) else initial[name]
            final_val = final[name].get('value', 0) if isinstance(final[name], dict) else final[name]
            unit = final[name].get('unit', '') if isinstance(final[name], dict) else ''
            change = final_val - init_val
            
            with cols[idx % 4]:
                st.metric(
                    name.replace('_', ' ').title(),
                    f"{final_val:.1f}{unit}",
                    f"{change:+.1f}",
                    delta_color="normal" if change >= 0 else "inverse"
                )
            idx += 1
    
    # Complete decision history
    st.divider()
    st.subheader("📜 Complete Decision History")
    
    for e in st.session_state.history:
        with st.expander(f"Round {e['round']}: {e['decision'][:60]}... (Overall: {e.get('overall_score', 0)}/100)", expanded=False):
            st.write("**Scenario:**", e['scenario'])
            st.write("**Module Topic Tested:**", e.get('module_connection', 'N/A'))
            st.write("**Your Decision:**", e['decision'])
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Business Score", f"{e.get('business_score', 0)}/100")
            with col_b:
                st.metric("Module Score", f"{e.get('module_score', 0)}/100")
            with col_c:
                st.metric("Overall Score", f"{e.get('overall_score', 0)}/100")
            
            if 'feedback' in e and e['feedback']:
                st.info(f"**Feedback:** {e['feedback']}")
            
            if 'better_approach' in e and e['better_approach']:
                st.success(f"**💡 Better Approach:** {e['better_approach']}")
    
    # Download functionality
    st.divider()
    st.subheader("📥 Download Your Report")
    
    # Prepare report content
    report_content = f"""# CEO BOARD MEETING SIMULATION - FINAL REPORT

**Company:** {st.session_state.company_name}
**Module:** {st.session_state.module_name} ({st.session_state.module_subject})
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Rounds Completed:** {MAX_ROUNDS}

---

## EXECUTIVE SUMMARY

{st.session_state.final_summary}

---

## OVERALL PERFORMANCE

- **Business Performance:** {avg_biz:.1f}/100
- **Module Knowledge:** {avg_mod:.1f}/100
- **Overall Score:** {avg_all:.1f}/100

---

## DECISION HISTORY

"""
    
    for e in st.session_state.history:
        report_content += f"""
### Round {e['round']} - {e.get('module_connection', 'N/A')}

**Scenario:** {e['scenario']}

**Question:** {e['question']}

**Your Decision:** {e['decision']}

**Scores:**
- Business: {e.get('business_score', 0)}/100
- Module: {e.get('module_score', 0)}/100
- Overall: {e.get('overall_score', 0)}/100

**Feedback:** {e.get('feedback', 'N/A')}

**Better Approach:** {e.get('better_approach', 'N/A')}

---

"""
    
    report_content += f"""
## FINAL METRICS COMPARISON

"""
    
    for name in final.keys():
        if name in initial:
            init_val = initial[name].get('value', 0) if isinstance(initial[name], dict) else initial[name]
            final_val = final[name].get('value', 0) if isinstance(final[name], dict) else final[name]
            unit = final[name].get('unit', '') if isinstance(final[name], dict) else ''
            change = final_val - init_val
            report_content += f"- **{name.replace('_', ' ').title()}:** {init_val}{unit} → {final_val}{unit} ({change:+.1f})\n"
    
    report_content += "\n---\n\n*Report generated by CEO Board Meeting Simulation + Module Assessment*"
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.download_button(
            label="📄 Download Report (Markdown)",
            data=report_content,
            file_name=f"CEO_Report_{st.session_state.company_name.replace(' ', '_')}_{time.strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary"
        )
    
    with col_dl2:
        # JSON export
        json_export = {
            "company": st.session_state.company_name,
            "module": st.session_state.module_name,
            "date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "scores": {
                "business_avg": round(avg_biz, 2),
                "module_avg": round(avg_mod, 2),
                "overall_avg": round(avg_all, 2)
            },
            "decisions": [
                {
                    "round": e['round'],
                    "scenario": e['scenario'],
                    "question": e['question'],
                    "decision": e['decision'],
                    "module_topic": e.get('module_connection', 'N/A'),
                    "scores": {
                        "business": e.get('business_score', 0),
                        "module": e.get('module_score', 0),
                        "overall": e.get('overall_score', 0)
                    },
                    "feedback": e.get('feedback', ''),
                    "better_approach": e.get('better_approach', '')
                }
                for e in st.session_state.history
            ]
        }
        
        st.download_button(
            label="📊 Download Data (JSON)",
            data=json.dumps(json_export, indent=2),
            file_name=f"CEO_Data_{st.session_state.company_name.replace(' ', '_')}_{time.strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Reset button
    st.divider()
    if st.button("🔄 Start New Simulation", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==================== MAIN SIMULATION INTERFACE ====================

else:
    st.title(f"🏢 {st.session_state.company_name} - Board Meeting Simulation")
    st.caption(st.session_state.company_overview)
    st.info(f"📚 Testing Module: **{st.session_state.module_name}** ({st.session_state.module_subject})")
    
    # Progress bar
    progress = st.session_state.round / MAX_ROUNDS
    st.progress(progress, text=f"Round {st.session_state.round} of {MAX_ROUNDS}")
    
    # ==================== SIDEBAR ====================
    
    with st.sidebar:
        st.header("📊 Company Metrics")
        
        metrics = st.session_state.simulation.metrics
        
        # Display first 12 metrics
        for i, (name, data) in enumerate(list(metrics.items())[:12]):
            val = data.get('value', 0) if isinstance(data, dict) else data
            unit = data.get('unit', '') if isinstance(data, dict) else ''
            st.metric(
                name.replace('_', ' ').title(),
                f"{float(val):.1f}{unit}"
            )
        
        # Show remaining metrics in expander
        if len(metrics) > 12:
            with st.expander(f"View all {len(metrics)} metrics"):
                for name, data in list(metrics.items())[12:]:
                    val = data.get('value', 0) if isinstance(data, dict) else data
                    unit = data.get('unit', '') if isinstance(data, dict) else ''
                    st.metric(
                        name.replace('_', ' ').title(),
                        f"{float(val):.1f}{unit}"
                    )
        
        # Module reference
        st.divider()
        with st.expander("📖 Module Reference"):
            st.caption(f"**{st.session_state.module_name}**")
            
            if st.session_state.module_data.get('topics'):
                st.write("**Key Topics:**")
                for i, topic in enumerate(st.session_state.module_data['topics'][:8], 1):
                    st.caption(f"{i}. {topic['name']}")
                
                if len(st.session_state.module_data['topics']) > 8:
                    with st.expander(f"All {len(st.session_state.module_data['topics'])} topics"):
                        for i, topic in enumerate(st.session_state.module_data['topics'], 1):
                            st.caption(f"{i}. {topic['name']}")
            
            if st.session_state.module_data.get('frameworks'):
                st.write("**Frameworks:**")
                for fw in st.session_state.module_data['frameworks'][:5]:
                    st.caption(f"• {fw['name']}")
        
        # Performance tracking
        if st.session_state.history:
            st.divider()
            st.caption("**Current Performance**")
            
            avg_biz = sum([e.get('business_score', 0) for e in st.session_state.history]) / len(st.session_state.history)
            avg_mod = sum([e.get('module_score', 0) for e in st.session_state.history]) / len(st.session_state.history)
            avg_all = sum([e.get('overall_score', 0) for e in st.session_state.history]) / len(st.session_state.history)
            
            st.metric("Avg Business", f"{avg_biz:.0f}/100")
            st.metric("Avg Module", f"{avg_mod:.0f}/100")
            st.metric("Avg Overall", f"{avg_all:.0f}/100")
            
            # Trend indicator
            if len(st.session_state.history) >= 2:
                recent = sum([e.get('overall_score', 0) for e in st.session_state.history[-2:]]) / 2
                previous = sum([e.get('overall_score', 0) for e in st.session_state.history[:-2]]) / max(len(st.session_state.history) - 2, 1)
                trend = recent - previous

                if trend > 5:
                    st.success("📈 Improving!")
                elif trend < -5:
                    st.warning("📉 Declining")
                else:
                    st.info("➡️ Steady")

            # Personalized Learning Path
            if len(st.session_state.history) >= 2:
                st.divider()
                with st.expander("🎯 Your Learning Path"):
                    learning_path = st.session_state.simulation.get_personalized_learning_path()

                    # Current difficulty
                    diff_info = st.session_state.simulation.get_difficulty_descriptor()
                    st.caption(f"**Current Level:** {diff_info['emoji']} {diff_info['name']}")
                    st.caption(f"_{diff_info['description']}_")

                    # Topic coverage
                    if st.session_state.simulation.module_topics:
                        total_topics = len(st.session_state.simulation.module_topics)
                        covered = len(st.session_state.simulation.performance.topics_covered)
                        coverage = (covered / total_topics * 100) if total_topics > 0 else 0
                        st.caption(f"**📊 Coverage:** {covered}/{total_topics} topics ({coverage:.0f}%)")

                    # Weak topics
                    if learning_path['review_topics']:
                        st.write("**📝 Topics to Review:**")
                        for topic in learning_path['review_topics'][:3]:
                            mastery = st.session_state.simulation.calculate_framework_mastery(topic)
                            st.caption(f"• {topic} ({mastery:.0f}% mastery)")

                    # Strengths
                    if learning_path['strengths']:
                        st.write("**✅ Strong Areas:**")
                        for topic in learning_path['strengths'][:3]:
                            mastery = st.session_state.simulation.calculate_framework_mastery(topic)
                            st.caption(f"• {topic} ({mastery:.0f}% mastery)")

                    # Suggested actions
                    if learning_path['suggested_actions']:
                        st.write("**💡 Recommendations:**")
                        for action in learning_path['suggested_actions']:
                            st.caption(f"• {action}")

        st.divider()
        st.caption(f"Round: {st.session_state.round} / {MAX_ROUNDS}")
        
        if st.button("🔄 Reset Simulation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # ==================== MAIN CONTENT ====================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Start simulation or show scenario
        if st.session_state.current_scenario is None:
            st.info(f"👋 **Welcome, CEO!** \n\nYou'll face {MAX_ROUNDS} critical decisions that test both your business judgment and knowledge of **{st.session_state.module_name}**. Each decision will be scored on:\n\n- 🎯 **Business Impact** - How it affects company metrics\n- 📚 **Module Knowledge** - How well you apply course concepts\n\nConsult your board members for insights!")
            
            if st.button("🎬 Start Board Meeting", type="primary", use_container_width=True):
                with st.spinner("🤔 Preparing your first challenge..."):
                    st.session_state.current_scenario = st.session_state.simulation.generate_scenario(
                        "", st.session_state.simulation.metrics, is_first=True, round_num=1
                    )
                    st.session_state.round += 1
                    time.sleep(1)
                    st.rerun()
        
        else:
            scenario_data = st.session_state.current_scenario
            
            # Show round header
            if st.session_state.round == MAX_ROUNDS:
                st.warning(f"⚠️ **FINAL ROUND ({MAX_ROUNDS}/{MAX_ROUNDS})** - Your defining moment as CEO!")
            else:
                st.subheader(f"📋 Round {st.session_state.round} of {MAX_ROUNDS}: Critical Decision")
            
            # Show difficulty level
            diff_info = st.session_state.simulation.get_difficulty_descriptor()
            col_diff1, col_diff2 = st.columns([3, 1])
            with col_diff1:
                # Show module connection with targeting reason
                if 'module_connection' in scenario_data:
                    topic = scenario_data['module_connection']
                    topic_text = f"📚 **Module Topic:** {topic}"

                    # Show why this topic was chosen (if available)
                    if st.session_state.round > 1 and hasattr(st.session_state.simulation, 'module_topics'):
                        if topic in st.session_state.simulation.performance.topics_covered:
                            mastery = st.session_state.simulation.calculate_framework_mastery(topic)
                            if mastery < 65:
                                topic_text += f" • 🎯 *Remediation focus ({mastery:.0f}% mastery)*"
                            elif mastery < 80:
                                topic_text += f" • 📈 *Reinforcement ({mastery:.0f}% mastery)*"
                        else:
                            topic_text += " • 🔍 *New topic exploration*"

                    st.info(topic_text)
            with col_diff2:
                st.metric(
                    label="Difficulty",
                    value=f"{diff_info['emoji']} {diff_info['name'].split()[0]}",
                    help=diff_info['description']
                )

            # Show adaptive hints if struggling
            if st.session_state.simulation.should_provide_hints():
                hints = st.session_state.simulation.generate_contextual_hints(scenario_data)
                if hints:
                    with st.expander("💡 Helpful Hints (Click to expand)", expanded=False):
                        for hint in hints:
                            st.markdown(hint)
                        st.caption("_Hints are provided when performance indicates you might benefit from additional guidance._")

            # Display scenario
            st.write("**Business Scenario:**")
            st.info(scenario_data['scenario'])
            
            st.write("**Decision Required:**")
            st.warning(scenario_data['question'])
            
            # Decision options
            st.write("**Your Options:**")
            
            options = scenario_data['options']
            if options and isinstance(options[0], dict):
                options = [str(o.get('text', o.get('value', o))) for o in options]
            else:
                options = [str(o) for o in options]
            
            selected = st.radio(
                "Select your decision:",
                options,
                key=f"decision_{st.session_state.round}",
                label_visibility="collapsed"
            )
            
            # Action buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("📊 Get All Board Input", type="secondary", use_container_width=True):
                    with st.spinner("🗣️ Consulting all board members..."):
                        opts = scenario_data['options']
                        if opts and isinstance(opts[0], dict):
                            opts = [str(o.get('text', o.get('value', o))) for o in opts]
                        else:
                            opts = [str(o) for o in opts]

                        st.session_state.board_opinions = st.session_state.simulation.generate_board_opinions(
                            scenario_data['scenario'],
                            scenario_data['question'],
                            opts,
                            st.session_state.simulation.metrics
                        )
                        # Track consultation
                        st.session_state.simulation.performance.consultation_count += 1
                    st.rerun()
            
            with col_b:
                button_label = "✅ Make Final Decision" if st.session_state.round == MAX_ROUNDS else "✅ Make Decision"
                if st.button(button_label, type="primary", use_container_width=True):
                    with st.spinner("⚙️ Evaluating your decision on both business and module dimensions..."):
                        # Evaluate decision
                        eval_result = st.session_state.simulation.evaluate_decision(
                            scenario_data['scenario'],
                            selected,
                            scenario_data,
                            st.session_state.simulation.metrics
                        )
                        
                        new_metrics = eval_result['new_metrics']
                        
                        # Store in history
                        st.session_state.history.append({
                            'round': st.session_state.round,
                            'scenario': scenario_data['scenario'],
                            'question': scenario_data['question'],
                            'decision': selected,
                            'module_connection': scenario_data.get('module_connection', 'N/A'),
                            'old_metrics': st.session_state.simulation.metrics.copy(),
                            'new_metrics': new_metrics.copy(),
                            'business_score': eval_result.get('business_evaluation', {}).get('score', 0),
                            'module_score': eval_result.get('module_evaluation', {}).get('score', 0),
                            'overall_score': eval_result.get('overall_score', 0),
                            'feedback': eval_result.get('feedback', ''),
                            'better_approach': eval_result.get('better_approach', ''),
                            'evaluation': eval_result
                        })
                        
                        # Update metrics
                        st.session_state.simulation.metrics = new_metrics

                        # Store previous difficulty
                        prev_difficulty = st.session_state.simulation.performance.current_difficulty

                        # Update performance analytics
                        st.session_state.simulation.update_performance_analytics(
                            eval_result,
                            scenario_data.get('module_connection', 'N/A')
                        )

                        # Update scenario analytics (intelligent targeting)
                        overall_score = eval_result.get('overall_score', 0)
                        st.session_state.simulation.update_scenario_analytics(scenario_data, overall_score)

                        # Check if difficulty changed
                        new_difficulty = st.session_state.simulation.performance.current_difficulty
                        difficulty_changed = prev_difficulty != new_difficulty

                        # Show immediate results
                        st.success("✅ Decision evaluated!")
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            biz_score = eval_result.get('business_evaluation', {}).get('score', 0)
                            st.metric("🎯 Business", f"{biz_score}/100")
                        with c2:
                            mod_score = eval_result.get('module_evaluation', {}).get('score', 0)
                            st.metric("📚 Module", f"{mod_score}/100")
                        with c3:
                            overall = eval_result.get('overall_score', 0)
                            st.metric("⭐ Overall", f"{overall}/100")
                        
                        st.info(f"**📝 Feedback:** {eval_result.get('feedback', '')}")

                        if eval_result.get('better_approach'):
                            st.success(f"**💡 Better Approach:** {eval_result['better_approach']}")

                        # Show difficulty change notification
                        if difficulty_changed:
                            new_diff_info = st.session_state.simulation.get_difficulty_descriptor()
                            if new_difficulty.value > prev_difficulty.value:
                                st.success(f"🎉 **Difficulty Increased!** You've advanced to {new_diff_info['emoji']} **{new_diff_info['name']}** level. Next scenarios will be more challenging!")
                            else:
                                st.info(f"ℹ️ **Difficulty Adjusted:** Scenarios adapted to {new_diff_info['emoji']} **{new_diff_info['name']}** level to support your learning.")
                        
                        # Check if final round
                        if st.session_state.round >= MAX_ROUNDS:
                            st.session_state.simulation_complete = True
                            st.session_state.current_scenario = None
                            st.session_state.final_summary = None
                        else:
                            # Generate next scenario
                            context = f"Previous decision: {selected}. Business score: {biz_score}/100, Module score: {mod_score}/100"
                            st.session_state.current_scenario = st.session_state.simulation.generate_scenario(
                                context, new_metrics, is_first=False, round_num=st.session_state.round + 1
                            )
                            st.session_state.round += 1
                        
                        # Clear opinions and chat
                        st.session_state.board_opinions = None
                        st.session_state.chat_history = {}
                        st.session_state.selected_chat_members = []
                    
                    time.sleep(3)
                    st.rerun()
            
            # Selective consultation
            st.divider()
            st.subheader("💬 Consult Specific Board Members")
            
            members = [m.name for m in st.session_state.simulation.board_members]
            
            selected_members = st.multiselect(
                "Select members to consult:",
                members,
                key="selected_opinion_members"
            )
            
            if selected_members:
                if st.button("📊 Get Selected Members' Input", use_container_width=True):
                    with st.spinner(f"Consulting {len(selected_members)} member(s)..."):
                        opts = scenario_data['options']
                        if opts and isinstance(opts[0], dict):
                            opts = [str(o.get('text', o.get('value', o))) for o in opts]
                        else:
                            opts = [str(o) for o in opts]

                        opinions = st.session_state.simulation.generate_board_opinions(
                            scenario_data['scenario'],
                            scenario_data['question'],
                            opts,
                            st.session_state.simulation.metrics,
                            selected_members=selected_members
                        )

                        if st.session_state.board_opinions is None:
                            st.session_state.board_opinions = {}
                        st.session_state.board_opinions.update(opinions)
                        # Track consultation
                        st.session_state.simulation.performance.consultation_count += 1
                    st.rerun()
            
            # Chat feature
            st.divider()
            st.subheader("💭 Chat with Board Members")
            
            chat_member = st.selectbox(
                "Select a board member to chat with:",
                [""] + members,
                key="selected_chat_member"
            )
            
            if chat_member:
                if chat_member not in st.session_state.chat_history:
                    st.session_state.chat_history[chat_member] = []
                
                # Display chat history
                if st.session_state.chat_history[chat_member]:
                    st.write(f"**Conversation with {chat_member}:**")
                    for msg in st.session_state.chat_history[chat_member]:
                        with st.chat_message("user"):
                            st.write(msg['question'])
                        with st.chat_message("assistant"):
                            st.write(msg['answer'])
                
                # Chat input
                question = st.text_input(
                    f"Ask {chat_member}:",
                    key=f"chat_input_{chat_member}",
                    placeholder="e.g., Which framework should we apply here?"
                )
                
                if st.button("Send Question", key=f"send_{chat_member}"):
                    if question.strip():
                        with st.spinner(f"Getting response from {chat_member}..."):
                            opts = scenario_data['options']
                            if opts and isinstance(opts[0], dict):
                                opts = [str(o.get('text', o.get('value', o))) for o in opts]
                            else:
                                opts = [str(o) for o in opts]
                            
                            answer = st.session_state.simulation.chat_with_board_member(
                                chat_member,
                                question,
                                scenario_data['scenario'],
                                scenario_data['question'],
                                opts,
                                st.session_state.simulation.metrics,
                                st.session_state.chat_history[chat_member]
                            )
                            
                            st.session_state.chat_history[chat_member].append({
                                'question': question,
                                'answer': answer
                            })
                        st.rerun()
                    else:
                        st.warning("Please enter a question first.")
    
    # Board members column
    with col2:
        st.subheader("👔 Board Members")
        
        if st.session_state.board_opinions:
            for member in st.session_state.simulation.board_members:
                if member.name in st.session_state.board_opinions:
                    with st.expander(f"**{member.name}**", expanded=True):
                        st.caption(f"*{member.role}*")
                        st.write(st.session_state.board_opinions[member.name])
                else:
                    st.write(f"**{member.name}**")
                    st.caption(f"*{member.role}*")
        else:
            st.info("Use the buttons below to consult board members for their expert input.")
            for member in st.session_state.simulation.board_members:
                st.write(f"**{member.name}**")
                st.caption(f"*{member.role}*")
                st.caption(f"_{member.personality[:80]}..._")
    
    # Recent decisions
    if st.session_state.history:
        st.divider()
        st.subheader("📜 Recent Decisions")
        
        for e in reversed(st.session_state.history[-3:]):
            with st.expander(f"Round {e['round']}: {e['decision'][:50]}... (Score: {e.get('overall_score', 0)}/100)"):
                st.write("**Scenario:**", e['scenario'])
                st.write("**Module Topic:**", e.get('module_connection', 'N/A'))
                st.write("**Your Decision:**", e['decision'])
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Business", f"{e.get('business_score', 0)}/100")
                with c2:
                    st.metric("Module", f"{e.get('module_score', 0)}/100")
                with c3:
                    st.metric("Overall", f"{e.get('overall_score', 0)}/100")
                
                if 'feedback' in e and e['feedback']:
                    st.caption(f"💬 {e['feedback'][:120]}...")

# ==================== HELP SECTION ====================

if st.session_state.initialized and not st.session_state.get('simulation_complete', False):
    st.divider()
    with st.expander("ℹ️ How This Simulation Works"):
        st.markdown("""
        ### 📚 Dual Assessment System
        
        Every decision is evaluated on **TWO dimensions**:
        
        **🎯 Business Performance (0-100)**
        - Evaluates practical decision impact
        - Tracks metric changes (revenue, profit, satisfaction, etc.)
        - Assesses strategic coherence
        - Measures leadership effectiveness
        
        **📖 Module Knowledge (0-100)**
        - Tests understanding of course concepts
        - Evaluates framework application
        - Checks theoretical soundness
        - Identifies knowledge gaps
        
        ### 💡 Tips for Success
        
        1. **Read scenarios carefully** - They contain clues about which concepts apply
        2. **Consult board members** - They provide insights on both business and theory
        3. **Use the chat feature** - Ask specific questions about frameworks
        4. **Balance both dimensions** - Best decisions consider theory AND practice
        5. **Review feedback** - Learn from each round to improve
        
        ### 📊 Scoring Guide
        
        - **90-100**: Excellent - Strong grasp of both business and theory
        - **80-89**: Good - Solid understanding with minor gaps
        - **70-79**: Satisfactory - Acceptable but room for improvement
        - **60-69**: Needs Work - Significant gaps
        - **Below 60**: Poor - Major misunderstandings
        
        ### 🏆 What Makes a Great Decision?
        
        ✅ Applies correct module framework for the situation  
        ✅ Considers business metrics and practical constraints  
        ✅ Balances short-term and long-term impacts  
        ✅ Uses appropriate terminology from the course  
        ✅ Shows strategic thinking aligned with company goals
        """)

# ==================== END OF CODE ====================