"""
Board Room Simulation Application
A business and module simulation using Streamlit, LangChain, and Gemini 2.5 Flash Lite
"""

import streamlit as st
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Google Generative AI direct import
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Time pressure settings (in minutes)
TIME_PRESSURE_MINUTES = {
    "relaxed": 15,
    "normal": 10,
    "urgent": 5
}

# Page configuration
st.set_page_config(
    page_title="Board Room Simulation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .board-member-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .selected-role-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .scenario-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
    .decision-box {
        background: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dc3545;
    }
    .info-box {
        background: #cce5ff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #007bff;
    }
    .round-indicator {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E3A5F;
        padding: 0.5rem 1rem;
        background: #e9ecef;
        border-radius: 20px;
        display: inline-block;
    }
    .consultation-counter {
        font-size: 1rem;
        padding: 0.5rem 1rem;
        background: #e7f3ff;
        border-radius: 10px;
        border: 1px solid #007bff;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .option-button {
        width: 100%;
        margin: 0.5rem 0;
    }
    .committee-card {
        background: #f0f7ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .timer-container {
        text-align: center;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
    }
    .timer-relaxed {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .timer-normal {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border: 2px solid #ffc107;
    }
    .timer-urgent {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    .timer-expired {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border: 2px solid #bd2130;
        color: white;
    }
    .timer-display {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .timer-label {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .stance-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #666;
    }
    .stance-approve {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .stance-oppose {
        background: #f8d7da;
        border-left-color: #dc3545;
    }
    .stance-neutral {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
    .stance-convinced {
        background: #d1ecf1;
        border-left-color: #17a2b8;
    }
    .deliberation-header {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f0ff 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .debate-box {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .conviction-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }
    .conviction-fill {
        height: 100%;
        background: linear-gradient(90deg, #ffc107 0%, #dc3545 100%);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


class SimulationPhase(Enum):
    SETUP = "setup"
    ROLE_SELECT = "role_select"
    BRIEFING = "briefing"
    DISCUSSION = "discussion"
    DECISION = "decision"
    FEEDBACK = "feedback"
    SUMMARY = "summary"


@dataclass
class SimulationState:
    """Tracks the current state of the simulation"""
    current_round: int = 0
    current_phase: SimulationPhase = SimulationPhase.SETUP
    total_rounds: int = 5
    score: int = 0
    decisions_made: List[Dict] = None
    conversation_history: List[Dict] = None
    consultations_used: int = 0
    max_consultations: int = 2

    def __post_init__(self):
        if self.decisions_made is None:
            self.decisions_made = []
        if self.conversation_history is None:
            self.conversation_history = []


def load_simulation_data(file_path: str) -> Optional[Dict]:
    """Load simulation data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading simulation data: {e}")
        return None


def initialize_llm(api_key: str) -> genai.GenerativeModel:
    """Initialize Gemini 2.0 Flash Lite model"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 2048,
        }
    )


def get_board_member_prompt(member: Dict, company_data: Dict, module_data: Dict) -> str:
    """Generate a system prompt for a board member persona"""
    return f"""You are {member['name']}, {member['role']} at {company_data['company_name']}.

PERSONALITY & BACKGROUND:
{member['personality']}

EXPERTISE: {member['expertise']}
TENURE: {member['tenure_years']} years on the board

COMPANY CONTEXT:
{company_data['company_overview']}

CURRENT CHALLENGES:
{chr(10).join(f"- {problem}" for problem in company_data['current_problems'])}

YOUR ROLE IN THIS SIMULATION:
- Stay in character as {member['name']} throughout the discussion
- Express opinions based on your expertise and personality
- Challenge or support decisions based on your perspective
- Reference relevant metrics and data when appropriate
- Consider the module topic: {module_data['module_name']}

KEY METRICS TO CONSIDER:
- Annual Revenue: {company_data['metrics']['total_revenue_annual']['value']} {company_data['metrics']['total_revenue_annual']['unit']}
- EBITDA: {company_data['metrics']['ebitda']['value']} {company_data['metrics']['ebitda']['unit']}
- Employee Count: {company_data['metrics']['employee_count']['value']}
- Platform Uptime: {company_data['metrics']['platform_uptime']['value']}%

Respond naturally as this board member would, considering their biases and priorities."""


def get_committee_prompt(committee: Dict, company_data: Dict, module_data: Dict, members_data: List[Dict]) -> str:
    """Generate a system prompt for a committee consultation"""
    member_details = []
    for member_name in committee['members']:
        member = next((m for m in members_data if m['name'] == member_name), None)
        if member:
            member_details.append(f"- {member['name']} ({member['role']}): {member['expertise']} expertise")

    return f"""You are representing the {committee['name']} of {company_data['company_name']}.

COMMITTEE DETAILS:
- Type: {committee['type']}
- Purpose: {committee['purpose']}
- Chairperson: {committee['chairperson']}
- Members:
{chr(10).join(member_details)}

COMPANY CONTEXT:
{company_data['company_overview']}

CURRENT CHALLENGES:
{chr(10).join(f"- {problem}" for problem in company_data['current_problems'])}

MODULE FOCUS: {module_data['module_name']}

As the {committee['name']}, provide a collective perspective that:
- Reflects the committee's specialized focus ({committee['purpose']})
- Incorporates viewpoints from all committee members
- Provides actionable recommendations
- References relevant governance frameworks and best practices

Respond as a unified committee voice, acknowledging different member perspectives where relevant."""


def get_member_stance_prompt(member: Dict, company_data: Dict, module_data: Dict,
                              scenario: str, player_decision: str, player_role: Dict) -> str:
    """Generate prompt for a board member to evaluate the player's decision"""
    return f"""You are {member['name']}, {member['role']} at {company_data['company_name']}.

PERSONALITY & BACKGROUND:
{member['personality']}

EXPERTISE: {member['expertise']}
TENURE: {member['tenure_years']} years on the board

YOUR TASK: Evaluate a fellow board member's decision and determine your stance.

SCENARIO PRESENTED:
{scenario}

DECISION MADE BY {player_role['name']} ({player_role['role']}):
{player_decision}

Based on your expertise in {member['expertise']} and your personality:
1. Would you APPROVE, OPPOSE, or remain NEUTRAL on this decision?
2. How does this decision relate to your area of expertise?
3. What is your honest reaction?

Respond in this EXACT format:
STANCE: [APPROVE/OPPOSE/NEUTRAL]
CONVICTION_LEVEL: [1-10, where 10 is absolute certainty]
EXPERTISE_RELEVANCE: [Brief explanation of how your expertise relates to this decision]
REACTION: [Your 2-3 sentence reaction as this board member, in first person]
COUNTER_OPINION: [If OPPOSE: Your specific objection and what you believe should be done instead. If APPROVE/NEUTRAL: Write "N/A"]

Stay in character. Consider how {member['name']}'s biases and priorities would influence their view."""


def get_debate_evaluation_prompt(member: Dict, company_data: Dict,
                                  original_counter_opinion: str,
                                  player_response: str,
                                  debate_history: List[Dict],
                                  player_role: Dict) -> str:
    """Generate prompt to evaluate player's response to a dissenter's argument"""

    history_text = ""
    for exchange in debate_history:
        history_text += f"\n{member['name']}'s argument: {exchange.get('dissenter_argument', '')}"
        history_text += f"\n{player_role['name']}'s response: {exchange.get('player_response', '')}\n"

    return f"""You are {member['name']}, {member['role']} at {company_data['company_name']}.

PERSONALITY & BACKGROUND:
{member['personality']}

EXPERTISE: {member['expertise']}

You OPPOSED a decision and raised this counter-opinion:
"{original_counter_opinion}"

DEBATE HISTORY SO FAR:
{history_text if history_text else "This is the first exchange."}

{player_role['name']} ({player_role['role']}) NOW RESPONDS:
"{player_response}"

Evaluate this response considering:
1. Does it address your specific concerns?
2. Is the reasoning sound and well-supported?
3. Does it account for your area of expertise ({member['expertise']})?
4. Would {member['name']} be convinced by this argument?

Respond in this EXACT format:
EVALUATION: [2-3 sentences assessing the response quality]
RESPONSE_SCORE: [0-100, how effective the response was]
STANCE_CHANGED: [YES/NO - has this response convinced you?]
FOLLOW_UP: [If STANCE_CHANGED is NO: Your follow-up challenge or continued objection. If YES: Your acknowledgment of their point and why you now support the decision]

Remember to stay in character as {member['name']}."""


def get_consultation_alignment_prompt(consultations: List[Dict],
                                       player_decision: str,
                                       member_stances: Dict) -> str:
    """Generate prompt to evaluate how well player's consultations aligned with their decision"""

    consultation_text = "\n".join([
        f"- Consulted {c.get('member', 'Unknown')}: Asked about '{c.get('content', '')[:100]}...'"
        for c in consultations if c.get('role') == 'user'
    ])

    approving_members = [name for name, stance in member_stances.items()
                         if stance.get('stance') in ['APPROVE', 'approve']]
    opposing_members = [name for name, stance in member_stances.items()
                        if stance.get('stance') in ['OPPOSE', 'oppose']]

    return f"""Analyze the alignment between a player's board consultations and their final decision.

CONSULTATIONS MADE:
{consultation_text if consultation_text else "No consultations were made."}

FINAL DECISION:
{player_decision}

BOARD MEMBER REACTIONS:
- Approving: {', '.join(approving_members) if approving_members else 'None'}
- Opposing: {', '.join(opposing_members) if opposing_members else 'None'}

Evaluate:
1. Did the player consult members whose expertise was relevant to their decision?
2. Did the consultations help anticipate board opposition?
3. Was the consultation strategy effective?

Respond in this EXACT format:
ALIGNMENT_SCORE: [0-100]
REASONING: [2-3 sentences explaining the score]"""


def get_scenario_generator_prompt(company_data: Dict, module_data: Dict, round_config: Dict, player_role: Dict) -> str:
    """Generate prompt for creating simulation scenarios"""
    return f"""You are a corporate governance simulation scenario generator.

COMPANY: {company_data['company_name']}
{company_data['company_overview']}

CURRENT PROBLEMS:
{chr(10).join(f"- {problem}" for problem in company_data['current_problems'])}

MODULE FOCUS: {module_data['module_name']}
{module_data['overview']}

LEARNING OBJECTIVES:
{chr(10).join(f"- {obj}" for obj in module_data['learning_objectives'])}

PLAYER'S ROLE: {player_role['name']} - {player_role['role']}
Player's Expertise: {player_role['expertise']}

ROUND CONFIGURATION:
- Round Number: {round_config['round_number']}
- Difficulty: {round_config['difficulty']}
- Focus Area: {round_config.get('focus_area', 'General')}
- Round Type: {round_config['round_type']}

Generate a realistic boardroom scenario that:
1. Presents a specific challenge or decision point relevant to the player's role
2. Requires application of concepts from the module
3. Has multiple valid approaches with trade-offs
4. Creates tension between different board member perspectives
5. Is appropriate for the specified difficulty level

Format your response as:
SCENARIO TITLE: [Title]

SITUATION: [Detailed description of the situation - 2-3 paragraphs]

KEY QUESTION: [The main decision or question the player must address]

STAKEHOLDERS AFFECTED: [List of affected parties]

TIME SENSITIVITY: [How urgent is this decision]

OPTIONS TO CONSIDER:
A) [First option - brief description]
B) [Second option - brief description]
C) [Third option - brief description]
D) [Fourth option - brief description]

Make sure each option is distinct and represents a different strategic approach."""


def generate_scenario(llm: genai.GenerativeModel, company_data: Dict,
                      module_data: Dict, round_config: Dict, player_role: Dict) -> str:
    """Generate a new scenario for the current round"""
    prompt = get_scenario_generator_prompt(company_data, module_data, round_config, player_role)

    full_prompt = f"""You are an expert corporate governance simulation designer.

{prompt}"""

    response = llm.generate_content(full_prompt)
    return response.text


def get_board_member_response(llm: genai.GenerativeModel, members: List[Dict],
                               company_data: Dict, module_data: Dict,
                               scenario: str, user_input: str,
                               conversation_history: List[Dict],
                               player_role: Dict) -> str:
    """Get a response from one or multiple board member personas"""

    if len(members) == 1:
        # Single member response
        member = members[0]
        system_prompt = get_board_member_prompt(member, company_data, module_data)

        history_text = ""
        for entry in conversation_history[-10:]:
            if entry['role'] == 'user':
                history_text += f"{player_role['name']}: {entry['content']}\n"
            else:
                history_text += f"{entry.get('member', 'Board Member')}: {entry['content']}\n"

        full_prompt = f"""{system_prompt}

CONVERSATION HISTORY:
{history_text}

CURRENT SCENARIO:
{scenario}

{player_role['name']} ({player_role['role']}) ASKS:
{user_input}

Respond as {member['name']} would, considering your expertise in {member['expertise']} and your personality traits.
Be concise but insightful. Express your genuine opinion based on your character.
Address {player_role['name']} directly in your response."""

    else:
        # Multiple members - group discussion
        member_names = [m['name'] for m in members]
        member_details = "\n".join([f"- {m['name']} ({m['role']}): {m['personality'][:150]}..." for m in members])

        history_text = ""
        for entry in conversation_history[-10:]:
            if entry['role'] == 'user':
                history_text += f"{player_role['name']}: {entry['content']}\n"
            else:
                history_text += f"{entry.get('member', 'Board Members')}: {entry['content']}\n"

        full_prompt = f"""You are simulating a group discussion between the following board members at {company_data['company_name']}:

{member_details}

COMPANY CONTEXT:
{company_data['company_overview']}

CURRENT CHALLENGES:
{chr(10).join(f"- {problem}" for problem in company_data['current_problems'])}

CONVERSATION HISTORY:
{history_text}

CURRENT SCENARIO:
{scenario}

{player_role['name']} ({player_role['role']}) ASKS THE GROUP:
{user_input}

Provide a group discussion response where each board member ({', '.join(member_names)}) briefly shares their perspective.
Format as:
**[Member Name]:** [Their response]

Each member should respond according to their expertise and personality. Keep each response concise (2-3 sentences).
Address {player_role['name']} directly."""

    response = llm.generate_content(full_prompt)
    return response.text


def get_committee_response(llm: genai.GenerativeModel, committee: Dict,
                           company_data: Dict, module_data: Dict,
                           scenario: str, user_input: str,
                           conversation_history: List[Dict],
                           player_role: Dict,
                           all_members: List[Dict]) -> str:
    """Get a response from a committee"""
    system_prompt = get_committee_prompt(committee, company_data, module_data, all_members)

    history_text = ""
    for entry in conversation_history[-10:]:
        if entry['role'] == 'user':
            history_text += f"{player_role['name']}: {entry['content']}\n"
        else:
            history_text += f"{entry.get('member', 'Committee')}: {entry['content']}\n"

    full_prompt = f"""{system_prompt}

CONVERSATION HISTORY:
{history_text}

CURRENT SCENARIO:
{scenario}

{player_role['name']} ({player_role['role']}) CONSULTS THE {committee['name'].upper()}:
{user_input}

Provide the committee's collective response, incorporating perspectives from:
- Chairperson {committee['chairperson']}'s leadership view
- Key insights from committee members

Be concise but comprehensive. Offer actionable recommendations aligned with the committee's purpose.
Address {player_role['name']} directly."""

    response = llm.generate_content(full_prompt)
    return response.text


def calculate_metric_impacts(llm: genai.GenerativeModel, company_data: Dict,
                              scenario: str, decision: str, score: int) -> Dict:
    """Calculate the impact of a decision on company metrics"""

    metrics = company_data['metrics']

    # Build metrics context
    metrics_context = "\n".join([
        f"- {metrics[key]['description']}: {metrics[key]['value']} {metrics[key]['unit']} (Priority: {metrics[key].get('priority', 'Normal')})"
        for key in metrics.keys()
    ])

    impact_prompt = f"""You are a business analyst evaluating the impact of a board decision on company metrics.

COMPANY: {company_data['company_name']}

CURRENT METRICS:
{metrics_context}

SCENARIO:
{scenario}

DECISION MADE:
{decision}

DECISION QUALITY SCORE: {score}/100

Based on this decision, analyze the realistic impact on company metrics. Consider:
1. Direct impacts from the decision
2. Indirect/ripple effects
3. Short-term vs long-term implications
4. The quality of the decision (score: {score})

Provide metric impacts in this EXACT format (use these exact metric keys):
METRIC_IMPACTS:
- total_revenue_annual: [change as number, e.g., +5 or -3 or 0] | [brief reason]
- ebitda: [change as number] | [brief reason]
- net_profit_margin: [change as decimal, e.g., +0.5 or -0.2] | [brief reason]
- platform_uptime: [change as decimal] | [brief reason]
- net_promoter_score: [change as number] | [brief reason]
- customer_churn_rate_annual: [change as decimal] | [brief reason]
- employee_engagement_score: [change as number] | [brief reason]
- annual_attrition_rate: [change as decimal] | [brief reason]
- regulatory_compliance_score: [change as number] | [brief reason]
- open_high_severity_risks: [change as number] | [brief reason]
- deployment_frequency: [change as number] | [brief reason]
- revenue_growth_yoy: [change as decimal] | [brief reason]

IMPACT_SUMMARY: [2-3 sentence summary of overall business impact]

Be realistic - not every decision affects all metrics. Use 0 for unaffected metrics.
Good decisions (score > 70) should generally have positive impacts.
Poor decisions (score < 50) should have negative impacts."""

    response = llm.generate_content(impact_prompt)
    content = response.text

    # Parse metric impacts
    impacts = {}
    reasons = {}

    if "METRIC_IMPACTS:" in content:
        try:
            impacts_section = content.split("METRIC_IMPACTS:")[1]
            if "IMPACT_SUMMARY:" in impacts_section:
                impacts_section = impacts_section.split("IMPACT_SUMMARY:")[0]

            for line in impacts_section.strip().split("\n"):
                line = line.strip()
                if line.startswith("-") and ":" in line and "|" in line:
                    # Parse: - metric_key: value | reason
                    parts = line[1:].strip().split(":", 1)
                    if len(parts) == 2:
                        metric_key = parts[0].strip()
                        value_reason = parts[1].strip().split("|")
                        if len(value_reason) >= 2:
                            try:
                                change_str = value_reason[0].strip().replace("+", "")
                                change = float(change_str)
                                reason = value_reason[1].strip()
                                impacts[metric_key] = change
                                reasons[metric_key] = reason
                            except ValueError:
                                pass
        except Exception:
            pass

    # Extract impact summary
    impact_summary = ""
    if "IMPACT_SUMMARY:" in content:
        try:
            impact_summary = content.split("IMPACT_SUMMARY:")[1].strip().split("\n")[0]
        except:
            pass

    return {
        "impacts": impacts,
        "reasons": reasons,
        "summary": impact_summary
    }


def apply_metric_impacts(metrics: Dict, impacts: Dict) -> Dict:
    """Apply calculated impacts to metrics and return updated metrics"""
    updated_metrics = {}

    for key, metric in metrics.items():
        updated_metrics[key] = metric.copy()
        if key in impacts:
            change = impacts[key]
            old_value = metric['value']
            new_value = old_value + change

            # Apply bounds based on metric type
            if metric['unit'] == '%':
                new_value = max(0, min(100, new_value))
            elif metric['unit'] == 'count':
                new_value = max(0, int(new_value))
            elif 'Cr' in metric['unit'] or 'L' in metric['unit']:
                new_value = max(0, round(new_value, 1))

            updated_metrics[key]['value'] = new_value
            updated_metrics[key]['previous_value'] = old_value
            updated_metrics[key]['change'] = change

    return updated_metrics


def evaluate_decision(llm: genai.GenerativeModel, company_data: Dict,
                      module_data: Dict, scenario: str,
                      decision: str, round_config: Dict,
                      player_role: Dict) -> Dict:
    """Evaluate user's decision and provide feedback"""

    evaluation_prompt = f"""You are a fair and constructive corporate governance educator and evaluator.

COMPANY CONTEXT:
{company_data['company_name']}
{company_data['company_overview']}

PLAYER'S ROLE: {player_role['name']} - {player_role['role']}
Player's Expertise: {player_role['expertise']}

MODULE: {module_data['module_name']}
Learning Objectives:
{chr(10).join(f"- {obj}" for obj in module_data['learning_objectives'])}

RELEVANT TOPICS:
{chr(10).join(f"- {topic['name']}: {topic['description']}" for topic in module_data['topics'][:5])}

SCENARIO PRESENTED:
{scenario}

PLAYER'S DECISION:
{decision}

DIFFICULTY: {round_config['difficulty']}

Evaluate the decision based on:
1. Understanding of board governance principles
2. Application of relevant legal frameworks (Companies Act, 2013)
3. Consideration of stakeholder interests
4. Strategic thinking and risk assessment
5. Alignment with learning objectives
6. Appropriateness for the player's role as {player_role['role']}

Provide your evaluation in this EXACT format:
SCORE: [0-100]

SCORE_REASONING: [Explain specifically why you gave this score. Break down the scoring:
- Governance Understanding (0-25 pts): [points earned] - [brief explanation]
- Legal/Regulatory Compliance (0-20 pts): [points earned] - [brief explanation]
- Stakeholder Consideration (0-20 pts): [points earned] - [brief explanation]
- Strategic Thinking (0-20 pts): [points earned] - [brief explanation]
- Role Alignment (0-15 pts): [points earned] - [brief explanation]]

STRENGTHS: [What was done well - be specific with 2-3 bullet points]

AREAS_FOR_IMPROVEMENT: [What could be better - be specific with 2-3 bullet points]

KEY_LEARNING_POINTS: [Relevant concepts from the module that apply - 2-3 points]

BEST_APPROACH: [Describe in detail what the ideal/optimal decision would have been for this scenario. Include:
- The recommended action
- Key considerations that should have been addressed
- How it aligns with corporate governance best practices
- Expected positive outcomes of the ideal approach]

ENCOURAGEMENT: [Motivational feedback for the learner]"""

    response = llm.generate_content(evaluation_prompt)

    # Parse the response
    content = response.text

    # Extract score
    score = 70  # Default score
    if "SCORE:" in content:
        try:
            score_line = content.split("SCORE:")[1].split("\n")[0]
            score = int(''.join(filter(str.isdigit, score_line[:10])))
        except:
            pass

    score = min(100, max(0, score))

    # Extract score reasoning
    score_reasoning = ""
    if "SCORE_REASONING:" in content:
        try:
            reasoning_section = content.split("SCORE_REASONING:")[1]
            # Find the next section marker
            for marker in ["STRENGTHS:", "AREAS_FOR_IMPROVEMENT:", "KEY_LEARNING_POINTS:"]:
                if marker in reasoning_section:
                    reasoning_section = reasoning_section.split(marker)[0]
                    break
            score_reasoning = reasoning_section.strip()
        except:
            pass

    # Extract strengths
    strengths = ""
    if "STRENGTHS:" in content:
        try:
            strengths_section = content.split("STRENGTHS:")[1]
            for marker in ["AREAS_FOR_IMPROVEMENT:", "KEY_LEARNING_POINTS:", "BEST_APPROACH:"]:
                if marker in strengths_section:
                    strengths_section = strengths_section.split(marker)[0]
                    break
            strengths = strengths_section.strip()
        except:
            pass

    # Extract areas for improvement
    improvements = ""
    if "AREAS_FOR_IMPROVEMENT:" in content:
        try:
            improvements_section = content.split("AREAS_FOR_IMPROVEMENT:")[1]
            for marker in ["KEY_LEARNING_POINTS:", "BEST_APPROACH:", "ENCOURAGEMENT:"]:
                if marker in improvements_section:
                    improvements_section = improvements_section.split(marker)[0]
                    break
            improvements = improvements_section.strip()
        except:
            pass

    # Extract key learning points
    learning_points = ""
    if "KEY_LEARNING_POINTS:" in content:
        try:
            learning_section = content.split("KEY_LEARNING_POINTS:")[1]
            for marker in ["BEST_APPROACH:", "ENCOURAGEMENT:", "RECOMMENDED_APPROACH:"]:
                if marker in learning_section:
                    learning_section = learning_section.split(marker)[0]
                    break
            learning_points = learning_section.strip()
        except:
            pass

    # Extract best approach (for round summary)
    best_approach = ""
    if "BEST_APPROACH:" in content:
        try:
            best_section = content.split("BEST_APPROACH:")[1]
            if "ENCOURAGEMENT:" in best_section:
                best_section = best_section.split("ENCOURAGEMENT:")[0]
            best_approach = best_section.strip()
        except:
            pass

    # Extract encouragement
    encouragement = ""
    if "ENCOURAGEMENT:" in content:
        try:
            encouragement = content.split("ENCOURAGEMENT:")[1].strip()
        except:
            pass

    # Calculate metric impacts
    metric_impacts = calculate_metric_impacts(llm, company_data, scenario, decision, score)

    return {
        "score": score,
        "feedback": content,
        "score_reasoning": score_reasoning,
        "strengths": strengths,
        "improvements": improvements,
        "learning_points": learning_points,
        "best_approach": best_approach,
        "encouragement": encouragement,
        "decision": decision,
        "scenario": scenario,
        "metric_impacts": metric_impacts
    }


def generate_member_stances(llm: genai.GenerativeModel, company_data: Dict,
                             module_data: Dict, scenario: str,
                             player_decision: str, player_role: Dict) -> Dict[str, Dict]:
    """Generate each board member's stance on the player's decision"""
    logger.debug(f"generate_member_stances called with {len(company_data.get('board_members', []))} board members")

    stances = {}
    available_members = [m for m in company_data['board_members']
                         if m['name'] != player_role['name']]

    logger.debug(f"Processing {len(available_members)} available members (excluding player)")

    for member in available_members:
        logger.debug(f"Generating stance for {member['name']}")
        prompt = get_member_stance_prompt(member, company_data, module_data,
                                          scenario, player_decision, player_role)

        try:
            response = llm.generate_content(prompt)
            content = response.text
            logger.debug(f"Got response for {member['name']}, length: {len(content)}")
        except Exception as e:
            logger.error(f"Error getting stance for {member['name']}: {e}")
            content = "STANCE: NEUTRAL\nCONVICTION_LEVEL: 5\nREACTION: Unable to evaluate.\nCOUNTER_OPINION: N/A"

        # Parse response
        stance = "NEUTRAL"
        conviction = 5
        relevance = ""
        reaction = ""
        counter_opinion = None

        if "STANCE:" in content:
            stance_line = content.split("STANCE:")[1].split("\n")[0].strip().upper()
            if "APPROVE" in stance_line:
                stance = "APPROVE"
            elif "OPPOSE" in stance_line:
                stance = "OPPOSE"
            else:
                stance = "NEUTRAL"

        if "CONVICTION_LEVEL:" in content:
            try:
                conv_str = content.split("CONVICTION_LEVEL:")[1].split("\n")[0].strip()
                conviction = int(''.join(filter(str.isdigit, conv_str[:3])))
                conviction = max(1, min(10, conviction))
            except:
                conviction = 5

        if "EXPERTISE_RELEVANCE:" in content:
            try:
                relevance = content.split("EXPERTISE_RELEVANCE:")[1].split("REACTION:")[0].strip()
            except:
                pass

        if "REACTION:" in content:
            try:
                reaction = content.split("REACTION:")[1].split("COUNTER_OPINION:")[0].strip()
            except:
                pass

        if "COUNTER_OPINION:" in content and stance == "OPPOSE":
            try:
                counter_opinion = content.split("COUNTER_OPINION:")[1].strip()
                if counter_opinion.upper().startswith("N/A"):
                    counter_opinion = None
            except:
                pass

        stances[member['name']] = {
            'member_name': member['name'],
            'member_role': member['role'],
            'member_expertise': member['expertise'],
            'stance': stance,
            'initial_reaction': reaction,
            'counter_opinion': counter_opinion,
            'expertise_relevance': relevance,
            'conviction_level': conviction,
            'convinced_in_round': None,
            'debate_exchanges': 0
        }
        logger.debug(f"Member {member['name']} stance: {stance}, conviction: {conviction}")

    logger.debug(f"Generated stances for {len(stances)} members")
    return stances


def evaluate_debate_response(llm: genai.GenerativeModel, member: Dict,
                              company_data: Dict, original_counter: str,
                              player_response: str, debate_history: List[Dict],
                              player_role: Dict) -> Dict:
    """Evaluate player's response to a dissenter and determine if stance changes"""

    prompt = get_debate_evaluation_prompt(member, company_data, original_counter,
                                           player_response, debate_history, player_role)

    response = llm.generate_content(prompt)
    content = response.text

    evaluation = ""
    score = 50
    stance_changed = False
    follow_up = ""

    if "EVALUATION:" in content:
        try:
            evaluation = content.split("EVALUATION:")[1].split("RESPONSE_SCORE:")[0].strip()
        except:
            pass

    if "RESPONSE_SCORE:" in content:
        try:
            score_str = content.split("RESPONSE_SCORE:")[1].split("\n")[0].strip()
            score = int(''.join(filter(str.isdigit, score_str[:3])))
            score = max(0, min(100, score))
        except:
            score = 50

    if "STANCE_CHANGED:" in content:
        stance_line = content.split("STANCE_CHANGED:")[1].split("\n")[0].strip().upper()
        stance_changed = "YES" in stance_line

    if "FOLLOW_UP:" in content:
        try:
            follow_up = content.split("FOLLOW_UP:")[1].strip()
        except:
            pass

    return {
        'evaluation': evaluation,
        'score': score,
        'stance_changed': stance_changed,
        'follow_up': follow_up
    }


def evaluate_consultation_alignment(llm: genai.GenerativeModel, consultations: List[Dict],
                                     player_decision: str, member_stances: Dict) -> Dict:
    """Evaluate how well player's consultations aligned with their decision"""

    prompt = get_consultation_alignment_prompt(consultations, player_decision, member_stances)

    response = llm.generate_content(prompt)
    content = response.text

    alignment_score = 50
    reasoning = ""

    if "ALIGNMENT_SCORE:" in content:
        try:
            score_str = content.split("ALIGNMENT_SCORE:")[1].split("\n")[0].strip()
            alignment_score = int(''.join(filter(str.isdigit, score_str[:3])))
            alignment_score = max(0, min(100, alignment_score))
        except:
            alignment_score = 50

    if "REASONING:" in content:
        try:
            reasoning = content.split("REASONING:")[1].strip()
        except:
            pass

    return {
        'alignment_score': alignment_score,
        'reasoning': reasoning
    }


def calculate_board_effectiveness_score(round_number: int,
                                          member_stances: Dict,
                                          debate_history: List[Dict],
                                          consultation_alignment: float,
                                          force_submitted: bool,
                                          max_debate_rounds: int = 3) -> Dict:
    """Calculate the board effectiveness score for a round"""

    total_members = len(member_stances)
    initially_approving = sum(1 for s in member_stances.values()
                              if s.get('stance') == 'APPROVE')
    initially_opposing = sum(1 for s in member_stances.values()
                             if s.get('stance') == 'OPPOSE')
    convinced = sum(1 for s in member_stances.values()
                    if s.get('convinced_in_round') is not None)

    # Count total debate exchanges
    total_debate_exchanges = sum(s.get('debate_exchanges', 0) for s in member_stances.values())

    # Score components (total 100 points)

    # 1. Initial approval rate (25 points max)
    initial_approval_score = (initially_approving / max(total_members, 1)) * 25

    # 2. Consultation alignment (25 points max)
    consultation_score = (consultation_alignment / 100) * 25

    # 3. Debate effectiveness (30 points max)
    if initially_opposing > 0:
        debate_effectiveness = (convinced / initially_opposing) * 30
    else:
        debate_effectiveness = 30  # No opposition = full points

    # 4. Efficiency bonus (20 points max)
    if force_submitted:
        efficiency_score = 5  # Penalty for force submit
    elif initially_opposing == 0:
        efficiency_score = 20  # No debate needed = full efficiency
    elif total_debate_exchanges == 0:
        efficiency_score = 20
    else:
        # Fewer exchanges = better efficiency
        efficiency_score = max(5, 20 - (total_debate_exchanges * 2))

    total_score = initial_approval_score + consultation_score + debate_effectiveness + efficiency_score

    return {
        'round_number': round_number,
        'consultation_alignment_score': consultation_alignment,
        'members_initially_approving': initially_approving,
        'members_initially_opposing': initially_opposing,
        'members_convinced': convinced,
        'total_debate_exchanges': total_debate_exchanges,
        'force_submitted': force_submitted,
        'deliberation_score': round(total_score, 1),
        'score_breakdown': {
            'initial_approval': round(initial_approval_score, 1),
            'consultation': round(consultation_score, 1),
            'debate_effectiveness': round(debate_effectiveness, 1),
            'efficiency': round(efficiency_score, 1)
        }
    }


def display_company_dashboard(company_data: Dict):
    """Display company metrics dashboard"""
    st.subheader(f"📊 {company_data['company_name']} Dashboard")

    # Key metrics in columns
    metrics = company_data['metrics']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Annual Revenue",
            f"₹{metrics['total_revenue_annual']['value']} Cr",
            f"+{metrics['revenue_growth_yoy']['value']}% YoY"
        )

    with col2:
        st.metric(
            "EBITDA",
            f"₹{metrics['ebitda']['value']} Cr",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "Net Profit Margin",
            f"{metrics['net_profit_margin']['value']}%"
        )

    with col4:
        st.metric(
            "Employees",
            f"{metrics['employee_count']['value']:,}"
        )

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Platform Uptime",
            f"{metrics['platform_uptime']['value']}%"
        )

    with col2:
        st.metric(
            "NPS Score",
            f"{metrics['net_promoter_score']['value']}"
        )

    with col3:
        st.metric(
            "Customer Churn",
            f"{metrics['customer_churn_rate_annual']['value']}%"
        )

    with col4:
        high_risks = metrics['open_high_severity_risks']['value']
        st.metric(
            "High-Severity Risks",
            f"{high_risks}",
            delta=f"{high_risks} open" if high_risks > 0 else "All clear",
            delta_color="inverse" if high_risks > 0 else "normal"
        )

    # Expandable section for all metrics
    with st.expander("📈 View All Metrics"):
        metric_cols = st.columns(3)
        for idx, (key, metric) in enumerate(metrics.items()):
            with metric_cols[idx % 3]:
                priority_badge = "🔴 " if metric.get('priority') == 'High' else ""
                st.markdown(f"""
                **{priority_badge}{metric['description']}**
                `{metric['value']} {metric['unit']}`
                """)


def display_board_members_for_selection(board_members: List[Dict]) -> Optional[Dict]:
    """Display board members as clickable selection cards"""
    st.subheader("👤 Select Your Role")
    st.markdown("Choose which board member you want to play as:")

    # Display in columns with clickable buttons
    cols = st.columns(2)

    for idx, member in enumerate(board_members):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"""
                <div class="board-member-card">
                    <h4>{member['name']}</h4>
                    <p><strong>{member['role']}</strong></p>
                    <p><em>Expertise: {member['expertise']} | Tenure: {member['tenure_years']} years</em></p>
                    <p style="font-size: 0.9rem;">{member['personality'][:150]}...</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Play as {member['name']}", key=f"select_role_{idx}", use_container_width=True):
                    return member

    return None


def display_board_members(board_members: List[Dict], player_role: Optional[Dict] = None):
    """Display board member cards"""
    st.subheader("👥 Board of Directors")

    # Display in columns
    cols = st.columns(2)

    for idx, member in enumerate(board_members):
        with cols[idx % 2]:
            is_player = player_role and member['name'] == player_role['name']
            card_class = "selected-role-card" if is_player else "board-member-card"
            player_badge = " (YOU)" if is_player else ""

            with st.container():
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>{member['name']}{player_badge}</h4>
                    <p><strong>{member['role']}</strong></p>
                    <p><em>Expertise: {member['expertise']} | Tenure: {member['tenure_years']} years</em></p>
                    <p style="font-size: 0.9rem;">{member['personality'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)


def display_current_problems(problems: List[str]):
    """Display current company problems"""
    st.subheader("⚠️ Current Challenges")

    for problem in problems:
        st.markdown(f"- {problem}")


def display_module_info(module_data: Dict):
    """Display module information"""
    st.subheader(f"📚 {module_data['module_name']}")
    st.markdown(module_data['overview'])

    with st.expander("🎯 Learning Objectives"):
        for obj in module_data['learning_objectives']:
            st.markdown(f"- {obj}")

    with st.expander("📖 Key Topics"):
        for topic in module_data['topics']:
            st.markdown(f"**{topic['name']}**")
            st.markdown(f"_{topic['description']}_")
            st.markdown("---")


def parse_scenario_options(scenario: str) -> List[Dict]:
    """Parse options from scenario text"""
    options = []
    lines = scenario.split('\n')

    current_option = None
    for line in lines:
        line = line.strip()
        # Look for options like A), B), C), D) or A., B., C., D.
        for letter in ['A', 'B', 'C', 'D']:
            if line.startswith(f"{letter})") or line.startswith(f"{letter}."):
                if current_option:
                    options.append(current_option)
                option_text = line[2:].strip()
                current_option = {"letter": letter, "text": option_text}
                break

    if current_option:
        options.append(current_option)

    return options


def get_time_pressure_minutes(time_pressure: str) -> int:
    """Get the time limit in minutes based on time pressure setting"""
    return TIME_PRESSURE_MINUTES.get(time_pressure, 10)


def display_deliberation_phase(llm: genai.GenerativeModel, data: Dict,
                                state: SimulationState, player_decision: str) -> bool:
    """
    Display and manage the board deliberation phase.
    Returns True if deliberation is complete, False if still in progress.
    """
    logger.debug(f"display_deliberation_phase called for round {state.current_round}")

    company_data = data['company_data']
    module_data = data['module_data']  # Used in generate_member_stances
    player_role = st.session_state.get('player_role')
    round_num = state.current_round

    if not player_role:
        logger.error("player_role not found in session state!")
        st.error("Error: Player role not found. Please restart the simulation.")
        return False

    # Session state keys for this round
    delib_phase_key = f"deliberation_phase_{round_num}"
    stances_key = f"member_stances_{round_num}"
    current_dissenter_key = f"current_dissenter_{round_num}"
    debate_history_key = f"debate_history_{round_num}"
    force_key = f"force_submitted_{round_num}"

    # Initialize deliberation state if needed
    if delib_phase_key not in st.session_state:
        st.session_state[delib_phase_key] = 'inactive'
        logger.debug(f"Initialized delib_phase to 'inactive'")

    logger.debug(f"Current delib_phase: {st.session_state[delib_phase_key]}")

    # PHASE 1: Generate member stances (on first entry)
    if st.session_state[delib_phase_key] == 'inactive':
        logger.debug("Starting stance generation (phase: inactive -> generating)")
        st.session_state[delib_phase_key] = 'generating'
        st.session_state[debate_history_key] = []
        st.session_state[force_key] = False
        st.session_state[current_dissenter_key] = 0

        with st.spinner("Board members are reviewing your decision..."):
            scenario = st.session_state.get(f"scenario_round_{round_num}", "")
            logger.debug(f"Generating stances for scenario length: {len(scenario)}, decision length: {len(player_decision)}")
            try:
                stances = generate_member_stances(llm, company_data, module_data,
                                                  scenario, player_decision, player_role)
                st.session_state[stances_key] = stances
                logger.debug(f"Generated {len(stances)} member stances")
            except Exception as e:
                logger.error(f"Error generating stances: {e}")
                st.error(f"Error generating board member stances: {e}")
                st.session_state[delib_phase_key] = 'inactive'
                return False

        st.session_state[delib_phase_key] = 'review'
        logger.debug("Stance generation complete, transitioning to review phase")
        st.rerun()

    # Get current state
    member_stances = st.session_state.get(stances_key, {})

    # Display section header
    st.markdown("### 🏛️ Board Deliberation")
    st.markdown("""
    <div class="deliberation-header">
        <strong>The board is reviewing your decision.</strong> Members will share their perspectives based on their expertise.
    </div>
    """, unsafe_allow_html=True)

    # Categorize members by stance
    approving = [(n, s) for n, s in member_stances.items() if s['stance'] == 'APPROVE']
    opposing = [(n, s) for n, s in member_stances.items()
                if s['stance'] == 'OPPOSE' and s.get('convinced_in_round') is None]
    neutral = [(n, s) for n, s in member_stances.items() if s['stance'] == 'NEUTRAL']
    convinced = [(n, s) for n, s in member_stances.items() if s.get('convinced_in_round') is not None]

    # Display summary counts
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Supporting", len(approving), delta_color="normal")
    with col2:
        st.metric("Opposing", len(opposing), delta_color="inverse")
    with col3:
        st.metric("Neutral", len(neutral))
    with col4:
        st.metric("Convinced", len(convinced), delta_color="normal")

    st.markdown("---")

    # Display approving members
    if approving:
        st.markdown("**✅ Supporting your decision:**")
        for name, stance in approving:
            with st.expander(f"✅ {name} ({stance['member_role']}) - APPROVES"):
                st.markdown(f"*\"{stance['initial_reaction']}\"*")
                st.caption(f"**Expertise:** {stance['member_expertise']} | **Relevance:** {stance['expertise_relevance']}")

    # Display neutral members
    if neutral:
        st.markdown("**➖ Neutral:**")
        for name, stance in neutral:
            with st.expander(f"➖ {name} ({stance['member_role']}) - NEUTRAL"):
                st.markdown(f"*\"{stance['initial_reaction']}\"*")
                st.caption(f"**Expertise:** {stance['member_expertise']}")

    # Display convinced members
    if convinced:
        st.markdown("**🔄 Convinced during debate:**")
        for name, stance in convinced:
            with st.expander(f"🔄 {name} ({stance['member_role']}) - CONVINCED"):
                st.success(f"Convinced in debate round {stance['convinced_in_round']}")
                st.markdown(f"*Original objection:* {stance.get('original_counter_opinion', stance.get('counter_opinion', 'N/A'))}")

    # Display opposing members with debate interface
    if opposing:
        st.markdown("**⚠️ Opposing your decision:**")

        # Get current dissenter index
        current_idx = st.session_state.get(current_dissenter_key, 0)

        for idx, (name, stance) in enumerate(opposing):
            conviction_pct = stance['conviction_level'] * 10
            is_current = (idx == current_idx)

            if is_current:
                # Active debate with this dissenter
                st.markdown(f"""
                <div class="stance-card stance-oppose">
                    <h4>⚠️ {name} ({stance['member_role']}) - OPPOSES</h4>
                    <p><strong>Expertise:</strong> {stance['member_expertise']}</p>
                    <p><em>"{stance['initial_reaction']}"</em></p>
                    <div class="conviction-bar">
                        <div class="conviction-fill" style="width: {conviction_pct}%;"></div>
                    </div>
                    <small>Conviction Level: {stance['conviction_level']}/10</small>
                </div>
                """, unsafe_allow_html=True)

                # Show counter opinion
                st.error(f"**Counter-opinion:** {stance['counter_opinion']}")

                # Debate exchanges for this member
                exchanges = stance.get('debate_exchanges', 0)
                max_exchanges = 3

                if exchanges < max_exchanges:
                    st.markdown(f"#### 💬 Debate with {name} (Exchange {exchanges + 1} of {max_exchanges})")

                    # Use exchange number in key so it resets when exchange increments
                    response_key = f"debate_response_{round_num}_{name}_{exchanges}"

                    player_response = st.text_area(
                        f"Your response to {name}'s objection:",
                        key=response_key,
                        placeholder="Address their specific concerns and provide your reasoning...",
                        height=120
                    )

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if st.button(f"Submit Response", key=f"submit_debate_{round_num}_{name}_{exchanges}",
                                    type="primary", disabled=not player_response):
                            if player_response:
                                with st.spinner(f"{name} is considering your response..."):
                                    member_data = next(m for m in company_data['board_members']
                                                     if m['name'] == name)

                                    # Get debate history for this member
                                    member_debate_history = [
                                        h for h in st.session_state.get(debate_history_key, [])
                                        if h.get('dissenter_name') == name
                                    ]

                                    result = evaluate_debate_response(
                                        llm, member_data, company_data,
                                        stance['counter_opinion'],
                                        player_response,
                                        member_debate_history,
                                        player_role
                                    )

                                    # Record the exchange
                                    exchange_record = {
                                        'dissenter_name': name,
                                        'dissenter_argument': stance['counter_opinion'],
                                        'player_response': player_response,
                                        'llm_evaluation': result['evaluation'],
                                        'response_score': result['score'],
                                        'stance_changed': result['stance_changed']
                                    }
                                    st.session_state[debate_history_key].append(exchange_record)

                                    # Update stance data
                                    st.session_state[stances_key][name]['debate_exchanges'] = exchanges + 1

                                    if result['stance_changed']:
                                        # Store original counter opinion before updating
                                        st.session_state[stances_key][name]['original_counter_opinion'] = stance['counter_opinion']
                                        st.session_state[stances_key][name]['convinced_in_round'] = exchanges + 1
                                        # Move to next dissenter
                                        st.session_state[current_dissenter_key] = current_idx + 1
                                    else:
                                        # Update counter_opinion for next exchange
                                        st.session_state[stances_key][name]['counter_opinion'] = result['follow_up']

                                st.rerun()

                    with col2:
                        if st.button(f"Skip to Next Dissenter", key=f"skip_{round_num}_{name}"):
                            st.session_state[current_dissenter_key] = current_idx + 1
                            st.rerun()
                else:
                    st.warning(f"Maximum debate exchanges ({max_exchanges}) reached with {name}.")
                    if st.button(f"Move to Next Dissenter", key=f"move_next_{round_num}_{name}"):
                        st.session_state[current_dissenter_key] = current_idx + 1
                        st.rerun()
            else:
                # Not the current dissenter - show collapsed
                with st.expander(f"⚠️ {name} ({stance['member_role']}) - OPPOSES (waiting)"):
                    st.markdown(f"*\"{stance['initial_reaction']}\"*")
                    st.caption(f"Conviction: {stance['conviction_level']}/10")

    # Check if all dissenters have been addressed
    all_addressed = (st.session_state.get(current_dissenter_key, 0) >= len(opposing))

    st.markdown("---")

    # Resolution options
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if len(opposing) == 0 or all_addressed:
            # All resolved
            if len(opposing) == 0:
                st.success("✅ All board members support your decision!")
            else:
                remaining_opposed = sum(1 for n, s in member_stances.items()
                                       if s['stance'] == 'OPPOSE' and s.get('convinced_in_round') is None)
                if remaining_opposed > 0:
                    st.warning(f"⚠️ {remaining_opposed} board member(s) still oppose after debate.")
                else:
                    st.success("✅ All dissenters have been convinced!")

            if st.button("✓ Proceed with Decision", key=f"proceed_{round_num}", type="primary",
                        use_container_width=True):
                logger.debug("Proceed with Decision clicked")
                st.session_state[delib_phase_key] = 'resolved'
                st.rerun()
        else:
            # Still have dissenters to address
            remaining = len(opposing) - st.session_state.get(current_dissenter_key, 0)
            st.info(f"📋 {remaining} dissenter(s) remaining to address.")

        # Force submit option (always available)
        st.markdown("---")
        if st.button("⚡ Force Submit Decision", key=f"force_submit_{round_num}",
                    help="Submit without full board approval (scoring penalty applies)",
                    use_container_width=True):
            logger.debug("Force Submit clicked")
            st.session_state[force_key] = True
            st.session_state[delib_phase_key] = 'resolved'
            st.rerun()

    # Check if deliberation is complete
    is_resolved = st.session_state.get(delib_phase_key) == 'resolved'
    logger.debug(f"Deliberation phase returning: {is_resolved}")
    return is_resolved


def run_simulation_round(llm: genai.GenerativeModel, data: Dict,
                         state: SimulationState) -> None:
    """Run a single simulation round"""

    company_data = data['company_data']
    module_data = data['module_data']
    round_config = data['simulation_config']['rounds'][state.current_round]
    player_role = st.session_state.get('player_role')

    # Initialize consultation counter for this round
    round_consult_key = f"consultations_round_{state.current_round}"
    if round_consult_key not in st.session_state:
        st.session_state[round_consult_key] = 0

    # Initialize timer for this round
    timer_key = f"round_start_time_{state.current_round}"
    if timer_key not in st.session_state:
        st.session_state[timer_key] = datetime.now()

    # Get time pressure settings
    time_pressure = round_config.get('time_pressure', 'normal')
    time_limit_minutes = get_time_pressure_minutes(time_pressure)

    # Check if decision was already submitted (to stop timer)
    eval_key = f"evaluation_{state.current_round}"
    decision_submitted = eval_key in st.session_state

    # Phase: Briefing
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="round-indicator">
            Round {state.current_round + 1} of {state.total_rounds} |
            Difficulty: {round_config['difficulty'].title()} |
            Focus: {round_config.get('focus_area', 'General') or 'General'}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        consultations_left = 2 - st.session_state[round_consult_key]
        st.markdown(f"""
        <div class="consultation-counter">
            🗣️ Consultations: {consultations_left}/2 remaining
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Display countdown timer
        if not decision_submitted:
            # Calculate remaining time
            start_time = st.session_state[timer_key]
            elapsed = datetime.now() - start_time
            total_seconds = time_limit_minutes * 60
            remaining_seconds = max(0, int(total_seconds - elapsed.total_seconds()))

            # Store timer expired status
            timer_expired_key = f"timer_expired_{state.current_round}"
            if remaining_seconds <= 0:
                st.session_state[timer_expired_key] = True

            # Use JavaScript for live countdown with fixed end time
            timer_id = f"timer_{state.current_round}"

            # Calculate the absolute end timestamp (milliseconds since epoch)
            end_time = start_time + timedelta(seconds=total_seconds)
            end_timestamp_ms = int(end_time.timestamp() * 1000)

            # Determine initial timer class
            if remaining_seconds <= 0:
                timer_class = "timer-expired"
            elif time_pressure == "urgent":
                timer_class = "timer-urgent"
            elif time_pressure == "normal":
                timer_class = "timer-normal"
            else:
                timer_class = "timer-relaxed"

            st.markdown(f"""
            <div id="{timer_id}" class="timer-container {timer_class}">
                <div class="timer-display" id="{timer_id}_display">⏱️ {remaining_seconds // 60:02d}:{remaining_seconds % 60:02d}</div>
                <div class="timer-label" id="{timer_id}_label">Time Pressure: {time_pressure.title()}</div>
            </div>
            <script>
                (function() {{
                    var endTime = {end_timestamp_ms};
                    var timerId = "{timer_id}";
                    var timePressure = "{time_pressure}";

                    // Check if timer already exists to prevent duplicates
                    if (window['timerInterval_' + timerId]) {{
                        clearInterval(window['timerInterval_' + timerId]);
                    }}

                    function updateTimer() {{
                        var now = Date.now();
                        var remainingMs = endTime - now;
                        var remainingSeconds = Math.max(0, Math.floor(remainingMs / 1000));

                        var displayEl = document.getElementById(timerId + "_display");
                        var labelEl = document.getElementById(timerId + "_label");
                        var container = document.getElementById(timerId);

                        if (!displayEl || !labelEl || !container) {{
                            // Elements not found, stop timer
                            clearInterval(window['timerInterval_' + timerId]);
                            return;
                        }}

                        if (remainingSeconds <= 0) {{
                            displayEl.innerHTML = "⏱️ 00:00";
                            labelEl.innerHTML = "⚠️ Time's Up!";
                            container.className = "timer-container timer-expired";
                            clearInterval(window['timerInterval_' + timerId]);
                            return;
                        }}

                        var minutes = Math.floor(remainingSeconds / 60);
                        var seconds = remainingSeconds % 60;
                        displayEl.innerHTML = "⏱️ " + String(minutes).padStart(2, '0') + ":" + String(seconds).padStart(2, '0');

                        // Update color based on remaining time
                        if (remainingSeconds < 60) {{
                            container.className = "timer-container timer-urgent";
                        }} else if (remainingSeconds < 180 || timePressure === "urgent") {{
                            container.className = "timer-container timer-urgent";
                        }} else if (timePressure === "normal") {{
                            container.className = "timer-container timer-normal";
                        }} else {{
                            container.className = "timer-container timer-relaxed";
                        }}
                    }}

                    // Run immediately and then every second
                    updateTimer();
                    window['timerInterval_' + timerId] = setInterval(updateTimer, 1000);
                }})();
            </script>
            """, unsafe_allow_html=True)
        else:
            # Show completion message when decision was submitted
            st.markdown(f"""
            <div class="timer-container timer-relaxed">
                <div class="timer-display">✅ Submitted</div>
                <div class="timer-label">Decision Recorded</div>
            </div>
            """, unsafe_allow_html=True)

    # Display player role
    st.markdown(f"**Playing as:** {player_role['name']} - {player_role['role']}")

    # Check if timer has expired
    timer_expired_key = f"timer_expired_{state.current_round}"
    timer_expired = st.session_state.get(timer_expired_key, False)

    if timer_expired and not decision_submitted:
        st.warning("⚠️ **Time has expired!** You can still submit your decision, but this may affect your score.")

    # Generate or retrieve scenario
    scenario_key = f"scenario_round_{state.current_round}"
    if scenario_key not in st.session_state:
        with st.spinner("Generating scenario..."):
            st.session_state[scenario_key] = generate_scenario(
                llm, company_data, module_data, round_config, player_role
            )

    scenario = st.session_state[scenario_key]

    # Display scenario
    st.markdown("### 📋 Scenario")
    st.markdown(f"""
    <div class="scenario-box">
        {scenario}
    </div>
    """, unsafe_allow_html=True)

    # Parse options from scenario
    options = parse_scenario_options(scenario)

    # Consultation Section
    st.markdown("### 💬 Consultation")

    consultations_remaining = 2 - st.session_state[round_consult_key]

    if consultations_remaining > 0:
        consult_tab1, consult_tab2 = st.tabs(["👥 Consult Board Members", "🏛️ Consult Committee"])

        with consult_tab1:
            # Get available members (exclude player)
            available_members = [m for m in company_data['board_members'] if m['name'] != player_role['name']]
            member_names = [m['name'] for m in available_members]

            selected_members = st.multiselect(
                "Select board member(s) to consult:",
                member_names,
                key=f"member_select_{state.current_round}",
                help="You can select multiple members for a group discussion"
            )

            user_question = st.text_input(
                "Your question or topic for discussion:",
                key=f"member_question_{state.current_round}",
                placeholder="e.g., What are your thoughts on the compliance implications?"
            )

            if st.button("Ask Board Member(s)", key=f"ask_members_btn_{state.current_round}",
                        disabled=len(selected_members) == 0 or not user_question):
                if selected_members and user_question:
                    selected_member_data = [m for m in available_members if m['name'] in selected_members]

                    with st.spinner(f"{'Board members are' if len(selected_members) > 1 else selected_members[0] + ' is'} responding..."):
                        response = get_board_member_response(
                            llm, selected_member_data, company_data, module_data,
                            scenario, user_question,
                            st.session_state.get('conversation_history', []),
                            player_role
                        )

                        # Update consultation counter
                        st.session_state[round_consult_key] += 1

                        # Store in conversation history
                        if 'conversation_history' not in st.session_state:
                            st.session_state.conversation_history = []

                        member_label = ", ".join(selected_members) if len(selected_members) > 1 else selected_members[0]

                        st.session_state.conversation_history.append({
                            'role': 'user',
                            'content': user_question,
                            'member': member_label
                        })
                        st.session_state.conversation_history.append({
                            'role': 'assistant',
                            'content': response,
                            'member': member_label
                        })

                    st.rerun()

        with consult_tab2:
            # Committee consultation
            committees = company_data.get('committees', [])

            if committees:
                committee_names = [c['name'] for c in committees]
                selected_committee = st.selectbox(
                    "Select committee to consult:",
                    committee_names,
                    key=f"committee_select_{state.current_round}"
                )

                committee_question = st.text_input(
                    "Your question for the committee:",
                    key=f"committee_question_{state.current_round}",
                    placeholder="e.g., What is the committee's recommendation on this matter?"
                )

                if st.button("Consult Committee", key=f"ask_committee_btn_{state.current_round}",
                            disabled=not committee_question):
                    if committee_question:
                        selected_committee_data = next(c for c in committees if c['name'] == selected_committee)

                        with st.spinner(f"{selected_committee} is deliberating..."):
                            response = get_committee_response(
                                llm, selected_committee_data, company_data, module_data,
                                scenario, committee_question,
                                st.session_state.get('conversation_history', []),
                                player_role,
                                company_data['board_members']
                            )

                            # Update consultation counter
                            st.session_state[round_consult_key] += 1

                            # Store in conversation history
                            if 'conversation_history' not in st.session_state:
                                st.session_state.conversation_history = []

                            st.session_state.conversation_history.append({
                                'role': 'user',
                                'content': committee_question,
                                'member': selected_committee
                            })
                            st.session_state.conversation_history.append({
                                'role': 'assistant',
                                'content': response,
                                'member': selected_committee
                            })

                        st.rerun()
            else:
                st.info("No committees are available for consultation.")
    else:
        st.warning("⚠️ You have used all 2 consultations for this round. Please make your decision.")

    # Display conversation history for this round
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        with st.expander("📝 Discussion History", expanded=True):
            for entry in st.session_state.conversation_history:
                if entry['role'] == 'user':
                    st.markdown(f"**You asked {entry.get('member', 'Board')}:** {entry['content']}")
                else:
                    st.markdown(f"**{entry.get('member', 'Board Member')}:** {entry['content']}")
                st.markdown("---")

    # Decision Phase
    st.markdown("### ✅ Your Decision")

    # Initialize decision text key if not exists
    decision_key = f"decision_input_{state.current_round}"
    if decision_key not in st.session_state:
        st.session_state[decision_key] = ""

    # Clickable options if parsed
    if options:
        st.markdown("**Quick Select an Option:**")

        option_cols = st.columns(2)
        for idx, opt in enumerate(options):
            with option_cols[idx % 2]:
                if st.button(f"Option {opt['letter']}: {opt['text'][:50]}...",
                           key=f"option_{opt['letter']}_{state.current_round}",
                           use_container_width=True):
                    st.session_state[f"selected_option_{state.current_round}"] = opt
                    # Directly set the widget's session state key
                    st.session_state[decision_key] = f"Option {opt['letter']}: {opt['text']}"
                    st.rerun()

        st.markdown("---")
        st.markdown("**Or provide your detailed reasoning:**")

    decision = st.text_area(
        "Enter your decision and reasoning:",
        key=decision_key,
        placeholder="Describe your decision, the rationale behind it, and how you would implement it...",
        height=200
    )

    # Check if we're in deliberation phase
    pending_decision_key = f"pending_decision_{state.current_round}"
    delib_phase_key = f"deliberation_phase_{state.current_round}"
    eval_key = f"evaluation_{state.current_round}"

    logger.debug(f"Round {state.current_round}: delib_phase_key={delib_phase_key}, exists={delib_phase_key in st.session_state}")
    logger.debug(f"Round {state.current_round}: pending_decision_key={pending_decision_key}, exists={pending_decision_key in st.session_state}")
    if delib_phase_key in st.session_state:
        logger.debug(f"Round {state.current_round}: delib_phase value='{st.session_state[delib_phase_key]}'")

    # If deliberation has been triggered (pending decision exists and not yet resolved), show deliberation UI
    # 'inactive' means we just submitted and need to start deliberation
    # 'generating', 'review', 'debate' mean we're in the middle of deliberation
    pending_exists = pending_decision_key in st.session_state
    delib_not_exists = delib_phase_key not in st.session_state
    delib_not_resolved = st.session_state.get(delib_phase_key) != 'resolved'
    should_enter_delib = pending_exists and (delib_not_exists or delib_not_resolved)
    logger.debug(f"Round {state.current_round}: pending_exists={pending_exists}, delib_not_exists={delib_not_exists}, delib_not_resolved={delib_not_resolved}, should_enter_delib={should_enter_delib}")

    if should_enter_delib:
        logger.debug(f"Round {state.current_round}: Entering deliberation phase")

        deliberation_complete = display_deliberation_phase(
            llm, data, state, st.session_state[pending_decision_key]
        )
        logger.debug(f"Round {state.current_round}: deliberation_complete={deliberation_complete}")

        if not deliberation_complete:
            logger.debug(f"Round {state.current_round}: Returning from deliberation block (deliberation not complete)")
            return  # Don't show rest of round while in deliberation
        # If deliberation is complete, fall through to evaluation below

    # Check if deliberation is resolved but evaluation hasn't been done yet
    deliberation_resolved = st.session_state.get(delib_phase_key) == 'resolved'
    needs_evaluation = pending_decision_key in st.session_state and deliberation_resolved and eval_key not in st.session_state

    logger.debug(f"Round {state.current_round}: deliberation_resolved={deliberation_resolved}, needs_evaluation={needs_evaluation}")

    if needs_evaluation:
        logger.debug("Running evaluation after deliberation")
        with st.spinner("Evaluating your decision and calculating impacts..."):
            # Calculate board effectiveness
            stances = st.session_state.get(f"member_stances_{state.current_round}", {})
            debate_history = st.session_state.get(f"debate_history_{state.current_round}", [])
            force_submitted = st.session_state.get(f"force_submitted_{state.current_round}", False)

            # Get consultation alignment
            consultations = st.session_state.get('conversation_history', [])
            alignment_result = evaluate_consultation_alignment(
                llm, consultations, st.session_state[pending_decision_key], stances
            )

            # Calculate effectiveness score
            effectiveness = calculate_board_effectiveness_score(
                state.current_round, stances, debate_history,
                alignment_result.get('alignment_score', 50), force_submitted
            )

            # Store effectiveness data
            if "board_effectiveness_history" not in st.session_state:
                st.session_state.board_effectiveness_history = []
            st.session_state.board_effectiveness_history.append(effectiveness)
            st.session_state[f"board_effectiveness_{state.current_round}"] = effectiveness

            # Now evaluate the decision
            evaluation = evaluate_decision(
                llm, company_data, module_data,
                scenario, st.session_state[pending_decision_key], round_config, player_role
            )

            # Add board effectiveness to evaluation
            evaluation['board_effectiveness'] = effectiveness

            st.session_state[eval_key] = evaluation
            logger.debug(f"Evaluation stored, score: {evaluation.get('score', 'N/A')}")

            # Apply metric impacts
            if 'metric_impacts' in evaluation:
                impacts = evaluation['metric_impacts']
                logger.debug(f"Applying metric impacts")

                # Get current metrics or initialize from company data
                current_metrics = st.session_state.get('current_metrics', company_data['metrics'].copy())

                # Apply the impacts (with penalty for force submit)
                impact_values = impacts.get('impacts', {})
                if force_submitted:
                    logger.debug("Applying force submit penalty (15% reduction)")
                    # Apply 15% reduction to positive impacts as penalty
                    impact_values = {k: v * 0.85 if v > 0 else v for k, v in impact_values.items()}

                updated_metrics = apply_metric_impacts(current_metrics, impact_values)
                st.session_state.current_metrics = updated_metrics

                # Store impact reasons for display
                st.session_state.metric_impact_reasons = impacts.get('reasons', {})

                # Store impact summary
                st.session_state[f"impact_summary_{state.current_round}"] = impacts.get('summary', '')

            st.session_state.round_complete = True
            logger.debug("Round marked complete, calling rerun")
        st.rerun()

    logger.debug(f"Round {state.current_round}: Passed deliberation check, showing submit/evaluation UI")

    # Only show submit button if evaluation not yet done for this round
    if eval_key not in st.session_state:
        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("Submit Decision", key=f"submit_decision_{state.current_round}", type="primary"):
                if decision:
                    logger.debug(f"Submit Decision clicked. Decision length: {len(decision)}")
                    # Store decision and reset deliberation phase to start fresh
                    st.session_state[pending_decision_key] = decision
                    st.session_state[delib_phase_key] = 'inactive'
                    logger.debug(f"Stored pending decision, reset delib_phase to 'inactive', triggering deliberation")
                    st.rerun()
                else:
                    st.warning("Please enter your decision before submitting.")

    # Display evaluation if available
    logger.debug(f"Round {state.current_round}: eval_key exists={eval_key in st.session_state}")
    if eval_key in st.session_state:
        evaluation = st.session_state[eval_key]

        st.markdown("### 📊 Evaluation & Feedback")

        # Score display
        score = evaluation['score']
        score_color = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#dc3545"

        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: {score_color}; margin: 0;">Score: {score}/100</h2>
        </div>
        """, unsafe_allow_html=True)

        # Score Reasoning - show breakdown
        if evaluation.get('score_reasoning'):
            with st.expander("📋 Score Breakdown & Reasoning", expanded=True):
                st.markdown(evaluation['score_reasoning'])

        # Strengths and Improvements side by side
        col1, col2 = st.columns(2)
        with col1:
            if evaluation.get('strengths'):
                st.markdown("#### ✅ Strengths")
                st.success(evaluation['strengths'])
        with col2:
            if evaluation.get('improvements'):
                st.markdown("#### 🔧 Areas for Improvement")
                st.warning(evaluation['improvements'])

        # Key Learning Points
        if evaluation.get('learning_points'):
            st.markdown("#### 📚 Key Learning Points")
            st.info(evaluation['learning_points'])

        # Best Approach - What should have been done
        if evaluation.get('best_approach'):
            with st.expander("💡 Recommended Best Approach", expanded=False):
                st.markdown(evaluation['best_approach'])

        # Encouragement
        if evaluation.get('encouragement'):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <strong>💪 {evaluation['encouragement']}</strong>
            </div>
            """, unsafe_allow_html=True)

        # Display Metric Impact Summary
        impact_summary_key = f"impact_summary_{state.current_round}"
        if impact_summary_key in st.session_state and st.session_state[impact_summary_key]:
            st.markdown("### 📈 Business Impact")
            st.info(st.session_state[impact_summary_key])

        # Display detailed metric changes
        if 'metric_impacts' in evaluation and evaluation['metric_impacts'].get('impacts'):
            impacts = evaluation['metric_impacts']['impacts']
            reasons = evaluation['metric_impacts'].get('reasons', {})

            changed_metrics = {k: v for k, v in impacts.items() if v != 0}

            if changed_metrics:
                st.markdown("### 📊 Metric Changes from Your Decision")

                # Separate positive and negative impacts
                positive_impacts = {k: v for k, v in changed_metrics.items() if v > 0}
                negative_impacts = {k: v for k, v in changed_metrics.items() if v < 0}

                col1, col2 = st.columns(2)

                with col1:
                    if positive_impacts:
                        st.markdown("**✅ Positive Impacts:**")
                        for key, change in positive_impacts.items():
                            metric_name = key.replace('_', ' ').title()
                            reason = reasons.get(key, '')
                            st.success(f"**{metric_name}**: +{change}")
                            if reason:
                                st.caption(f"↳ {reason}")

                with col2:
                    if negative_impacts:
                        st.markdown("**⚠️ Negative Impacts:**")
                        for key, change in negative_impacts.items():
                            metric_name = key.replace('_', ' ').title()
                            reason = reasons.get(key, '')
                            st.error(f"**{metric_name}**: {change}")
                            if reason:
                                st.caption(f"↳ {reason}")

        # Next round button
        if st.button("Proceed to Next Round", key=f"next_round_{state.current_round}"):
            logger.debug(f"Proceed to Next Round clicked, advancing from round {state.current_round}")
            current_round = state.current_round
            st.session_state.current_round += 1
            st.session_state.conversation_history = []
            st.session_state.round_complete = False
            st.session_state.total_score = st.session_state.get('total_score', 0) + score
            # Clear impact reasons for next round display
            st.session_state.metric_impact_reasons = {}
            # Clear deliberation state for the completed round (keep history for final summary)
            # Note: We don't need to clear these as they're round-specific keys
            logger.debug(f"Advanced to round {st.session_state.current_round}")
            st.rerun()


def calculate_overall_grade(initial_metrics: Dict, final_metrics: Dict, avg_decision_score: float,
                            avg_board_effectiveness: float = None) -> Dict:
    """Calculate overall simulation grade based on metric changes, decision scores, and board effectiveness"""

    # Key metrics to evaluate (with weights and whether higher is better)
    key_metrics = {
        'total_revenue_annual': {'weight': 1.5, 'higher_better': True},
        'ebitda': {'weight': 1.5, 'higher_better': True},
        'net_profit_margin': {'weight': 1.2, 'higher_better': True},
        'revenue_growth_yoy': {'weight': 1.2, 'higher_better': True},
        'net_promoter_score': {'weight': 1.0, 'higher_better': True},
        'customer_churn_rate_annual': {'weight': 1.0, 'higher_better': False},
        'employee_engagement_score': {'weight': 0.8, 'higher_better': True},
        'annual_attrition_rate': {'weight': 0.8, 'higher_better': False},
        'regulatory_compliance_score': {'weight': 1.0, 'higher_better': True},
        'open_high_severity_risks': {'weight': 1.0, 'higher_better': False},
        'platform_uptime': {'weight': 0.8, 'higher_better': True},
    }

    metric_score = 0
    total_weight = 0
    improvements = 0
    declines = 0

    for metric_key, config in key_metrics.items():
        if metric_key in initial_metrics and metric_key in final_metrics:
            initial_val = initial_metrics[metric_key]['value']
            final_val = final_metrics[metric_key]['value']

            if initial_val != 0:
                pct_change = ((final_val - initial_val) / abs(initial_val)) * 100
            else:
                pct_change = final_val * 10  # Arbitrary scaling if initial was 0

            # Adjust for metrics where lower is better
            if not config['higher_better']:
                pct_change = -pct_change

            # Score contribution: positive change = positive score
            # Cap at ±20% impact per metric
            capped_change = max(-20, min(20, pct_change))
            metric_score += capped_change * config['weight']
            total_weight += config['weight']

            if pct_change > 0:
                improvements += 1
            elif pct_change < 0:
                declines += 1

    # Normalize metric score to 0-100 scale
    # -20 to +20 weighted average -> 0 to 100
    if total_weight > 0:
        avg_metric_change = metric_score / total_weight
        normalized_metric_score = 50 + (avg_metric_change * 2.5)  # Scale to 0-100
        normalized_metric_score = max(0, min(100, normalized_metric_score))
    else:
        normalized_metric_score = 50

    # Combine scores with board effectiveness if available
    # New weights: Decision 50%, Metrics 30%, Board Effectiveness 20%
    if avg_board_effectiveness is not None:
        final_score = (avg_decision_score * 0.5) + (normalized_metric_score * 0.3) + (avg_board_effectiveness * 0.2)
        board_effectiveness_component = avg_board_effectiveness * 0.2
    else:
        # Fallback to original weights if no board effectiveness data
        final_score = (avg_decision_score * 0.6) + (normalized_metric_score * 0.4)
        board_effectiveness_component = 0

    # Determine grade
    if final_score >= 90:
        grade = 'A+'
        grade_description = 'Outstanding Performance'
    elif final_score >= 85:
        grade = 'A'
        grade_description = 'Excellent Performance'
    elif final_score >= 80:
        grade = 'A-'
        grade_description = 'Very Good Performance'
    elif final_score >= 75:
        grade = 'B+'
        grade_description = 'Good Performance'
    elif final_score >= 70:
        grade = 'B'
        grade_description = 'Above Average Performance'
    elif final_score >= 65:
        grade = 'B-'
        grade_description = 'Satisfactory Performance'
    elif final_score >= 60:
        grade = 'C+'
        grade_description = 'Fair Performance'
    elif final_score >= 55:
        grade = 'C'
        grade_description = 'Average Performance'
    elif final_score >= 50:
        grade = 'C-'
        grade_description = 'Below Average Performance'
    elif final_score >= 45:
        grade = 'D'
        grade_description = 'Poor Performance'
    else:
        grade = 'F'
        grade_description = 'Needs Significant Improvement'

    return {
        'grade': grade,
        'grade_description': grade_description,
        'final_score': final_score,
        'decision_score_component': avg_decision_score * (0.5 if avg_board_effectiveness is not None else 0.6),
        'metric_score_component': normalized_metric_score * (0.3 if avg_board_effectiveness is not None else 0.4),
        'board_effectiveness_component': board_effectiveness_component,
        'metrics_improved': improvements,
        'metrics_declined': declines,
        'normalized_metric_score': normalized_metric_score,
        'avg_board_effectiveness': avg_board_effectiveness
    }


def display_board_effectiveness_summary(total_rounds: int):
    """Display the board effectiveness score in the final summary"""

    st.markdown("### 🏛️ Board Effectiveness Performance")

    effectiveness_history = st.session_state.get("board_effectiveness_history", [])

    if not effectiveness_history:
        st.info("No board effectiveness data available.")
        return 0

    # Calculate overall score
    total_score = sum(r['deliberation_score'] for r in effectiveness_history)
    avg_score = total_score / len(effectiveness_history)

    # Display overall grade
    if avg_score >= 80:
        grade = "A"
        grade_color = "#28a745"
        grade_desc = "Excellent Board Management"
    elif avg_score >= 60:
        grade = "B"
        grade_color = "#5cb85c"
        grade_desc = "Good Board Management"
    elif avg_score >= 40:
        grade = "C"
        grade_color = "#ffc107"
        grade_desc = "Fair Board Management"
    else:
        grade = "D"
        grade_color = "#dc3545"
        grade_desc = "Needs Improvement"

    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f0f7ff 0%, #e6f0ff 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: {grade_color}; margin: 0;">Board Effectiveness: {grade}</h2>
        <p style="color: #666; font-size: 1.1rem;">{grade_desc}</p>
        <p style="color: #333;">Average Score: {avg_score:.1f}/100</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)

    total_convinced = sum(r.get('members_convinced', 0) for r in effectiveness_history)
    total_force_submits = sum(1 for r in effectiveness_history if r.get('force_submitted', False))
    total_debates = sum(r.get('total_debate_exchanges', 0) for r in effectiveness_history)
    total_opposing = sum(r.get('members_initially_opposing', 0) for r in effectiveness_history)

    with col1:
        st.metric("Dissenters Convinced", total_convinced,
                 delta=f"of {total_opposing}" if total_opposing > 0 else None)
    with col2:
        st.metric("Force Submits", total_force_submits,
                 delta_color="inverse" if total_force_submits > 0 else "normal")
    with col3:
        st.metric("Debate Exchanges", total_debates)
    with col4:
        avg_alignment = sum(r.get('consultation_alignment_score', 50) for r in effectiveness_history) / len(effectiveness_history)
        st.metric("Avg Consultation Alignment", f"{avg_alignment:.0f}%")

    # Per-round breakdown
    with st.expander("📊 Round-by-Round Board Effectiveness", expanded=False):
        for round_data in effectiveness_history:
            score = round_data.get('deliberation_score', 0)
            score_color = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#dc3545"

            st.markdown(f"""
            **Round {round_data.get('round_number', 0) + 1}:**
            <span style="color: {score_color}; font-weight: bold;">{score}/100</span>
            """, unsafe_allow_html=True)

            breakdown = round_data.get('score_breakdown', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption(f"Initial Approval: {breakdown.get('initial_approval', 0)}/25")
            with col2:
                st.caption(f"Consultation: {breakdown.get('consultation', 0)}/25")
            with col3:
                st.caption(f"Debate: {breakdown.get('debate_effectiveness', 0)}/30")
            with col4:
                st.caption(f"Efficiency: {breakdown.get('efficiency', 0)}/20")

            # Show key stats
            approving = round_data.get('members_initially_approving', 0)
            opposing = round_data.get('members_initially_opposing', 0)
            convinced = round_data.get('members_convinced', 0)

            st.caption(f"👍 {approving} approved | 👎 {opposing} opposed | 🔄 {convinced} convinced")

            if round_data.get('force_submitted', False):
                st.warning("⚠️ Decision was force-submitted")
            st.markdown("---")

    return avg_score


def display_final_summary(data: Dict):
    """Display final simulation summary"""
    st.markdown("## 🎉 Simulation Complete!")

    player_role = st.session_state.get('player_role', {})
    st.markdown(f"**You played as:** {player_role.get('name', 'Unknown')} - {player_role.get('role', 'Unknown')}")

    total_score = st.session_state.get('total_score', 0)
    rounds_completed = st.session_state.get('current_round', 0)
    avg_score = total_score / max(rounds_completed, 1)

    # Get initial and final metrics
    initial_metrics = st.session_state.get('initial_metrics', data['company_data']['metrics'])
    final_metrics = st.session_state.get('current_metrics', data['company_data']['metrics'])

    # Calculate board effectiveness average
    effectiveness_history = st.session_state.get("board_effectiveness_history", [])
    avg_board_effectiveness = None
    if effectiveness_history:
        avg_board_effectiveness = sum(r['deliberation_score'] for r in effectiveness_history) / len(effectiveness_history)

    # Calculate overall grade with board effectiveness
    grade_info = calculate_overall_grade(initial_metrics, final_metrics, avg_score, avg_board_effectiveness)

    # Display Overall Grade prominently
    grade_color = {
        'A+': '#28a745', 'A': '#28a745', 'A-': '#5cb85c',
        'B+': '#8bc34a', 'B': '#9acd32', 'B-': '#cddc39',
        'C+': '#ffc107', 'C': '#ff9800', 'C-': '#ff5722',
        'D': '#f44336', 'F': '#d32f2f'
    }.get(grade_info['grade'], '#666')

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: {grade_color}; font-size: 4rem; margin: 0;">{grade_info['grade']}</h1>
        <h3 style="color: #333; margin: 0.5rem 0;">{grade_info['grade_description']}</h3>
        <p style="color: #666; font-size: 1.2rem;">Overall Score: {grade_info['final_score']:.1f}/100</p>
    </div>
    """, unsafe_allow_html=True)

    # Score breakdown
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rounds Completed", rounds_completed)

    with col2:
        st.metric("Decision Score", f"{avg_score:.1f}/100")

    with col3:
        st.metric("Metrics Improved", f"{grade_info['metrics_improved']}", delta=f"+{grade_info['metrics_improved']}")

    with col4:
        st.metric("Metrics Declined", f"{grade_info['metrics_declined']}", delta=f"-{grade_info['metrics_declined']}", delta_color="inverse")

    # Score composition
    st.markdown("### 📊 Score Composition")
    if avg_board_effectiveness is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            **Decision Quality (50%):** {grade_info['decision_score_component']:.1f} pts
            *Based on your choices across {rounds_completed} rounds*
            """)
        with col2:
            st.markdown(f"""
            **Business Impact (30%):** {grade_info['metric_score_component']:.1f} pts
            *Based on metric improvements: {grade_info['normalized_metric_score']:.1f}/100*
            """)
        with col3:
            st.markdown(f"""
            **Board Effectiveness (20%):** {grade_info['board_effectiveness_component']:.1f} pts
            *Based on board management: {avg_board_effectiveness:.1f}/100*
            """)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Decision Quality (60%):** {grade_info['decision_score_component']:.1f} pts
            *Based on your choices across {rounds_completed} rounds*
            """)
        with col2:
            st.markdown(f"""
            **Business Impact (40%):** {grade_info['metric_score_component']:.1f} pts
            *Based on metric improvements: {grade_info['normalized_metric_score']:.1f}/100*
            """)

    # Display Board Effectiveness Summary
    if effectiveness_history:
        display_board_effectiveness_summary(rounds_completed)

    # Metrics Before vs After Comparison
    st.markdown("### 📈 Metrics Comparison: Before vs After Simulation")

    # Group metrics by category
    metric_categories = {
        '💰 Financial': ['total_revenue_annual', 'annual_recurring_revenue', 'ebitda',
                        'net_profit_margin', 'revenue_growth_yoy', 'monthly_burn_rate'],
        '👥 Customer': ['net_promoter_score', 'customer_churn_rate_annual',
                       'customer_lifetime_value', 'customer_acquisition_cost'],
        '⚙️ Operations': ['platform_uptime', 'deployment_frequency',
                         'average_incident_resolution_time', 'automation_coverage'],
        '👔 Human Resources': ['employee_count', 'employee_engagement_score',
                              'annual_attrition_rate', 'avg_training_hours_per_employee'],
        '🛡️ Risk & Compliance': ['regulatory_compliance_score', 'open_high_severity_risks',
                                 'data_privacy_incident_count']
    }

    # Metrics where lower is better
    inverse_metrics = ['customer_churn_rate_annual', 'annual_attrition_rate',
                      'open_high_severity_risks', 'monthly_burn_rate',
                      'data_processing_latency', 'average_incident_resolution_time',
                      'data_privacy_incident_count', 'customer_acquisition_cost']

    for category, metric_keys in metric_categories.items():
        with st.expander(category, expanded=True):
            for key in metric_keys:
                if key in initial_metrics and key in final_metrics:
                    initial = initial_metrics[key]
                    final = final_metrics[key]

                    initial_val = initial['value']
                    final_val = final['value']
                    change = final_val - initial_val

                    # Calculate percentage change
                    if initial_val != 0:
                        pct_change = ((final_val - initial_val) / abs(initial_val)) * 100
                    else:
                        pct_change = 0

                    # Determine if change is positive or negative
                    is_inverse = key in inverse_metrics
                    is_improvement = (change < 0 if is_inverse else change > 0)
                    is_decline = (change > 0 if is_inverse else change < 0)

                    # Format values
                    unit = initial.get('unit', '')

                    # Create comparison display
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                    with col1:
                        priority_badge = "🔴 " if initial.get('priority') == 'High' else ""
                        st.markdown(f"**{priority_badge}{initial['description']}**")

                    with col2:
                        st.markdown(f"Before: `{initial_val} {unit}`")

                    with col3:
                        st.markdown(f"After: `{final_val} {unit}`")

                    with col4:
                        if change != 0:
                            change_str = f"{change:+.1f}" if isinstance(change, float) else f"{change:+d}"
                            pct_str = f"({pct_change:+.1f}%)"
                            if is_improvement:
                                st.markdown(f"✅ {change_str} {pct_str}")
                            elif is_decline:
                                st.markdown(f"⚠️ {change_str} {pct_str}")
                            else:
                                st.markdown(f"➡️ {change_str} {pct_str}")
                        else:
                            st.markdown("➡️ No change")

    # Performance assessment
    if avg_score >= 80:
        performance = "Excellent! You demonstrated strong understanding of corporate governance and made decisions that positively impacted the business."
    elif avg_score >= 60:
        performance = "Good performance! You have a solid grasp of board dynamics. Focus on understanding the broader business impact of your decisions."
    else:
        performance = "Keep learning! Review the module materials to strengthen your understanding. Consider how decisions affect multiple stakeholders and metrics."

    st.markdown(f"""
    <div class="info-box">
        <h3>Performance Assessment</h3>
        <p>{performance}</p>
    </div>
    """, unsafe_allow_html=True)

    # Round-by-Round Summary
    st.markdown("### 🎯 Round-by-Round Performance Review")
    st.markdown("*Review your decisions, scores, and see what the best approach would have been for each scenario.*")

    for round_num in range(rounds_completed):
        eval_key = f"evaluation_{round_num}"
        scenario_key = f"scenario_round_{round_num}"

        if eval_key in st.session_state:
            evaluation = st.session_state[eval_key]
            scenario = st.session_state.get(scenario_key, "Scenario not available")

            # Get round config
            round_config = data['simulation_config']['rounds'][round_num]

            # Determine score color
            round_score = evaluation.get('score', 0)
            score_color = "#28a745" if round_score >= 70 else "#ffc107" if round_score >= 50 else "#dc3545"
            score_emoji = "🟢" if round_score >= 70 else "🟡" if round_score >= 50 else "🔴"

            with st.expander(f"{score_emoji} Round {round_num + 1}: Score {round_score}/100 | Difficulty: {round_config.get('difficulty', 'N/A').title()}", expanded=False):
                # Round Overview
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>Focus Area:</strong> {round_config.get('focus_area', 'General')}<br>
                    <strong>Time Pressure:</strong> {round_config.get('time_pressure', 'normal').title()}<br>
                    <strong>Your Score:</strong> <span style="color: {score_color}; font-weight: bold;">{round_score}/100</span>
                </div>
                """, unsafe_allow_html=True)

                # Scenario
                st.markdown("#### 📋 Scenario Presented")
                with st.container():
                    st.markdown(f"""
                    <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; max-height: 200px; overflow-y: auto;">
                        {scenario[:1000]}{'...' if len(scenario) > 1000 else ''}
                    </div>
                    """, unsafe_allow_html=True)

                # Your Decision
                st.markdown("#### 🎯 Your Decision")
                decision_text = evaluation.get('decision', 'Decision not recorded')
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3;">
                    {decision_text}
                </div>
                """, unsafe_allow_html=True)

                # Score Breakdown
                if evaluation.get('score_reasoning'):
                    st.markdown("#### 📊 Score Breakdown")
                    st.markdown(evaluation['score_reasoning'])

                # What You Did Well vs What Could Be Improved
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### ✅ What You Did Well")
                    if evaluation.get('strengths'):
                        st.success(evaluation['strengths'])
                    else:
                        st.info("No specific strengths recorded")

                with col2:
                    st.markdown("#### 🔧 Areas for Improvement")
                    if evaluation.get('improvements'):
                        st.warning(evaluation['improvements'])
                    else:
                        st.info("No specific improvements recorded")

                # THE BEST APPROACH - Key Section
                st.markdown("#### 💡 Recommended Best Approach")
                if evaluation.get('best_approach'):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1.5rem; border-radius: 10px; border: 2px solid #28a745;">
                        <strong style="color: #155724;">What would have been the ideal decision:</strong><br><br>
                        {evaluation['best_approach']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Best approach recommendation not available")

                # Key Learning Points
                if evaluation.get('learning_points'):
                    st.markdown("#### 📚 Key Learning Points")
                    st.info(evaluation['learning_points'])

                # Metric Impact Summary for this round
                impact_summary = st.session_state.get(f"impact_summary_{round_num}", "")
                if impact_summary:
                    st.markdown("#### 📈 Business Impact from This Decision")
                    st.markdown(impact_summary)

                st.markdown("---")

    # Key learnings from module
    module_data = data['module_data']
    st.markdown("### 📚 Key Concepts Covered")

    for topic in module_data['topics'][:5]:
        with st.expander(topic['name']):
            st.markdown(topic['description'])
            if topic.get('key_principles'):
                st.markdown("**Key Principles:**")
                for principle in topic['key_principles']:
                    st.markdown(f"- {principle}")

    if st.button("Start New Simulation"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main():
    """Main application entry point"""

    # Title
    st.markdown('<h1 class="main-header">🏢 Board Room Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Corporate Governance Training & Decision Making</p>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Load API key from Streamlit secrets
        if "GEMINI_API_KEY" in st.secrets:
            st.session_state.api_key = st.secrets["GEMINI_API_KEY"]

        st.markdown("---")

        # File selection - use the directory where app.py is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        json_files = [f for f in os.listdir(app_dir) if f.endswith('.json')]

        if json_files:
            selected_file = st.selectbox(
                "Select Simulation File",
                json_files
            )
            st.session_state.selected_file = os.path.join(app_dir, selected_file)
        else:
            st.warning("No JSON files found in the directory.")
            st.session_state.selected_file = None

        st.markdown("---")

        # Display current role if selected
        if st.session_state.get('player_role'):
            st.markdown("**Your Role:**")
            role = st.session_state.player_role
            st.info(f"{role['name']}\n{role['role']}")

        st.markdown("---")

        # Simulation controls
        if st.button("🔄 Reset Simulation"):
            for key in list(st.session_state.keys()):
                if key not in ['api_key', 'selected_file']:
                    del st.session_state[key]
            st.rerun()

    # Function to display metrics in sidebar during simulation
    def display_sidebar_metrics(company_data: Dict, impact_reasons: Dict = None):
        """Display company metrics in sidebar during simulation"""
        with st.sidebar:
            st.markdown("---")

            # Get current metrics (may be updated from session state)
            metrics = st.session_state.get('current_metrics', company_data['metrics'])

            # HIGH PRIORITY METRICS SECTION - Always visible at top
            st.header("🔴 High Priority Metrics")

            high_priority_metrics = {k: v for k, v in metrics.items() if v.get('priority') == 'High'}

            if high_priority_metrics:
                for key, metric in high_priority_metrics.items():
                    change = metric.get('change', 0)
                    delta_str = None
                    if change != 0:
                        delta_str = f"{change:+.1f}" if isinstance(change, float) else f"{change:+d}"

                    st.metric(
                        metric['description'],
                        f"{metric['value']} {metric['unit']}",
                        delta=delta_str,
                        delta_color="normal" if change >= 0 else "inverse"
                    )

                    # Show reason for change if available
                    if impact_reasons and key in impact_reasons:
                        st.caption(f"📝 {impact_reasons[key]}")
            else:
                st.info("No high priority metrics flagged")

            st.markdown("---")
            st.header("📊 All Metrics")

            # Get impact reasons from session state if not provided
            if impact_reasons is None:
                impact_reasons = st.session_state.get('metric_impact_reasons', {})

            # Helper function to display metric with change
            def show_metric(key, label, format_str, suffix=""):
                if key in metrics:
                    metric = metrics[key]
                    value = metric['value']
                    change = metric.get('change', 0)

                    # Format the display value
                    if isinstance(value, float):
                        display_val = f"{format_str.format(value)}{suffix}"
                    else:
                        display_val = f"{value}{suffix}"

                    # Calculate delta string
                    delta_str = None
                    if change != 0:
                        if isinstance(change, float):
                            delta_str = f"{change:+.2f}"
                        else:
                            delta_str = f"{change:+d}"

                    # Determine if this metric is "inverse" (lower is better)
                    inverse_metrics = ['customer_churn_rate_annual', 'annual_attrition_rate',
                                      'open_high_severity_risks', 'monthly_burn_rate',
                                      'data_processing_latency', 'average_incident_resolution_time']

                    delta_color = "inverse" if key in inverse_metrics else "normal"

                    st.metric(label, display_val, delta=delta_str, delta_color=delta_color)

                    # Show reason if available
                    if key in impact_reasons and impact_reasons[key]:
                        st.caption(f"↳ {impact_reasons[key]}")

            # Key Financial Metrics
            with st.expander("💰 Financial", expanded=True):
                show_metric('total_revenue_annual', "Revenue", "₹{:.0f}", " Cr")
                show_metric('annual_recurring_revenue', "ARR", "₹{:.0f}", " Cr")
                show_metric('ebitda', "EBITDA", "₹{:.0f}", " Cr")
                show_metric('net_profit_margin', "Net Profit Margin", "{:.1f}", "%")
                show_metric('revenue_growth_yoy', "Revenue Growth YoY", "{:.1f}", "%")
                show_metric('monthly_burn_rate', "Monthly Burn Rate", "₹{:.1f}", " Cr")

            # Customer Metrics
            with st.expander("👥 Customer", expanded=False):
                show_metric('net_promoter_score', "NPS Score", "{:.0f}", "")
                show_metric('customer_churn_rate_annual', "Customer Churn", "{:.1f}", "%")
                show_metric('customer_lifetime_value', "CLV", "₹{:.1f}", " Cr")
                show_metric('customer_acquisition_cost', "CAC", "₹{:.1f}", " L")
                show_metric('average_contract_value', "Avg Contract Value", "₹{:.1f}", " Cr")
                show_metric('expansion_revenue_rate', "Expansion Revenue", "{:.0f}", "%")
                show_metric('support_ticket_csat', "Support CSAT", "{:.1f}", "/5")

            # Operational Metrics
            with st.expander("⚙️ Operations", expanded=False):
                show_metric('platform_uptime', "Platform Uptime", "{:.2f}", "%")
                show_metric('deployment_frequency', "Deployment Freq", "{:.0f}", "/month")
                show_metric('average_incident_resolution_time', "Incident Resolution", "{:.1f}", " hrs")
                show_metric('automation_coverage', "Automation Coverage", "{:.0f}", "%")
                show_metric('infrastructure_cost_efficiency', "Infra Cost Efficiency", "{:.0f}", "%")
                show_metric('data_processing_latency', "Data Latency", "{:.0f}", " ms")
                show_metric('project_delivery_on_time_rate', "On-Time Delivery", "{:.0f}", "%")

            # HR Metrics
            with st.expander("👔 Human Resources", expanded=False):
                show_metric('employee_count', "Employees", "{:,.0f}", "")
                show_metric('employee_engagement_score', "Engagement Score", "{:.0f}", "/100")
                show_metric('annual_attrition_rate', "Attrition Rate", "{:.1f}", "%")
                show_metric('avg_training_hours_per_employee', "Training Hours", "{:.0f}", " hrs")
                show_metric('internal_promotion_rate', "Internal Promotion", "{:.0f}", "%")
                show_metric('diversity_ratio_women_percentage', "Diversity (Women)", "{:.0f}", "%")

            # Risk & Compliance
            with st.expander("🛡️ Risk & Compliance", expanded=False):
                show_metric('regulatory_compliance_score', "Compliance Score", "{:.0f}", "%")
                show_metric('open_high_severity_risks', "High-Severity Risks", "{:.0f}", "")
                show_metric('data_privacy_incident_count', "Privacy Incidents", "{:.0f}", "")
                show_metric('carbon_footprint_yoy_change', "Carbon Footprint", "{:.0f}", "% YoY")
                show_metric('r_and_d_spend_percentage_of_revenue', "R&D Spend", "{:.1f}", "%")

    # Check prerequisites
    if not st.session_state.get('api_key'):
        st.error("⚠️ API Key not configured. Please add GEMINI_API_KEY to your Streamlit secrets.")
        st.markdown("""
        ### Configuration Required
        Add your Google AI API key to `.streamlit/secrets.toml`:
        ```toml
        GEMINI_API_KEY = "your-api-key-here"
        ```
        """)
        return

    if not st.session_state.get('selected_file'):
        st.warning("⚠️ Please select a simulation file in the sidebar.")
        return

    # Load simulation data
    data = load_simulation_data(st.session_state.selected_file)

    if not data:
        st.error("Failed to load simulation data.")
        return

    # Initialize LLM
    try:
        llm = initialize_llm(st.session_state.api_key)
    except Exception as e:
        st.error(f"Failed to initialize AI model: {e}")
        return

    # Initialize simulation state
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0

    if 'simulation_started' not in st.session_state:
        st.session_state.simulation_started = False

    company_data = data['company_data']
    module_data = data['module_data']
    simulation_config = data['simulation_config']

    # Main content area
    if not st.session_state.get('player_role'):
        # Role selection phase
        st.markdown("### Step 1: Choose Your Role")
        st.markdown("Select which board member you want to play as during this simulation.")

        selected_role = display_board_members_for_selection(company_data['board_members'])

        if selected_role:
            st.session_state.player_role = selected_role
            st.rerun()

        # Also show company and module info
        st.markdown("---")
        st.markdown("### 📊 Company Overview")
        display_company_dashboard(company_data)
        display_current_problems(company_data['current_problems'])

        st.markdown("---")
        st.markdown("### 📚 Module Information")
        display_module_info(module_data)

    elif not st.session_state.simulation_started:
        # Setup phase - show company and module info after role selection
        player_role = st.session_state.player_role

        st.success(f"✅ You are playing as **{player_role['name']}** - {player_role['role']}")

        tab1, tab2, tab3 = st.tabs(["🏢 Company Overview", "👥 Board Members", "📚 Module Info"])

        with tab1:
            display_company_dashboard(company_data)
            st.markdown("---")
            display_current_problems(company_data['current_problems'])

            # Initial scenario
            st.markdown("### 📋 Initial Scenario")
            st.markdown(f"""
            <div class="scenario-box">
                {company_data['initial_scenario']}
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            display_board_members(company_data['board_members'], player_role)

            # Committees
            if company_data.get('committees'):
                st.markdown("### 🏛️ Board Committees")
                for committee in company_data['committees']:
                    with st.expander(committee['name']):
                        st.markdown(f"**Type:** {committee['type']}")
                        st.markdown(f"**Purpose:** {committee['purpose']}")
                        st.markdown(f"**Chairperson:** {committee['chairperson']}")
                        st.markdown(f"**Members:** {', '.join(committee['members'])}")

        with tab3:
            display_module_info(module_data)

            # Key terms
            with st.expander("📖 Key Terms & Definitions"):
                for term, definition in list(module_data['key_terms'].items())[:15]:
                    st.markdown(f"**{term}:** {definition}")

        # Start simulation button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Simulation", type="primary", use_container_width=True):
                st.session_state.simulation_started = True
                st.session_state.current_round = 0
                st.session_state.total_score = 0
                st.session_state.conversation_history = []
                # Store INITIAL metrics (before simulation) for final comparison
                st.session_state.initial_metrics = {k: v.copy() for k, v in company_data['metrics'].items()}
                # Initialize current metrics with a deep copy
                st.session_state.current_metrics = {k: v.copy() for k, v in company_data['metrics'].items()}
                st.session_state.metric_impact_reasons = {}
                st.rerun()

    elif st.session_state.current_round >= simulation_config['total_rounds']:
        # Simulation complete - show metrics in sidebar with final impact reasons
        impact_reasons = st.session_state.get('metric_impact_reasons', {})
        display_sidebar_metrics(company_data, impact_reasons)
        display_final_summary(data)

    else:
        # Active simulation - show metrics in sidebar with impact reasons
        impact_reasons = st.session_state.get('metric_impact_reasons', {})
        display_sidebar_metrics(company_data, impact_reasons)

        state = SimulationState(
            current_round=st.session_state.current_round,
            total_rounds=simulation_config['total_rounds']
        )

        # Progress bar
        progress = (state.current_round) / state.total_rounds
        st.progress(progress, text=f"Progress: {state.current_round}/{state.total_rounds} rounds")

        run_simulation_round(llm, data, state)


if __name__ == "__main__":
    main()
