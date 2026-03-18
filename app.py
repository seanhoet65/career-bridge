"""
app.py  —  Skill Gap Analyzer  |  ESADE MBAn · Prototyping Assignment 2
────────────────────────────────────────────────────────────────────────
LLM features (all non-straightforward, all using Gemini):

  1. CV Skill Extraction  (2-call pipeline, sidebar)
     Call 1  : Extract raw skills from uploaded CV text
     Call 2  : Normalize to canonical role skill names — output feeds
               directly into the Python gap_score engine

  2. JD Analysis  (2-call pipeline, Part A tab)
     Call 1  : Extract structured requirements from free-text job description
     Call 2  : Score candidate against those structured requirements
               (cleaner input → better fit scoring)

  3. Career Advisor — Agentic multi-call with tools  (Tab 2)
     Gemini Automatic Function Calling loop (≤8 iterations per turn).
     LLM decides which of 6 Python tools to invoke; tools run real
     numerical computations; results feed subsequent LLM reasoning.

  4. AI-Enhanced Roadmap  (3-call pipeline, opt-in toggle, Part B)
     Call 1  : Strategic leverage analysis — rank gaps by strategic value
     Call 2  : Generate week-by-week JSON plan using Call 1 output
     Call 3  : Self-critique & refine the plan for realism/sequencing
               → output drives a Plotly Gantt chart visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import re

from roles_data import (
    CURATED_ROLES, LEARNING_CATALOGUE,
    get_courses, compute_gap, gap_score, roi_score,
)
from llm_client import simple_call, json_call
from agent import create_chat_session, run_advisor_turn

# ══════════════════════════════════════════════════════════════════════════════
# DEMO PROFILE — Sean Hoet · ESADE MBAn
# Canonical skill names matched from CV against CURATED_ROLES.
# No API call — hardcoded so the app is instantly interactive for visitors.
# ══════════════════════════════════════════════════════════════════════════════
DEMO_SKILLS = [
    # ── SQL (matched across multiple roles) ───────────────────────────────────
    "SQL (queries, joins, CTEs)",
    "SQL & large dataset handling",
    "SQL & data pipelines",
    "SQL & basic data analysis",
    "SQL & data querying",
    "SQL or Excel for operational data",
    # ── Python & Machine Learning ─────────────────────────────────────────────
    "Python — pandas & numpy",
    "Python (scikit-learn, statsmodels)",
    "ML — classification, regression, clustering",
    "ML algorithms (supervised/unsupervised)",
    "Feature engineering & selection",
    "NLP basics (text processing, embeddings)",
    "ML lifecycle (training, evaluation, deployment)",
    # ── Git / Version Control ─────────────────────────────────────────────────
    "Git version control",
    "Git & reproducible research",
    "Git & CI/CD",
    "Git & code review workflows",
    # ── Data & Visualisation ──────────────────────────────────────────────────
    "Data cleaning & wrangling",
    "Data visualisation (matplotlib, Tableau)",
    "Dashboard design",
    "KPI definition & reporting",
    "KPI definition, dashboards & reporting",
    # ── Excel ─────────────────────────────────────────────────────────────────
    "Excel & pivot tables",
    "Excel & financial modelling",
    "Excel (advanced)",
    # ── AI / LLM & Product ────────────────────────────────────────────────────
    "Prompt engineering & LLM product experience",
    "Defining AI success metrics & eval frameworks",
    "Data product roadmapping",
    "Agile with ML-specific sprints",
    "Stakeholder education on AI capabilities",
    "Working with data scientists & ML engineers",
    # ── Stakeholder & Communication ───────────────────────────────────────────
    "Stakeholder presentation of insights",
    "Communicating findings to non-technical stakeholders",
    "Stakeholder alignment & executive communication",
    "Stakeholder workshops & facilitation",
    "Executive stakeholder management",
    "Executive presentation skills",
    "Cross-functional stakeholder management",
    "Client relationship management & exec stakeholder mapping",
    # ── Strategy & Consulting ─────────────────────────────────────────────────
    "Structured problem solving (MECE, issue trees)",
    "PowerPoint storytelling & slide writing",
    "Market sizing & competitive analysis",
    "Competitive & market analysis",
    "Due diligence & research synthesis",
    "Industry benchmarking",
    "Business case writing",
    "Project management (workplan, milestones)",
    "Client workshop facilitation",
    # ── Agile & Product Management ────────────────────────────────────────────
    "Agile & sprint planning",
    "Data analysis for product decisions",
    "Requirements gathering & documentation",
    # ── Cloud & APIs ──────────────────────────────────────────────────────────
    "REST API design & development",
    "Cloud services (AWS, GCP, Azure)",
    # ── CRM, Sales & Account Management ──────────────────────────────────────
    "CRM (Salesforce, HubSpot)",
    "CRM tools (Salesforce, HubSpot)",
    "CRM data hygiene (Salesforce, HubSpot)",
    "Revenue forecasting & pipeline management",
    "Churn risk identification",
    "Campaign performance reporting",
    "Pipeline reporting & forecasting",
    # ── Operations ────────────────────────────────────────────────────────────
    "Budget ownership & cost optimisation",
    "Root cause analysis",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skill Gap Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #f4f6f9; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { background-color: #0d1b2a; border-right: 1px solid #1e2f42; }
section[data-testid="stSidebar"] * { color: #8faabe !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
  background: #162232 !important; border: 1px solid #1e3347 !important;
  border-radius: 8px !important; color: #e2eaf2 !important; font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] .stButton > button {
  background: #0e7fc0 !important; color: #fff !important; border: none !important;
  border-radius: 8px !important; font-weight: 600 !important; width: 100% !important;
  padding: 0.5rem !important;
}
section[data-testid="stSidebar"] label { font-size: 0.72rem !important; color: #5a7a94 !important; text-transform: uppercase; letter-spacing: 0.05em; }
.phase-pill { display:inline-flex; align-items:center; background:#1e3347; color:#8faabe; border-radius:20px; padding:4px 14px; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem; }
.phase-pill.active { background:#0e7fc0; color:#fff; }
.phase-title { font-size:1.3rem; font-weight:700; color:#0d1b2a; margin:0 0 0.1rem; }
.phase-sub { font-size:0.83rem; color:#64748b; margin:0 0 1rem; }
.card { background:#fff; border:1px solid #e4eaf2; border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:1rem; box-shadow:0 2px 8px rgba(13,27,42,0.04); }
.card-sm { background:#fff; border:1px solid #e4eaf2; border-radius:10px; padding:1rem 1.25rem; }
.slabel { font-size:0.7rem; font-weight:700; color:#8faabe; text-transform:uppercase; letter-spacing:0.07em; margin:1rem 0 0.4rem; }
.skill-row { display:flex; align-items:center; padding:0.48rem 0; border-bottom:1px solid #f0f4f8; }
.skill-name { font-size:0.82rem; color:#1e3347; font-weight:500; width:230px; flex-shrink:0; line-height:1.3; }
.bar-wrap { flex:1; margin:0 0.8rem; height:6px; background:#eef2f7; border-radius:10px; overflow:hidden; }
.bar-have { height:100%; background:#10b981; border-radius:10px; }
.bar-miss { height:100%; background:#f43f5e; border-radius:10px; }
.bar-pct  { font-size:0.72rem; color:#8faabe; width:28px; text-align:right; font-family:'DM Mono',monospace; }
.tag-have { display:inline-block; background:#d1fae5; color:#065f46; border-radius:20px; padding:3px 12px; font-size:0.77rem; font-weight:500; margin:3px; }
.tag-miss { display:inline-block; background:#ffe4e6; color:#9f1239; border-radius:20px; padding:3px 12px; font-size:0.77rem; font-weight:500; margin:3px; }
.insight-box { background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px; padding:1rem 1.25rem; margin:0.5rem 0 1rem; font-size:0.85rem; color:#0c4a6e; line-height:1.6; }
.tool-badge { display:inline-flex; align-items:center; gap:5px; background:#1e3347; color:#7fb3d3; border-radius:6px; padding:3px 10px; font-size:0.72rem; font-weight:600; margin:3px; font-family:'DM Mono',monospace; }
hr { border:none; border-top:1px solid #e4eaf2; margin:0.8rem 0; }
.phase-divider { border:none; border-top:2px dashed #c8d8e8; margin:2.5rem 0; opacity:0.5; }
[data-testid="stMetricValue"] { font-size:1.5rem !important; font-weight:700 !important; color:#0d1b2a !important; }
[data-testid="stMetricLabel"] { font-size:0.68rem !important; color:#8faabe !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.05em !important; }
.stTextArea textarea { border:1.5px solid #e4eaf2 !important; border-radius:10px !important; font-size:0.85rem !important; }
.stCheckbox label { font-size:0.83rem !important; color:#334155 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LLM FUNCTIONS  (all Gemini-powered, all non-straightforward)
# ══════════════════════════════════════════════════════════════════════════════

def extract_skills_with_ai(cv_text: str, skill_list: list) -> list:
    """
    2-Call CV Extraction Pipeline:
    Call 1 — LLM reads raw CV text and extracts mentioned skills as free text.
             This avoids rigid keyword matching — LLM understands context.
    Call 2 — LLM normalizes extracted skills to canonical names from CURATED_ROLES.
             This maps "data wrangling with Pandas" → "Python — pandas & numpy",
             ensuring LLM output feeds cleanly into the Python gap_score engine.
    """
    # ── Call 1: Extract ───────────────────────────────────────────────────────
    raw_text = simple_call(
        f"""Read this CV and list every technical skill, tool, framework, methodology,
and domain capability the candidate demonstrably has.
Be specific — capture tool names and techniques (e.g. "PyTorch", "SQL CTEs", "DCF modelling").
Do NOT list generic soft skills like 'communication' or 'teamwork'.

CV:
{cv_text[:4000]}

List skills (one per line, no bullets, no numbering):"""
    )
    raw_skills = [s.strip() for s in raw_text.strip().splitlines() if s.strip()]

    # ── Call 2: Normalize to canonical names ─────────────────────────────────
    normalized = json_call(
        f"""Match each extracted CV skill to the closest entry in the canonical skill list below.
Only include definitive matches (semantic similarity counts — e.g. "Pandas data wrangling"
matches "Python — pandas & numpy"). Return ONLY the exact canonical names as a JSON array.

Extracted CV skills: {json.dumps(raw_skills)}

Canonical skill list: {json.dumps(skill_list)}

Return JSON array of matched canonical names only (no explanations):
["canonical skill name 1", "canonical skill name 2", ...]"""
    )
    return normalized if isinstance(normalized, list) else []


def analyze_jd(job_desc: str, user_skills: list) -> dict:
    """
    2-Call JD Analysis Pipeline:
    Call 1 — Extract structured role requirements from free-text job description.
             Separates required vs nice-to-have, infers seniority and domain.
    Call 2 — Score candidate against structured requirements.
             Call 1 output provides cleaner input → more accurate scoring.
    """
    # ── Call 1: Extract structured requirements ───────────────────────────────
    requirements = json_call(
        f"""Extract structured requirements from this job description.

Job Description:
{job_desc[:3000]}

Return JSON:
{{
  "job_title": "inferred title",
  "company": "company name or Unknown",
  "required_skills": [{{"skill": "name", "importance": 90, "learn_hrs": 15}}],
  "nice_to_have": [{{"skill": "name", "importance": 60, "learn_hrs": 10}}],
  "seniority": "junior|mid|senior",
  "domain": "one-word domain"
}}"""
    )

    # Merge required + nice-to-have with importance weighting
    all_jd_skills = requirements.get("required_skills", []) + [
        {**s, "importance": max(40, int(s.get("importance", 60) * 0.7))}
        for s in requirements.get("nice_to_have", [])
    ]

    # ── Call 2: Score candidate ───────────────────────────────────────────────
    analysis = json_call(
        f"""Score this candidate against the job requirements. Be specific to THIS candidate's fit.

Job: {requirements.get("job_title", "Unknown")} at {requirements.get("company", "Unknown")}
Seniority level: {requirements.get("seniority", "mid")}
All required skills: {json.dumps(all_jd_skills)}
Candidate's skills: {json.dumps(user_skills)}

Return JSON:
{{
  "fit_score": 72,
  "summary": "2-3 sentences specific to this candidate's fit for this role",
  "skills_have": ["exact skill names the candidate has from the JD"],
  "skills_missing": [
    {{"skill": "name", "importance": 85, "learn_hrs": 12, "confidence": "high", "reason": "one sentence why this matters for the specific role"}}
  ]
}}"""
    )

    return {
        "job_title":      requirements.get("job_title", analysis.get("job_title", "Unknown")),
        "company":        requirements.get("company",   analysis.get("company",   "Unknown")),
        "fit_score":      analysis.get("fit_score", 0),
        "summary":        analysis.get("summary", ""),
        "skills_have":    analysis.get("skills_have", []),
        "skills_missing": analysis.get("skills_missing", []),
    }


def generate_3call_roadmap(
    missing_skills: list,
    role_name: str,
    current_score: float,
    hours_per_week: int = 10,
) -> dict:
    """
    3-Call AI Roadmap Pipeline:
    Call 1 — Strategic leverage analysis: rank gaps by strategic value,
             identify quick wins and foundation skills.
    Call 2 — Generate week-by-week JSON plan using Call 1 output.
             (Call 1 output is fed as context — chained prompts.)
    Call 3 — Self-critique and refinement: checks prerequisite ordering,
             realism, and sequencing. Returns improved plan + diff of changes.
    Output of Call 3 drives a Plotly Gantt chart visualization.
    """
    total_hrs  = sum(s.get("learn_hrs", 10) for s in missing_skills)
    num_weeks  = max(4, min(16, round(total_hrs / hours_per_week)))

    # ── Call 1: Strategic Leverage Analysis ──────────────────────────────────
    leverage = json_call(
        f"""You are a career strategist. Analyze these missing skills for someone targeting
{role_name} (current readiness: {current_score}%). Rank them by strategic leverage.

Missing skills:
{json.dumps(missing_skills, indent=2)}

Return JSON:
{{
  "ranked_skills": [
    {{"skill": "exact skill name", "strategic_rank": 1, "leverage_score": 88, "rationale": "one sentence"}}
  ],
  "quick_wins": ["skill names learnable in under 10 hours that unblock other skills"],
  "foundation_skills": ["skills other things depend on — do these first"],
  "recommended_order": ["skill name in recommended learning order"]
}}"""
    )

    # ── Call 2: Week-by-Week Draft Plan ───────────────────────────────────────
    draft = json_call(
        f"""Create a week-by-week learning plan for {role_name} using this strategic analysis.

Strategic analysis (use this to sequence skills correctly):
{json.dumps(leverage, indent=2)}

All missing skills with estimated hours:
{json.dumps(missing_skills, indent=2)}

Plan for {num_weeks} weeks at {hours_per_week} hours/week.

Return JSON:
{{
  "total_weeks": {num_weeks},
  "hours_per_week": {hours_per_week},
  "weeks": [
    {{
      "week": 1,
      "theme": "theme name",
      "skills": ["skill name"],
      "hours": {hours_per_week},
      "milestone": "what you can concretely do or build by end of this week"
    }}
  ]
}}"""
    )

    # ── Call 3: Self-Critique & Refinement ────────────────────────────────────
    refined = json_call(
        f"""Review and improve this learning plan for {role_name}. Fix any issues.

Draft plan:
{json.dumps(draft, indent=2)}

Check and fix:
1. Prerequisites ordered correctly? (e.g. Python basics before ML algorithms)
2. Week 1-2 achievable? Early momentum matters.
3. Skills grouped efficiently? Related skills in same week = faster learning.
4. Time estimates realistic for someone new to each topic?
5. Does the sequence match the strategic leverage analysis below?

Strategic leverage context:
{json.dumps(leverage.get("recommended_order", []))}

Return the IMPROVED plan with the SAME structure, plus:
{{
  ...same weeks structure...,
  "improvements_made": ["bullet: what you changed and why"],
  "confidence": "high|medium|low",
  "final_message": "One encouraging sentence for the learner"
}}"""
    )

    return {
        "leverage":      leverage,
        "draft":         draft,
        "refined":       refined,
        "num_weeks":     num_weeks,
        "hours_per_week": hours_per_week,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PURE PYTHON UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_roadmap(missing_skills: list, current_score: float, total_importance: float,
                   use_ai_roadmap: bool = False, role_name: str = ""):
    """Renders the full Part B roadmap section. Handles both standard and AI modes."""
    if not missing_skills:
        st.success("No gaps found — you are a strong match for this role.")
        return

    normalised = []
    for s in missing_skills:
        normalised.append({
            "skill":      s.get("skill",      s.get("Skill",     "Unknown")),
            "importance": float(s.get("importance", s.get("Importance", 70))),
            "learn_hrs":  float(s.get("learn_hrs",  s.get("LearnHrs",   10))),
            "confidence": s.get("confidence", "medium"),
            "reason":     s.get("reason", ""),
        })

    rdf = pd.DataFrame(normalised)
    rdf["roi"] = rdf.apply(lambda r: roi_score(r["importance"], r["learn_hrs"]), axis=1)
    rdf = rdf.sort_values("roi", ascending=False).reset_index(drop=True)

    # ── Time budget slider ────────────────────────────────────────────────────
    total_hrs_all = int(rdf["learn_hrs"].sum())
    st.markdown("<p class='slabel'>Your available learning time</p>", unsafe_allow_html=True)
    time_budget = st.slider(
        "Hours available",
        min_value=2, max_value=max(total_hrs_all, 10),
        value=min(20, total_hrs_all), step=1, format="%dh",
        label_visibility="collapsed",
    )

    rdf_sorted = rdf.sort_values("roi", ascending=False).copy()
    hours_used, in_budget = 0, []
    for _, row in rdf_sorted.iterrows():
        if hours_used + row["learn_hrs"] <= time_budget:
            in_budget.append(row["skill"])
            hours_used += row["learn_hrs"]

    gained_imp     = rdf[rdf["skill"].isin(in_budget)]["importance"].sum()
    projected      = min(100, round(current_score + (gained_imp / total_importance * 100), 1))
    gain           = round(projected - current_score, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Fit", f"{current_score}%")
    c2.metric(f"Projected in {time_budget}h", f"{projected}%",
              delta=f"+{gain}%" if gain > 0 else "No change — try more hours")
    c3.metric("Skills in budget", f"{len(in_budget)} of {len(rdf)}")

    if in_budget:
        st.markdown(
            f"<div class='insight-box'>In <strong>{time_budget} hours</strong>, focus on: "
            + ", ".join(f"<strong>{s}</strong>" for s in in_budget)
            + f". This lifts your fit from <strong>{current_score}%</strong> to <strong>{projected}%</strong>.</div>",
            unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── AI-Enhanced Roadmap (opt-in 3-call pipeline) ──────────────────────────
    if use_ai_roadmap and role_name:
        ai_key = f"ai_roadmap_{role_name.replace(' ', '_')}"
        if ai_key not in st.session_state:
            st.session_state[ai_key] = None

        if st.session_state[ai_key] is None:
            hrs_pw = st.number_input(
                "Hours per week you can commit",
                min_value=2, max_value=40, value=10, step=1,
                key=f"hrs_pw_{role_name}",
            )
            if st.button("✨ Generate AI Roadmap", key=f"gen_{role_name}", type="primary"):
                with st.status("Building your AI-enhanced roadmap...", expanded=True) as status:
                    st.write("🔍 Step 1 / 3 — Analysing strategic leverage...")
                    try:
                        result = generate_3call_roadmap(
                            missing_skills=normalised,
                            role_name=role_name,
                            current_score=current_score,
                            hours_per_week=hrs_pw,
                        )
                        # Fake brief pauses so status labels are visible
                        import time
                        st.write("📅 Step 2 / 3 — Building week-by-week plan...")
                        time.sleep(0.3)
                        st.write("✨ Step 3 / 3 — Refining for realism & sequencing...")
                        time.sleep(0.3)
                        st.session_state[ai_key] = result
                        status.update(label="Roadmap complete!", state="complete")
                        st.rerun()
                    except Exception as e:
                        status.update(label=f"Error: {e}", state="error")
        else:
            result = st.session_state[ai_key]
            _render_ai_roadmap_output(result, role_name, in_budget, current_score)
            if st.button("Regenerate", key=f"regen_{role_name}"):
                st.session_state.pop(ai_key, None)
                st.rerun()
    else:
        # ── Standard ROI chart ────────────────────────────────────────────────
        _render_standard_roadmap_charts(rdf, in_budget)

    # ── Per-skill course recommendations ─────────────────────────────────────
    st.markdown("<p class='slabel'>Recommended courses — sorted by ROI</p>", unsafe_allow_html=True)
    for _, row in rdf.iterrows():
        conf       = row["confidence"].lower() if isinstance(row["confidence"], str) else "medium"
        is_budget  = row["skill"] in in_budget
        conf_color = {"high": "#065f46", "medium": "#854d0e"}.get(conf, "#9f1239")
        conf_bg    = {"high": "#d1fae5", "medium": "#fef9c3"}.get(conf, "#ffe4e6")
        budget_tag = ('<span style="background:#dbeafe;color:#1d4ed8;border-radius:6px;'
                      'padding:2px 9px;font-size:0.7rem;font-weight:700;margin-left:6px">In budget</span>'
                      if is_budget else "")
        with st.expander(f"{'★ ' if is_budget else ''}{row['skill']}"):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:0.6rem">
              <span style="background:{conf_bg};color:{conf_color};border-radius:6px;padding:2px 9px;font-size:0.71rem;font-weight:600">{conf.capitalize()} confidence</span>
              <span style="background:#f1f5f9;color:#475569;border-radius:6px;padding:2px 9px;font-size:0.71rem;font-weight:600">~{int(row['learn_hrs'])}h to learn</span>
              <span style="background:#f1f5f9;color:#475569;border-radius:6px;padding:2px 9px;font-size:0.71rem;font-weight:600">{row['importance']:.0f} importance</span>
              <span style="background:#f1f5f9;color:#475569;border-radius:6px;padding:2px 9px;font-size:0.71rem;font-weight:600">ROI {row['roi']:.1f}</span>
              {budget_tag}
            </div>""", unsafe_allow_html=True)
            if row["reason"]:
                st.markdown(f"<div class='insight-box'>{row['reason']}</div>", unsafe_allow_html=True)
            courses = get_courses(row["skill"])
            if courses:
                st.markdown("<p style='font-size:0.72rem;color:#8faabe;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:0.6rem 0 0.4rem'>Recommended courses</p>", unsafe_allow_html=True)
                cards_html = '<div style="display:flex;flex-direction:column;gap:8px">'
                for c in courses:
                    cards_html += f"""
                    <a href="{c['url']}" target="_blank" style="text-decoration:none">
                      <div style="display:flex;align-items:center;gap:12px;background:#f8fafc;border:1px solid #e4eaf2;border-radius:10px;padding:10px 14px">
                        <div style="width:10px;height:10px;border-radius:50%;background:{c['color']};flex-shrink:0"></div>
                        <div style="flex:1">
                          <p style="margin:0;font-size:0.82rem;font-weight:600;color:#0d1b2a">{c['course']}</p>
                          <p style="margin:0;font-size:0.74rem;color:#64748b">{c['platform']}</p>
                        </div>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                      </div>
                    </a>"""
                cards_html += "</div>"
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.markdown(f"""<a href="https://www.linkedin.com/learning/search?keywords={row['skill'].replace(' ','+')}" target="_blank" style="text-decoration:none">
                  <div style="display:flex;align-items:center;gap:12px;background:#f8fafc;border:1px solid #e4eaf2;border-radius:10px;padding:10px 14px">
                    <div style="width:10px;height:10px;border-radius:50%;background:#0077b5;flex-shrink:0"></div>
                    <div style="flex:1"><p style="margin:0;font-size:0.82rem;font-weight:600;color:#0d1b2a">Search LinkedIn Learning</p><p style="margin:0;font-size:0.74rem;color:#64748b">LinkedIn Learning</p></div>
                  </div></a>""", unsafe_allow_html=True)


def _render_standard_roadmap_charts(rdf, in_budget):
    st.markdown("<p class='slabel'>Learning ROI — importance per hour (do these first)</p>", unsafe_allow_html=True)
    n = len(rdf)
    bar_colors = [f"rgba(14,127,192,{0.35 + 0.65*(i/(max(n-1,1)))})" for i in range(n-1, -1, -1)]
    fig_roi = go.Figure(go.Bar(
        x=rdf["roi"], y=rdf["skill"], orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}" for v in rdf["roi"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>ROI: %{x:.2f}<br>Importance: %{customdata[0]:.0f}<br>Hours: %{customdata[1]:.0f}h<extra></extra>",
        customdata=rdf[["importance", "learn_hrs"]].values,
    ))
    fig_roi.update_layout(
        height=max(260, n * 40), margin=dict(l=0, r=70, t=10, b=30),
        xaxis=dict(title="ROI (importance ÷ hours)", showgrid=True, gridcolor="#f0f4f8", zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, family="DM Sans")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11),
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("<p class='slabel'>Effort vs impact — aim for top-left (high impact, low effort)</p>", unsafe_allow_html=True)
    fig_bubble = go.Figure(go.Scatter(
        x=rdf["learn_hrs"], y=rdf["importance"], mode="markers+text",
        text=rdf["skill"].apply(lambda s: s[:24]+"…" if len(s) > 24 else s),
        textposition="top center", textfont=dict(size=9, family="DM Sans"),
        marker=dict(
            size=rdf["roi"] * 7,
            color=rdf["roi"],
            colorscale=[[0, "#bfdbfe"], [0.5, "#3b82f6"], [1, "#1d4ed8"]],
            showscale=True,
            colorbar=dict(title="ROI", thickness=10, len=0.5, tickfont=dict(size=9)),
            line=dict(width=1, color="white"),
        ),
        hovertemplate="<b>%{text}</b><br>Importance: %{y:.0f}<br>Hours: %{x:.0f}h<br>ROI: %{marker.color:.2f}<extra></extra>",
    ))
    fig_bubble.update_layout(
        height=330, margin=dict(l=0, r=20, t=20, b=40),
        xaxis=dict(title="Estimated hours to learn", showgrid=True, gridcolor="#f0f4f8"),
        yaxis=dict(title="Importance to role", showgrid=True, gridcolor="#f0f4f8"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)


def _render_ai_roadmap_output(result: dict, role_name: str, in_budget: list, current_score: float):
    """Render the 3-call AI roadmap output: insights, Gantt chart, and diff."""
    refined  = result.get("refined", {})
    leverage = result.get("leverage", {})
    draft    = result.get("draft", {})

    # ── Call 1 insights ───────────────────────────────────────────────────────
    quick_wins   = leverage.get("quick_wins", [])
    foundation   = leverage.get("foundation_skills", [])
    if quick_wins or foundation:
        qw  = ", ".join(f"<strong>{s}</strong>" for s in quick_wins[:3])
        fnd = ", ".join(f"<strong>{s}</strong>" for s in foundation[:3])
        st.markdown(
            f"<div class='insight-box'>"
            f"{'⚡ <strong>Quick wins:</strong> ' + qw + '<br>' if qw else ''}"
            f"{'🏗️ <strong>Foundation first:</strong> ' + fnd if fnd else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Gantt chart (from Call 3 refined output) ──────────────────────────────
    weeks = refined.get("weeks", draft.get("weeks", []))
    if weeks:
        st.markdown("<p class='slabel'>Week-by-week learning timeline (AI-refined)</p>", unsafe_allow_html=True)
        palette = ["#0e7fc0", "#10b981", "#f59e0b", "#8b5cf6",
                   "#f43f5e", "#06b6d4", "#84cc16", "#fb7185",
                   "#0ea5e9", "#a78bfa", "#34d399", "#fbbf24"]
        fig = go.Figure()
        for i, w in enumerate(weeks):
            skills_label = " + ".join(w.get("skills", [w.get("focus", "Learning")])) \
                if isinstance(w.get("skills"), list) else w.get("focus", "Learning")
            label = f"Wk {w['week']}: {skills_label[:40]}{'…' if len(skills_label) > 40 else ''}"
            milestone = w.get("milestone", "")
            fig.add_trace(go.Bar(
                name=label,
                y=[label],
                x=[w.get("hours", result.get("hours_per_week", 10))],
                orientation="h",
                marker_color=palette[i % len(palette)],
                text=[milestone[:55] + "…" if len(milestone) > 55 else milestone],
                textposition="inside", insidetextanchor="start",
                hovertemplate=(
                    f"<b>Week {w['week']}</b><br>"
                    f"Theme: {w.get('theme', '')}<br>"
                    f"Hours: {w.get('hours', 0)}<br>"
                    f"Milestone: {milestone}<extra></extra>"
                ),
            ))
        fig.update_layout(
            barmode="stack",
            height=max(300, len(weeks) * 48),
            xaxis_title="Hours",
            yaxis=dict(autorange="reversed", tickfont=dict(size=10, family="DM Sans")),
            margin=dict(l=0, r=20, t=35, b=40),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=10),
            title=dict(text=f"Learning Timeline — {role_name}", font=dict(size=13, color="#0d1b2a")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Call 3 improvements diff ──────────────────────────────────────────────
    improvements = refined.get("improvements_made", [])
    if improvements:
        st.markdown("<p class='slabel'>What the AI refined from the draft plan</p>", unsafe_allow_html=True)
        items_html = "".join(f"<li style='margin-bottom:0.3rem'>{imp}</li>" for imp in improvements)
        st.markdown(
            f"<div class='card-sm'><ul style='margin:0;padding-left:1.2rem;font-size:0.83rem;color:#334155;line-height:1.6'>{items_html}</ul></div>",
            unsafe_allow_html=True,
        )

    # ── Final message ─────────────────────────────────────────────────────────
    final_msg = refined.get("final_message", "")
    if final_msg:
        st.markdown(f"<div class='insight-box'>{final_msg}</div>", unsafe_allow_html=True)


def make_radar_chart(radar_data: dict) -> go.Figure:
    """Build a Plotly radar chart from get_skill_radar_data output."""
    cats   = radar_data["categories"] + [radar_data["categories"][0]]
    r_vals = radar_data["role_target"] + [radar_data["role_target"][0]]
    u_vals = radar_data["user_current"] + [radar_data["user_current"][0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_vals, theta=cats, fill="toself",
        name=radar_data["role"],
        line_color="rgba(14,127,192,0.9)",
        fillcolor="rgba(14,127,192,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=u_vals, theta=cats, fill="toself",
        name="You",
        line_color="rgba(16,185,129,0.95)",
        fillcolor="rgba(16,185,129,0.22)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=360,
        margin=dict(l=40, r=40, t=40, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=10),
        legend=dict(orientation="h", y=-0.2),
        title=dict(
            text=f"Skill coverage — {radar_data['role']} ({radar_data['coverage_pct']}%)",
            font=dict(size=12, color="#0d1b2a"),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [
    ("analysis_done",   False),
    ("analysis_result", {}),
    ("advisor_messages", []),
    ("advisor_chat",    None),
    ("advisor_tool_trace", []),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CV UPLOAD  (LLM Feature 1: 2-call extraction pipeline)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Skill Gap Analyzer")
    st.markdown(
        "<p style='font-size:0.73rem;color:#4a6a84;margin-top:-0.3rem'>ESADE MBAn · Prototyping Assignment</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem;font-weight:700;color:#4a6a84;text-transform:uppercase;letter-spacing:0.06em'>CV Extraction</p>",
        unsafe_allow_html=True,
    )

    # ── Auto-load demo profile on first visit ──────────────────────────────
    if "cv_extracted" not in st.session_state:
        st.session_state["persistent_skills"] = DEMO_SKILLS
        st.session_state["cv_extracted"] = True
        st.session_state["demo_cv_loaded"] = True
        st.rerun()

    # ── Sidebar state display ──────────────────────────────────────────────
    n = len(st.session_state.get("persistent_skills", []))
    if st.session_state.get("demo_cv_loaded"):
        st.markdown(
            f"<p style='font-size:0.82rem;color:#10b981'>✓ Demo profile loaded ({n} skills)</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.72rem;color:#4a6a84;margin-top:-0.3rem'>Sean Hoet — ESADE MBAn</p>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use my own CV"):
                for k in ["cv_extracted", "persistent_skills", "demo_cv_loaded"]:
                    st.session_state.pop(k, None)
                st.rerun()
    else:
        st.markdown(
            f"<p style='font-size:0.82rem;color:#10b981'>{n} skills loaded from CV</p>",
            unsafe_allow_html=True,
        )
        if st.button("Clear & re-upload CV"):
            for k in ["cv_extracted", "persistent_skills", "demo_cv_loaded"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Show CV upload form when demo cleared ─────────────────────────────
    if "cv_extracted" not in st.session_state:
        cv_file = st.file_uploader("Upload CV (.txt)", type=["txt"], label_visibility="collapsed")
        if cv_file:
            cv_text = cv_file.read().decode("utf-8")
            if st.button("Extract skills from CV"):
                all_flat = list({s["skill"] for r in CURATED_ROLES.values() for s in r})
                with st.spinner("Reading CV (2-step AI extraction)..."):
                    try:
                        found = extract_skills_with_ai(cv_text, all_flat)
                        st.session_state["persistent_skills"] = found
                        st.session_state["cv_extracted"] = True
                        st.success(f"Saved {len(found)} skills")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.68rem;color:#3a5a74;line-height:1.5'>Courses: DataCamp, Google, LinkedIn Learning, Coursera, AWS, HubSpot, Salesforce.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:white;border-bottom:1px solid #e4eaf2;padding:1rem 1.5rem;margin:-1rem -1rem 1.5rem -1rem;">
  <h1 style="font-size:1.3rem;font-weight:700;color:#0d1b2a;margin:0">Skill Gap Analyzer</h1>
  <p style="font-size:0.82rem;color:#64748b;margin:0.1rem 0 0">Identify your gaps · Get a prioritised roadmap · Chat with your AI career advisor.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_gap, tab_advisor = st.tabs(["📊 Gap Analyzer", "🤖 Career Advisor"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GAP ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
with tab_gap:
    st.markdown("<div class='phase-pill active'>Part A — Identify the Gap</div>", unsafe_allow_html=True)
    st.markdown("<p class='phase-title'>Where do you stand?</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='phase-sub'>Compare against a role from our database, or paste a specific job description.</p>",
        unsafe_allow_html=True,
    )

    method = st.radio(
        "", ["Role database", "Paste a job description"],
        horizontal=True, label_visibility="collapsed",
    )

    # ── Method A: Role database ───────────────────────────────────────────────
    if method == "Role database":
        all_roles = list(CURATED_ROLES.keys())
        col_s, col_sel = st.columns([1, 2])
        with col_s:
            role_search = st.text_input("", placeholder="Filter roles...", label_visibility="collapsed")
        filtered = [r for r in all_roles if role_search.lower() in r.lower()] if role_search.strip() else all_roles
        with col_sel:
            selected_role = st.selectbox("", filtered, label_visibility="collapsed")

        role_skills    = CURATED_ROLES[selected_role]
        all_skill_names = [s["skill"] for s in role_skills]
        default_checked = [s for s in st.session_state.get("persistent_skills", []) if s in all_skill_names]

        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown(f"<div class='card'><p class='slabel'>Your skills for {selected_role}</p>", unsafe_allow_html=True)
            if default_checked:
                st.markdown(
                    f"<p style='font-size:0.75rem;color:#0e7fc0;margin:0 0 0.5rem'>{len(default_checked)} pre-filled from CV</p>",
                    unsafe_allow_html=True,
                )
            user_skills = []
            for skill in all_skill_names:
                if st.checkbox(skill, value=(skill in default_checked), key=f"r_{selected_role}_{skill}"):
                    user_skills.append(skill)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            df     = compute_gap(role_skills, user_skills)
            score  = gap_score(df)
            total_imp = df["Importance"].sum()
            n_have = int(df["Have"].sum())
            n_miss = int((~df["Have"]).sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Readiness", f"{score}%")
            m2.metric("Matched",   f"{n_have}/{len(df)}")
            m3.metric("Gaps",      str(n_miss))

            st.markdown("<hr>", unsafe_allow_html=True)
            for _, row in df.iterrows():
                bc = "bar-have" if row["Have"] else "bar-miss"
                st.markdown(
                    f"""<div class="skill-row">
                      <span class="skill-name">{row['Skill']}</span>
                      <div class="bar-wrap"><div class="{bc}" style="width:{row['Importance']:.0f}%"></div></div>
                      <span class="bar-pct">{row['Importance']:.0f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("<hr>", unsafe_allow_html=True)
            st.progress(int(score))

            missing_df = df[~df["Have"]]
            if st.button("Build my learning roadmap →", type="primary"):
                st.session_state["analysis_done"] = True
                st.session_state["analysis_result"] = {
                    "type":             "role",
                    "title":            selected_role,
                    "company":          "",
                    "fit_score":        score,
                    "total_importance": total_imp,
                    "missing": missing_df[["Skill", "Importance", "LearnHrs"]].rename(
                        columns={"Skill": "skill", "Importance": "importance", "LearnHrs": "learn_hrs"}
                    ).to_dict("records"),
                    "have":    user_skills,
                    "summary": f"You match {n_have} of {len(df)} skills for {selected_role}, readiness {score}%.",
                }
                st.rerun()

    # ── Method B: Paste a job description (LLM Feature 2: 2-call pipeline) ───
    else:
        col_jd, col_my = st.columns(2)
        with col_jd:
            st.markdown("<p class='slabel'>Job description</p>", unsafe_allow_html=True)
            job_description = st.text_area(
                "", height=240, label_visibility="collapsed",
                placeholder="Paste the full LinkedIn or company job posting here...",
            )
        with col_my:
            st.markdown("<p class='slabel'>Your skills (one per line)</p>", unsafe_allow_html=True)
            prefill = "\n".join(st.session_state.get("persistent_skills", []))
            my_skills_raw = st.text_area(
                "", value=prefill, height=240, label_visibility="collapsed",
                placeholder="Python\nSQL\nStakeholder management\n...",
            )

        if st.button("Analyze my fit", type="primary"):
            if not job_description.strip():
                st.warning("Paste a job description first.")
            else:
                my_skills = [s.strip() for s in my_skills_raw.strip().splitlines() if s.strip()]
                with st.spinner("Analyzing role fit (2-step AI analysis)..."):
                    try:
                        res = analyze_jd(job_description, my_skills)
                        missing_raw = res.get("skills_missing", [])
                        missing_clean = [
                            s if isinstance(s, dict)
                            else {"skill": str(s), "importance": 75, "learn_hrs": 10, "confidence": "medium", "reason": ""}
                            for s in missing_raw
                        ]
                        total_imp = sum(s["importance"] for s in missing_clean) + \
                                    sum(75 for _ in res.get("skills_have", []))
                        st.session_state["analysis_done"]   = True
                        st.session_state["analysis_result"] = {
                            "type":             "jd",
                            "title":            res.get("job_title", ""),
                            "company":          res.get("company", ""),
                            "fit_score":        res.get("fit_score", 0),
                            "total_importance": max(total_imp, 1),
                            "missing":          missing_clean,
                            "have":             res.get("skills_have", []),
                            "summary":          res.get("summary", ""),
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        if st.session_state.get("analysis_done") and st.session_state["analysis_result"].get("type") == "jd":
            res = st.session_state["analysis_result"]
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='card-sm'><p style='font-size:1.1rem;font-weight:700;color:#0d1b2a;margin:0'>{res['title']}</p>"
                f"<p style='font-size:0.82rem;color:#64748b;margin:0.1rem 0 0'>{res['company']}</p></div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Fit Score",       f"{res['fit_score']}%")
            m2.metric("Skills Matched",  str(len(res["have"])))
            m3.metric("Gaps Identified", str(len(res["missing"])))
            st.progress(int(min(res["fit_score"], 100)))
            st.markdown(
                f"<div class='card'><p style='font-size:0.87rem;color:#334155;line-height:1.65;margin:0'>{res['summary']}</p></div>",
                unsafe_allow_html=True,
            )
            col_h, col_m = st.columns(2)
            with col_h:
                st.markdown("<p class='slabel'>What you already have</p>", unsafe_allow_html=True)
                tags = "".join(f'<span class="tag-have">{s}</span>' for s in res["have"]) \
                       or "<p style='color:#94a3b8;font-size:0.82rem'>Add skills above.</p>"
                st.markdown(f"<div class='card'>{tags}</div>", unsafe_allow_html=True)
            with col_m:
                st.markdown("<p class='slabel'>Gaps to address</p>", unsafe_allow_html=True)
                miss_names = [s["skill"] if isinstance(s, dict) else s for s in res["missing"]]
                tags = "".join(f'<span class="tag-miss">{s}</span>' for s in miss_names) \
                       or "<p style='color:#10b981;font-size:0.82rem'>No gaps found.</p>"
                st.markdown(f"<div class='card'>{tags}</div>", unsafe_allow_html=True)

    # ── PART B — Bridge the Gap ───────────────────────────────────────────────
    if st.session_state.get("analysis_done") and st.session_state["analysis_result"].get("missing"):
        res = st.session_state["analysis_result"]

        st.markdown("<hr class='phase-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='phase-pill active'>Part B — Bridge the Gap</div>", unsafe_allow_html=True)
        st.markdown("<p class='phase-title'>Your Learning Roadmap</p>", unsafe_allow_html=True)
        title_str = res["title"] + (
            f" at {res['company']}"
            if res.get("company") and res["company"] not in ["", "Unknown"] else ""
        )
        st.markdown(
            f"<p class='phase-sub'>Personalised to close your gaps for <strong>{title_str}</strong>. "
            "Sorted by ROI — which skills give you the most value per hour.</p>",
            unsafe_allow_html=True,
        )

        # ── AI roadmap toggle (opt-in, LLM Feature 4) ────────────────────────
        use_ai = st.toggle(
            "✨ AI-Enhanced Roadmap (3-step refinement pipeline)",
            value=False,
            help="Runs 3 sequential Gemini calls: strategic leverage analysis → week-by-week plan → self-critique & refinement. Produces a Gantt timeline.",
        )

        render_roadmap(
            missing_skills    = res["missing"],
            current_score     = res["fit_score"],
            total_importance  = res.get("total_importance", 800),
            use_ai_roadmap    = use_ai,
            role_name         = res["title"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAREER ADVISOR  (LLM Feature 3: multi-call agent with tools)
# ══════════════════════════════════════════════════════════════════════════════
with tab_advisor:
    st.markdown("<div class='phase-pill active'>Career Advisor</div>", unsafe_allow_html=True)
    st.markdown("<p class='phase-title'>AI Career Advisor</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='phase-sub'>Ask anything about role fit, career transitions, skill gaps, or learning timelines. "
        "The advisor uses live Python tools to compute real answers from the role database.</p>",
        unsafe_allow_html=True,
    )

    user_skills_for_advisor = st.session_state.get("persistent_skills", [])

    col_chat, col_trace = st.columns([3, 1])

    with col_chat:
        # ── Display conversation ──────────────────────────────────────────────
        for msg in st.session_state["advisor_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("radar_data"):
                    fig = make_radar_chart(msg["radar_data"])
                    st.plotly_chart(fig, use_container_width=True, key=f"radar_{id(msg)}")

        # ── Chat input ────────────────────────────────────────────────────────
        if user_prompt := st.chat_input("Ask about your career path, role fit, transitions, or timelines..."):
            # Show user message immediately
            st.session_state["advisor_messages"].append({"role": "user", "content": user_prompt})

            # Init chat session if needed
            if st.session_state["advisor_chat"] is None:
                st.session_state["advisor_chat"] = create_chat_session(user_skills_for_advisor)

            with st.spinner("Thinking..."):
                try:
                    result = run_advisor_turn(
                        user_message=user_prompt,
                        chat_obj=st.session_state["advisor_chat"],
                    )
                    st.session_state["advisor_messages"].append({
                        "role":       "assistant",
                        "content":    result["response"],
                        "radar_data": result.get("radar_data"),
                    })
                    if result["tool_calls"]:
                        st.session_state["advisor_tool_trace"].extend(result["tool_calls"])
                except Exception as e:
                    st.session_state["advisor_messages"].append({
                        "role":    "assistant",
                        "content": f"⚠️ Something went wrong: {e}. Please try again.",
                    })

            st.rerun()

    with col_trace:
        st.markdown("<p class='slabel'>Tool trace</p>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.74rem;color:#8faabe;margin-bottom:0.5rem'>"
            "Python tools called by the LLM this session:</p>",
            unsafe_allow_html=True,
        )

        if st.session_state["advisor_tool_trace"]:
            # Show the last 10 calls, newest first
            for tc in reversed(st.session_state["advisor_tool_trace"][-10:]):
                with st.expander(f"🔧 {tc['tool']}"):
                    st.markdown(
                        f"<p style='font-size:0.73rem;color:#8faabe;margin:0 0 0.3rem'>Args</p>",
                        unsafe_allow_html=True,
                    )
                    st.json({k: v for k, v in tc["args"].items() if k != "user_skills"})
                    if tc.get("result"):
                        st.markdown(
                            f"<p style='font-size:0.73rem;color:#8faabe;margin:0.4rem 0 0.3rem'>Result preview</p>",
                            unsafe_allow_html=True,
                        )
                        result_preview = tc["result"]
                        if isinstance(result_preview, dict):
                            # Show only top-level keys to keep it readable
                            preview = {k: (v if not isinstance(v, list) or len(v) <= 3 else v[:3])
                                       for k, v in list(result_preview.items())[:6]}
                            st.json(preview)
        else:
            st.markdown(
                "<p style='font-size:0.79rem;color:#94a3b8'>Tool calls appear here as you chat.</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Reset conversation", key="reset_advisor"):
            for k in ["advisor_messages", "advisor_chat", "advisor_tool_trace", "_advisor_gemini_client"]:
                st.session_state.pop(k, None)
            st.rerun()

        # Show user skills loaded from CV
        if user_skills_for_advisor:
            st.markdown(
                f"<p style='font-size:0.73rem;color:#10b981;margin-top:0.5rem'>"
                f"✓ {len(user_skills_for_advisor)} skills from CV loaded into advisor context</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='font-size:0.73rem;color:#f59e0b;margin-top:0.5rem'>"
                "⚡ Upload your CV (sidebar) to pre-load your skills.</p>",
                unsafe_allow_html=True,
            )
