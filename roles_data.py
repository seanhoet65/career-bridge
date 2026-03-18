"""
roles_data.py
Shared data and pure-Python computation functions used by both app.py and agent_tools.py.
Keeping CURATED_ROLES as the source of truth — specific, meaningful skill descriptors
rather than generic O*NET data.
"""
import pandas as pd

# ── Curated role skill profiles ────────────────────────────────────────────────
CURATED_ROLES = {
    "Data Analyst": [
        {"skill": "SQL (queries, joins, CTEs)", "importance": 96, "learn_hrs": 20},
        {"skill": "Python — pandas & numpy", "importance": 88, "learn_hrs": 30},
        {"skill": "Tableau or Power BI", "importance": 82, "learn_hrs": 15},
        {"skill": "Excel & pivot tables", "importance": 78, "learn_hrs": 8},
        {"skill": "Statistical analysis & hypothesis testing", "importance": 80, "learn_hrs": 25},
        {"skill": "Data cleaning & wrangling", "importance": 84, "learn_hrs": 12},
        {"skill": "A/B testing", "importance": 70, "learn_hrs": 10},
        {"skill": "Dashboard design", "importance": 72, "learn_hrs": 10},
        {"skill": "Git version control", "importance": 60, "learn_hrs": 6},
        {"skill": "Stakeholder presentation of insights", "importance": 74, "learn_hrs": 8},
    ],
    "Data Scientist": [
        {"skill": "Python (scikit-learn, statsmodels)", "importance": 96, "learn_hrs": 30},
        {"skill": "Statistical modelling & inference", "importance": 92, "learn_hrs": 30},
        {"skill": "SQL & large dataset handling", "importance": 84, "learn_hrs": 20},
        {"skill": "ML — classification, regression, clustering", "importance": 90, "learn_hrs": 40},
        {"skill": "Data visualisation (matplotlib, Tableau)", "importance": 82, "learn_hrs": 15},
        {"skill": "Experiment design & A/B testing", "importance": 80, "learn_hrs": 12},
        {"skill": "NLP basics (text processing, embeddings)", "importance": 70, "learn_hrs": 20},
        {"skill": "Cloud data platforms (BigQuery, Databricks)", "importance": 74, "learn_hrs": 15},
        {"skill": "Git & reproducible research", "importance": 72, "learn_hrs": 6},
        {"skill": "Communicating findings to non-technical stakeholders", "importance": 80, "learn_hrs": 8},
    ],
    "Machine Learning Engineer": [
        {"skill": "Python (PyTorch or TensorFlow)", "importance": 97, "learn_hrs": 50},
        {"skill": "ML algorithms (supervised/unsupervised)", "importance": 95, "learn_hrs": 40},
        {"skill": "Feature engineering & selection", "importance": 88, "learn_hrs": 15},
        {"skill": "Model evaluation & validation", "importance": 87, "learn_hrs": 12},
        {"skill": "MLOps & model deployment (Docker, FastAPI)", "importance": 83, "learn_hrs": 25},
        {"skill": "Cloud ML platforms (SageMaker, Vertex AI)", "importance": 80, "learn_hrs": 20},
        {"skill": "SQL & data pipelines", "importance": 75, "learn_hrs": 20},
        {"skill": "Deep learning & neural networks", "importance": 85, "learn_hrs": 50},
        {"skill": "Experiment tracking (MLflow, W&B)", "importance": 70, "learn_hrs": 8},
        {"skill": "Git & CI/CD", "importance": 78, "learn_hrs": 10},
    ],
    "AI / ML Product Manager": [
        {"skill": "ML lifecycle (training, evaluation, deployment)", "importance": 88, "learn_hrs": 15},
        {"skill": "Data product roadmapping", "importance": 90, "learn_hrs": 10},
        {"skill": "Prompt engineering & LLM product experience", "importance": 82, "learn_hrs": 8},
        {"skill": "Defining AI success metrics & eval frameworks", "importance": 86, "learn_hrs": 10},
        {"skill": "Working with data scientists & ML engineers", "importance": 87, "learn_hrs": 5},
        {"skill": "SQL & basic data analysis", "importance": 72, "learn_hrs": 20},
        {"skill": "Agile with ML-specific sprints", "importance": 78, "learn_hrs": 6},
        {"skill": "Ethics, bias & responsible AI", "importance": 74, "learn_hrs": 8},
        {"skill": "Competitive analysis of AI products", "importance": 76, "learn_hrs": 6},
        {"skill": "Stakeholder education on AI capabilities", "importance": 82, "learn_hrs": 5},
    ],
    "Product Manager": [
        {"skill": "Product roadmap & prioritization (RICE, MoSCoW)", "importance": 93, "learn_hrs": 10},
        {"skill": "User research & customer interviews", "importance": 88, "learn_hrs": 8},
        {"skill": "Writing PRDs & user stories", "importance": 85, "learn_hrs": 8},
        {"skill": "Agile & sprint planning", "importance": 84, "learn_hrs": 6},
        {"skill": "Data analysis for product decisions", "importance": 80, "learn_hrs": 15},
        {"skill": "A/B testing & experimentation", "importance": 76, "learn_hrs": 10},
        {"skill": "SQL (basic)", "importance": 64, "learn_hrs": 20},
        {"skill": "Wireframing (Figma, Miro)", "importance": 72, "learn_hrs": 8},
        {"skill": "Stakeholder alignment & executive communication", "importance": 87, "learn_hrs": 6},
        {"skill": "Competitive & market analysis", "importance": 74, "learn_hrs": 6},
    ],
    "Business Analyst": [
        {"skill": "Requirements gathering & documentation", "importance": 93, "learn_hrs": 8},
        {"skill": "Process mapping (BPMN, swimlane diagrams)", "importance": 86, "learn_hrs": 10},
        {"skill": "SQL & data querying", "importance": 80, "learn_hrs": 20},
        {"skill": "Excel & financial modelling", "importance": 82, "learn_hrs": 12},
        {"skill": "Stakeholder workshops & facilitation", "importance": 87, "learn_hrs": 6},
        {"skill": "User acceptance testing (UAT)", "importance": 74, "learn_hrs": 6},
        {"skill": "JIRA / Confluence", "importance": 68, "learn_hrs": 4},
        {"skill": "Business case writing", "importance": 78, "learn_hrs": 6},
        {"skill": "KPI definition & reporting", "importance": 80, "learn_hrs": 8},
        {"skill": "Change management basics", "importance": 65, "learn_hrs": 8},
    ],
    "Strategy Consultant": [
        {"skill": "Structured problem solving (MECE, issue trees)", "importance": 95, "learn_hrs": 12},
        {"skill": "Financial modelling (DCF, scenario analysis)", "importance": 88, "learn_hrs": 25},
        {"skill": "PowerPoint storytelling & slide writing", "importance": 91, "learn_hrs": 10},
        {"skill": "Market sizing & competitive analysis", "importance": 87, "learn_hrs": 10},
        {"skill": "Excel (advanced)", "importance": 85, "learn_hrs": 12},
        {"skill": "Client workshop facilitation", "importance": 80, "learn_hrs": 6},
        {"skill": "Due diligence & research synthesis", "importance": 78, "learn_hrs": 8},
        {"skill": "Executive stakeholder management", "importance": 85, "learn_hrs": 6},
        {"skill": "Project management (workplan, milestones)", "importance": 74, "learn_hrs": 8},
        {"skill": "Industry benchmarking", "importance": 68, "learn_hrs": 6},
    ],
    "Software Engineer (Backend)": [
        {"skill": "Python or Java or Node.js", "importance": 96, "learn_hrs": 60},
        {"skill": "REST API design & development", "importance": 92, "learn_hrs": 20},
        {"skill": "SQL & database design", "importance": 87, "learn_hrs": 20},
        {"skill": "System design & architecture", "importance": 88, "learn_hrs": 30},
        {"skill": "Docker & containerisation", "importance": 82, "learn_hrs": 12},
        {"skill": "Git & code review workflows", "importance": 90, "learn_hrs": 6},
        {"skill": "Testing (unit, integration)", "importance": 84, "learn_hrs": 10},
        {"skill": "Cloud services (AWS, GCP, Azure)", "importance": 80, "learn_hrs": 20},
        {"skill": "Microservices & event-driven architecture", "importance": 75, "learn_hrs": 20},
        {"skill": "Agile & technical documentation", "importance": 70, "learn_hrs": 5},
    ],
    "Financial Analyst": [
        {"skill": "Financial modelling (3-statement, DCF)", "importance": 95, "learn_hrs": 25},
        {"skill": "Excel (advanced: VLOOKUP, macros, pivot)", "importance": 92, "learn_hrs": 12},
        {"skill": "Valuation methods (comps, precedent transactions)", "importance": 88, "learn_hrs": 15},
        {"skill": "Accounting principles (P&L, balance sheet, cash flow)", "importance": 85, "learn_hrs": 20},
        {"skill": "Bloomberg or FactSet", "importance": 78, "learn_hrs": 10},
        {"skill": "PowerPoint for investment memos", "importance": 80, "learn_hrs": 8},
        {"skill": "SQL (basic data pulls)", "importance": 62, "learn_hrs": 20},
        {"skill": "Variance analysis & budgeting", "importance": 82, "learn_hrs": 8},
        {"skill": "Scenario & sensitivity analysis", "importance": 86, "learn_hrs": 10},
        {"skill": "Industry research & market analysis", "importance": 74, "learn_hrs": 8},
    ],
    "Account Manager": [
        {"skill": "Client relationship management & exec stakeholder mapping", "importance": 94, "learn_hrs": 8},
        {"skill": "Commercial negotiation & contract renewal", "importance": 90, "learn_hrs": 10},
        {"skill": "Upsell & cross-sell strategy", "importance": 86, "learn_hrs": 8},
        {"skill": "CRM (Salesforce, HubSpot)", "importance": 82, "learn_hrs": 8},
        {"skill": "QBR preparation & delivery", "importance": 80, "learn_hrs": 6},
        {"skill": "Revenue forecasting & pipeline management", "importance": 78, "learn_hrs": 8},
        {"skill": "Product knowledge & solution selling", "importance": 83, "learn_hrs": 6},
        {"skill": "Churn risk identification", "importance": 76, "learn_hrs": 5},
        {"skill": "Customer success metrics (NPS, CSAT, ARR)", "importance": 74, "learn_hrs": 5},
        {"skill": "Executive presentation skills", "importance": 80, "learn_hrs": 8},
    ],
    "Marketing Analyst": [
        {"skill": "Google Analytics 4 & web analytics", "importance": 92, "learn_hrs": 10},
        {"skill": "SQL for marketing data", "importance": 74, "learn_hrs": 20},
        {"skill": "Excel & data analysis", "importance": 85, "learn_hrs": 12},
        {"skill": "Paid media analysis (Meta, Google Ads)", "importance": 82, "learn_hrs": 10},
        {"skill": "A/B testing & CRO", "importance": 80, "learn_hrs": 10},
        {"skill": "CRM tools (Salesforce, HubSpot)", "importance": 72, "learn_hrs": 8},
        {"skill": "Campaign performance reporting", "importance": 87, "learn_hrs": 8},
        {"skill": "SEO & keyword analysis", "importance": 70, "learn_hrs": 8},
        {"skill": "Data visualisation (Tableau, Looker)", "importance": 74, "learn_hrs": 15},
        {"skill": "Attribution modelling", "importance": 68, "learn_hrs": 12},
    ],
    "Operations Manager": [
        {"skill": "Process design & improvement (Lean, Six Sigma)", "importance": 90, "learn_hrs": 20},
        {"skill": "KPI definition, dashboards & reporting", "importance": 87, "learn_hrs": 8},
        {"skill": "Project management (Gantt, milestones)", "importance": 85, "learn_hrs": 10},
        {"skill": "Cross-functional stakeholder management", "importance": 88, "learn_hrs": 6},
        {"skill": "Vendor & supplier management", "importance": 74, "learn_hrs": 6},
        {"skill": "Budget ownership & cost optimisation", "importance": 80, "learn_hrs": 8},
        {"skill": "SQL or Excel for operational data", "importance": 72, "learn_hrs": 15},
        {"skill": "Change management & team communication", "importance": 82, "learn_hrs": 8},
        {"skill": "ERP or workflow tools (SAP, Monday, Notion)", "importance": 68, "learn_hrs": 8},
        {"skill": "Root cause analysis", "importance": 76, "learn_hrs": 6},
    ],
    "UX / Product Designer": [
        {"skill": "Figma (components, auto layout, prototyping)", "importance": 96, "learn_hrs": 20},
        {"skill": "User research & usability testing", "importance": 90, "learn_hrs": 12},
        {"skill": "Information architecture & user flows", "importance": 86, "learn_hrs": 8},
        {"skill": "Design systems & component libraries", "importance": 82, "learn_hrs": 12},
        {"skill": "Wireframing & low-fi prototyping", "importance": 84, "learn_hrs": 6},
        {"skill": "Accessibility standards (WCAG)", "importance": 70, "learn_hrs": 6},
        {"skill": "Stakeholder presentation of design decisions", "importance": 78, "learn_hrs": 6},
        {"skill": "A/B testing & data-informed design", "importance": 72, "learn_hrs": 10},
        {"skill": "HTML/CSS basics", "importance": 58, "learn_hrs": 15},
        {"skill": "Workshop facilitation (design sprints)", "importance": 68, "learn_hrs": 6},
    ],
    "Sales Development Representative": [
        {"skill": "Cold outreach & sequencing (Outreach, Salesloft)", "importance": 92, "learn_hrs": 8},
        {"skill": "LinkedIn Sales Navigator prospecting", "importance": 87, "learn_hrs": 5},
        {"skill": "CRM data hygiene (Salesforce, HubSpot)", "importance": 84, "learn_hrs": 8},
        {"skill": "Discovery call & qualification (MEDDIC, BANT)", "importance": 88, "learn_hrs": 8},
        {"skill": "Objection handling", "importance": 82, "learn_hrs": 6},
        {"skill": "Email copywriting & personalisation", "importance": 85, "learn_hrs": 6},
        {"skill": "Pipeline reporting & forecasting", "importance": 72, "learn_hrs": 6},
        {"skill": "Account research & ICP targeting", "importance": 80, "learn_hrs": 5},
        {"skill": "Product demo basics", "importance": 68, "learn_hrs": 5},
        {"skill": "Commercial negotiation fundamentals", "importance": 70, "learn_hrs": 8},
    ],
}

# ── Learning catalogue ─────────────────────────────────────────────────────────
LEARNING_CATALOGUE = [
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "sql",             "course": "SQL Fundamentals Track",            "url": "https://www.datacamp.com/tracks/sql-fundamentals"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "python",          "course": "Python Programmer Track",           "url": "https://www.datacamp.com/tracks/python-programmer"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "pandas",          "course": "Data Manipulation with pandas",     "url": "https://www.datacamp.com/courses/data-manipulation-with-pandas"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "machine learning","course": "ML Scientist with Python",          "url": "https://www.datacamp.com/tracks/machine-learning-scientist-with-python"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "statistic",       "course": "Statistics Fundamentals",           "url": "https://www.datacamp.com/tracks/statistics-fundamentals-with-python"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "a/b test",        "course": "A/B Testing in Python",             "url": "https://www.datacamp.com/courses/customer-analytics-ab-testing-in-python"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "tableau",         "course": "Introduction to Tableau",           "url": "https://www.datacamp.com/courses/introduction-to-tableau"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "visuali",         "course": "Data Visualization with Python",    "url": "https://www.datacamp.com/tracks/data-visualization-with-python"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "nlp",             "course": "NLP Fundamentals in Python",        "url": "https://www.datacamp.com/tracks/natural-language-processing-in-python"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "deep learn",      "course": "Deep Learning in Python",           "url": "https://www.datacamp.com/tracks/deep-learning-in-python"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "git",             "course": "Introduction to Git",               "url": "https://www.datacamp.com/courses/introduction-to-git"},
    {"platform": "DataCamp",  "color": "#03ef62", "keyword": "power bi",        "course": "Introduction to Power BI",          "url": "https://www.datacamp.com/courses/introduction-to-power-bi"},
    {"platform": "Google",    "color": "#4285f4", "keyword": "machine learning","course": "Google ML Crash Course",            "url": "https://developers.google.com/machine-learning/crash-course"},
    {"platform": "Google",    "color": "#4285f4", "keyword": "cloud",           "course": "Google Cloud Associate Cert",       "url": "https://cloud.google.com/certification/cloud-engineer"},
    {"platform": "Google",    "color": "#4285f4", "keyword": "analytics",       "course": "Google Analytics Certification",    "url": "https://skillshop.docebosaas.com/learn/courses/14810"},
    {"platform": "Google",    "color": "#4285f4", "keyword": "project man",     "course": "Google PM Certificate",             "url": "https://grow.google/certificates/project-management/"},
    {"platform": "Google",    "color": "#4285f4", "keyword": "prompt",          "course": "Google Prompting Essentials Cert",  "url": "https://grow.google/certificates/prompting-essentials/"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "stakeholder", "course": "Stakeholder Management",       "url": "https://www.linkedin.com/learning/topics/stakeholder-management"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "excel",   "course": "Excel Essential Training",          "url": "https://www.linkedin.com/learning/excel-essential-training-microsoft-365"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "powerpoint","course": "PowerPoint Essential Training", "url": "https://www.linkedin.com/learning/powerpoint-essential-training-microsoft-365"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "agile",   "course": "Agile Foundations",                 "url": "https://www.linkedin.com/learning/agile-foundations"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "figma",   "course": "Figma Essential Training",          "url": "https://www.linkedin.com/learning/figma-essential-training"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "negotiation","course": "Negotiation Foundations",      "url": "https://www.linkedin.com/learning/negotiation-foundations"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "crm",     "course": "CRM Foundations",                   "url": "https://www.linkedin.com/learning/crm-foundations"},
    {"platform": "LinkedIn Learning", "color": "#0077b5", "keyword": "process", "course": "Business Process Management",       "url": "https://www.linkedin.com/learning/business-process-management-foundations"},
    {"platform": "Coursera",  "color": "#0056d2", "keyword": "financial model", "course": "Financial Modelling (Wharton)",     "url": "https://www.coursera.org/learn/wharton-financial-modeling"},
    {"platform": "Coursera",  "color": "#0056d2", "keyword": "product",         "course": "Product Management (Berkeley)",    "url": "https://www.coursera.org/professional-certificates/product-management"},
    {"platform": "Coursera",  "color": "#0056d2", "keyword": "deep learn",      "course": "Deep Learning Specialisation",      "url": "https://www.coursera.org/specializations/deep-learning"},
    {"platform": "Coursera",  "color": "#0056d2", "keyword": "docker",          "course": "Docker & Kubernetes",               "url": "https://www.coursera.org/learn/docker-for-developers"},
    {"platform": "Coursera",  "color": "#0056d2", "keyword": "sql",             "course": "SQL for Data Science",              "url": "https://www.coursera.org/learn/sql-for-data-science"},
    {"platform": "AWS",       "color": "#ff9900", "keyword": "cloud",           "course": "AWS Cloud Practitioner Cert",       "url": "https://aws.amazon.com/certification/certified-cloud-practitioner/"},
    {"platform": "Salesforce","color": "#00a1e0", "keyword": "salesforce",      "course": "Salesforce Admin Certification",    "url": "https://trailhead.salesforce.com/credentials/administrator"},
    {"platform": "HubSpot",   "color": "#ff7a59", "keyword": "hubspot",         "course": "HubSpot CRM Certification",         "url": "https://academy.hubspot.com/courses/hubspot-crm-certification"},
]

# ── Pure Python computation functions ─────────────────────────────────────────
def get_courses(skill_name: str) -> list:
    sl = skill_name.lower()
    return [c for c in LEARNING_CATALOGUE if c["keyword"] in sl or sl in c["keyword"]][:3]

def compute_gap(role_skills: list, user_skills: list) -> pd.DataFrame:
    rows = []
    for s in role_skills:
        have = s["skill"] in user_skills
        rows.append({
            "Skill": s["skill"],
            "Importance": s["importance"],
            "LearnHrs": s.get("learn_hrs", 10),
            "Have": have,
        })
    return pd.DataFrame(rows).sort_values("Importance", ascending=False)

def gap_score(df: pd.DataFrame) -> float:
    total = df["Importance"].sum()
    have  = df[df["Have"]]["Importance"].sum()
    return round((have / total) * 100, 1) if total > 0 else 0.0

def roi_score(importance: float, learn_hrs: float) -> float:
    return round(importance / max(learn_hrs, 1), 2)
