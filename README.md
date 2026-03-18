# 🎯 Career Bridge

https://skillgapanalyzer-miba.streamlit.app/

A Streamlit prototype that shows exactly which skills you're missing for your target job role — ranked by importance using O*NET labor data, with optional AI-powered CV parsing.

Built for the PDAI Prototyping Assignment.

---

## What It Does

- Select a target job role (Data Analyst, ML Engineer, Product Manager, etc.)
- Check off the skills you already have — or upload your CV as a `.txt` file and let AI extract them automatically
- See a visual bar chart of all required skills, color-coded green (have) vs red (missing)
- Get a readiness score weighted by skill importance
- View a personalized learning roadmap with course links for every missing skill

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/job-skill-gap-analyzer.git
cd job-skill-gap-analyzer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key (for CV parsing feature)
```bash
export ANTHROPIC_API_KEY="your-api-key-here"   # Mac/Linux
set ANTHROPIC_API_KEY=your-api-key-here         # Windows
```
Get a free API key at: https://console.anthropic.com

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## How to Use the CV Upload Feature

1. Save your CV as a plain `.txt` file (copy-paste from Word or PDF into a text file)
2. Upload it in the sidebar
3. Click "Extract Skills with AI"
4. Claude will read your CV and automatically check the skills you have

---

## Streamlit Widgets Used

- `st.tabs` — organizes the app into Gap Analysis, Learning Roadmap, and About sections
- `st.metric` — displays readiness score and skill counts as bold summary cards
- `st.progress` — shows overall readiness as a visual progress bar
- `st.expander` — each missing skill expands to show priority level and course link
- `st.file_uploader` — CV upload
- `st.checkbox` — skill selection
- `st.plotly_chart` — horizontal bar chart with color-coded skills

---

## Data Source

Skill importance scores are based on the **O*NET database** (U.S. Department of Labor, Employment and Training Administration) — the authoritative source for occupational skills and importance ratings.

---

## Roadmap (Future Features)

- Resume PDF parsing (not just .txt)
- Salary data per skill from Glassdoor/Levels.fyi
- Skill acquisition tracker over time
- More job roles
- Export gap report as PDF
