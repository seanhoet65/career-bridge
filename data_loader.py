"""
data_loader.py
Loads and processes real O*NET database files.
Run setup_data.py first to download the CSV files into the /data folder.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_occupations() -> pd.DataFrame:
    """Load occupation titles from O*NET Occupation Data."""
    path = os.path.join(DATA_DIR, "Occupation Data.txt")
    df = pd.read_csv(path, sep="\t", usecols=["O*NET-SOC Code", "Title"])
    df.columns = ["soc_code", "title"]
    return df


def load_skills() -> pd.DataFrame:
    """
    Load skills importance scores from O*NET Skills.txt.
    Scale ID 'IM' = Importance (1-5 scale, we normalise to 0-100).
    """
    path = os.path.join(DATA_DIR, "Skills.txt")

    # Read all columns first so we can handle any O*NET version
    df = pd.read_csv(path, sep="\t")

    # Normalise column names: strip whitespace, lowercase for matching
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Find the right columns regardless of exact capitalisation/spacing
    def find_col(candidates):
        for c in df.columns:
            if c.strip().lower() in [x.lower() for x in candidates]:
                return c
        raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")

    soc_col   = find_col(["O*NET-SOC Code", "onet-soc code"])
    elem_col  = find_col(["Element Name", "element name"])
    scale_col = find_col(["Scale ID", "scale id"])
    val_col   = find_col(["Data Value", "data value"])

    df = df[[soc_col, elem_col, scale_col, val_col]].copy()
    df.columns = ["soc_code", "skill", "scale_id", "value"]

    # Keep only Importance scores
    df = df[df["scale_id"] == "IM"].copy()
    # Normalise from 1-5 scale to 0-100
    df["importance"] = ((df["value"] - 1) / 4 * 100).round(1)
    return df[["soc_code", "skill", "importance"]]


def get_job_roles(min_skills: int = 8) -> list[str]:
    """
    Return a sorted list of occupation titles that have enough skill data.
    """
    occupations = load_occupations()
    skills = load_skills()
    # Only keep occupations that have at least min_skills entries
    counts = skills.groupby("soc_code").size().reset_index(name="n")
    valid_codes = counts[counts["n"] >= min_skills]["soc_code"]
    filtered = occupations[occupations["soc_code"].isin(valid_codes)]
    return sorted(filtered["title"].unique().tolist())


def get_skill_profile(job_title: str, top_n: int = 12) -> list[dict]:
    """
    Given a job title string, return the top N skills by importance
    as a list of dicts: [{skill, importance, category}, ...]
    """
    occupations = load_occupations()
    skills = load_skills()

    # Find the SOC code(s) for this title
    match = occupations[occupations["title"] == job_title]
    if match.empty:
        return []

    soc_code = match.iloc[0]["soc_code"]

    # Get skills for this occupation, sorted by importance
    role_skills = (
        skills[skills["soc_code"] == soc_code]
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    return [
        {
            "skill": row["skill"],
            "importance": row["importance"],
            "category": categorise_skill(row["skill"]),
        }
        for _, row in role_skills.iterrows()
    ]


def categorise_skill(skill_name: str) -> str:
    """
    Simple heuristic to tag a skill as Technical, Soft, or Domain.
    O*NET uses its own taxonomy — this is a lightweight proxy.
    """
    technical_keywords = [
        "programming", "software", "database", "network", "equipment",
        "technology", "computer", "systems", "mathematics", "science",
        "engineering", "design", "analysis", "statistics", "data",
    ]
    soft_keywords = [
        "communication", "listening", "speaking", "social", "coordination",
        "persuasion", "negotiation", "instructing", "service", "monitoring",
        "management", "leadership", "thinking", "judgment",
    ]
    skill_lower = skill_name.lower()
    if any(k in skill_lower for k in technical_keywords):
        return "Technical"
    if any(k in skill_lower for k in soft_keywords):
        return "Soft"
    return "Domain"


def data_is_ready() -> bool:
    """Check whether the O*NET data files have been downloaded."""
    required = ["Occupation Data.txt", "Skills.txt"]
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in required)
