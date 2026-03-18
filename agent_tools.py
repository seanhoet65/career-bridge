"""
agent_tools.py
Python computation tools for the Career Advisor agent.

Each function is a real Python computation against CURATED_ROLES data.
These are passed directly to Gemini as AFC tools — the LLM decides when to
call them, the SDK executes them with the LLM's chosen arguments, and the
results feed back into subsequent LLM reasoning turns.

The non-straightforwardness: these aren't string lookups — they run genuine
numerical scoring (gap_score engine), set arithmetic, timeline arithmetic,
and radar chart data generation. The LLM receives structured numeric output
and must interpret it for the user.
"""

from roles_data import CURATED_ROLES, compute_gap, gap_score
import pandas as pd


# ── Helper ────────────────────────────────────────────────────────────────────

def _fuzzy_match_role(role_name: str) -> str | None:
    """Case-insensitive partial match against CURATED_ROLES keys."""
    if role_name in CURATED_ROLES:
        return role_name
    lower = role_name.lower()
    for r in CURATED_ROLES:
        if lower in r.lower() or r.lower() in lower:
            return r
    return None


# ── Tool 1 ────────────────────────────────────────────────────────────────────

def find_closest_roles(skill_list: list[str]) -> dict:
    """
    Score ALL 14 curated career roles against the provided skill list using the
    gap_score engine. Returns a ranked list with numeric readiness percentages,
    matched skill counts, and total skills per role.
    """
    if not skill_list:
        return {"error": "Please provide at least one skill to match against."}

    results = []
    for role_name, role_skills in CURATED_ROLES.items():
        df = compute_gap(role_skills, skill_list)
        score = gap_score(df)
        n_have = int(df["Have"].sum())
        n_total = len(df)
        results.append({
            "role": role_name,
            "readiness_score": score,
            "skills_matched": n_have,
            "total_skills": n_total,
            "gap_skills": n_total - n_have,
        })

    results.sort(key=lambda x: x["readiness_score"], reverse=True)
    return {
        "ranked_roles": results,
        "top_match": results[0]["role"] if results else None,
        "top_score": results[0]["readiness_score"] if results else 0,
    }


# ── Tool 2 ────────────────────────────────────────────────────────────────────

def get_role_requirements(role_name: str) -> dict:
    """
    Return the full skill list for a specific role, with importance scores
    and estimated learning hours per skill. Fuzzy-matches role names.
    """
    match = _fuzzy_match_role(role_name)
    if not match:
        return {
            "error": f"Role '{role_name}' not found.",
            "available_roles": list(CURATED_ROLES.keys()),
        }
    skills = CURATED_ROLES[match]
    return {
        "role": match,
        "skill_count": len(skills),
        "skills": [
            {"skill": s["skill"], "importance": s["importance"], "learn_hrs": s["learn_hrs"]}
            for s in sorted(skills, key=lambda x: x["importance"], reverse=True)
        ],
        "total_learn_hours": sum(s["learn_hrs"] for s in skills),
    }


# ── Tool 3 ────────────────────────────────────────────────────────────────────

def compute_gap_analysis(role_name: str, user_skills: list[str]) -> dict:
    """
    Run full gap analysis between a user's skills and a target role.
    Computes readiness score, matched skills, missing skills with hours,
    and total learning investment required. Core Python computation engine.
    """
    match = _fuzzy_match_role(role_name)
    if not match:
        return {"error": f"Role '{role_name}' not found."}

    role_skills = CURATED_ROLES[match]
    df = compute_gap(role_skills, user_skills)
    score = gap_score(df)

    missing = df[~df["Have"]].copy()
    have = df[df["Have"]].copy()

    # Sort missing by ROI (importance per hour) so top skills come first
    missing["roi"] = missing["Importance"] / missing["LearnHrs"].clip(lower=1)
    missing = missing.sort_values("roi", ascending=False)

    return {
        "role": match,
        "readiness_score": score,
        "skills_matched": int(have.shape[0]),
        "skills_missing_count": int(missing.shape[0]),
        "total_learn_hours": int(missing["LearnHrs"].sum()),
        "missing_skills": [
            {"skill": r["Skill"], "importance": int(r["Importance"]),
             "learn_hrs": int(r["LearnHrs"]), "roi": round(r["roi"], 2)}
            for _, r in missing.iterrows()
        ],
        "matched_skills": list(have["Skill"]),
    }


# ── Tool 4 ────────────────────────────────────────────────────────────────────

def compare_roles(role_a: str, role_b: str, user_skills: list[str]) -> dict:
    """
    Compare two roles side-by-side. Computes readiness scores for both,
    shared vs unique skills, and calculates pivot hours — the additional
    learning hours needed to move from role A proficiency to role B.
    """
    match_a = _fuzzy_match_role(role_a)
    match_b = _fuzzy_match_role(role_b)

    if not match_a or not match_b:
        return {"error": "Could not match one or both role names. Try the full role name."}

    skills_a = {s["skill"] for s in CURATED_ROLES[match_a]}
    skills_b = {s["skill"] for s in CURATED_ROLES[match_b]}
    shared    = skills_a & skills_b
    only_a    = skills_a - skills_b
    only_b    = skills_b - skills_a

    df_a = compute_gap(CURATED_ROLES[match_a], user_skills)
    df_b = compute_gap(CURATED_ROLES[match_b], user_skills)

    # Pivot effort: hours for B-specific skills the user doesn't have
    pivot_hrs = sum(
        s["learn_hrs"] for s in CURATED_ROLES[match_b]
        if s["skill"] not in skills_a and s["skill"] not in user_skills
    )

    return {
        "role_a": {
            "name": match_a,
            "readiness_score": gap_score(df_a),
            "unique_skills": list(only_a),
        },
        "role_b": {
            "name": match_b,
            "readiness_score": gap_score(df_b),
            "unique_skills": list(only_b),
        },
        "shared_skills": list(shared),
        "shared_skill_count": len(shared),
        "skills_transferable_a_to_b": len(shared & set(user_skills)),
        "pivot_hours_a_to_b": int(pivot_hrs),
        "stronger_fit": match_a if gap_score(df_a) >= gap_score(df_b) else match_b,
    }


# ── Tool 5 ────────────────────────────────────────────────────────────────────

def estimate_transition_time(role_name: str, user_skills: list[str]) -> dict:
    """
    Calculate total learning hours needed to close skill gaps for a role,
    then project realistic timelines at 3, 5, 10, 15, and 20 hours per week.
    Also returns the top 3 quick-win skills (highest ROI, lowest time investment).
    """
    match = _fuzzy_match_role(role_name)
    if not match:
        return {"error": f"Role '{role_name}' not found."}

    role_skills = CURATED_ROLES[match]
    df = compute_gap(role_skills, user_skills)
    missing = df[~df["Have"]].copy()

    if missing.empty:
        return {
            "role": match,
            "total_gap_hours": 0,
            "message": "No skill gaps detected — you already meet all requirements!",
            "timelines": {},
            "top_quick_wins": [],
        }

    total_hrs = int(missing["LearnHrs"].sum())

    timelines = {}
    for hrs_pw in [3, 5, 10, 15, 20]:
        weeks  = total_hrs / hrs_pw
        months = weeks / 4.33
        timelines[f"{hrs_pw}_hrs_per_week"] = {
            "weeks": round(weeks, 1),
            "months": round(months, 1),
        }

    # Quick wins: highest ROI (importance ÷ hours), learn in ≤15 hrs
    missing["roi"] = missing["Importance"] / missing["LearnHrs"].clip(lower=1)
    quick = missing[missing["LearnHrs"] <= 15].nlargest(3, "roi")
    quick_wins = [
        {"skill": r["Skill"], "hours": int(r["LearnHrs"]), "importance": int(r["Importance"])}
        for _, r in quick.iterrows()
    ]

    return {
        "role": match,
        "total_gap_hours": total_hrs,
        "missing_skill_count": int(missing.shape[0]),
        "timelines": timelines,
        "top_quick_wins": quick_wins,
    }


# ── Tool 6 ────────────────────────────────────────────────────────────────────

def get_skill_radar_data(role_name: str, user_skills: list[str]) -> dict:
    """
    Generate Plotly radar chart data comparing the user's current skill coverage
    against a target role's top 8 skills. Returns chart-ready values scaled 0-100.
    The chart renders automatically in the Career Advisor when this tool is called.
    """
    match = _fuzzy_match_role(role_name)
    if not match:
        return {"error": f"Role '{role_name}' not found."}

    # Top 8 skills by importance for a readable radar
    role_skills = sorted(CURATED_ROLES[match], key=lambda x: x["importance"], reverse=True)[:8]
    df = compute_gap(role_skills, user_skills)

    categories   = [s[:22] + "…" if len(s) > 22 else s for s in df["Skill"].tolist()]
    role_vals    = df["Importance"].astype(float).tolist()
    user_vals    = [float(imp) if have else 0.0
                    for imp, have in zip(df["Importance"], df["Have"])]
    coverage_pct = round(sum(user_vals) / sum(role_vals) * 100, 1) if sum(role_vals) > 0 else 0.0

    return {
        "role": match,
        "categories": categories,
        "role_target": role_vals,
        "user_current": user_vals,
        "coverage_pct": coverage_pct,
        "chart_type": "radar",
    }


# ── All tools as a list for AFC registration ──────────────────────────────────

ALL_TOOLS = [
    find_closest_roles,
    get_role_requirements,
    compute_gap_analysis,
    compare_roles,
    estimate_transition_time,
    get_skill_radar_data,
]
