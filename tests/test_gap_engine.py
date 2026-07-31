"""
Tests for the scoring engine that everything else in the app sits on top of.

These cover the pure-Python logic only — no Streamlit, no network, no O*NET
files — so they run anywhere with just pandas installed.
"""

import pytest

from roles_data import CURATED_ROLES, compute_gap, gap_score, roi_score
from agent_tools import _fuzzy_match_role, compute_gap_analysis, find_closest_roles
from data_loader import categorise_skill


ROLE = [
    {"skill": "SQL", "importance": 100, "learn_hrs": 40},
    {"skill": "Python", "importance": 80, "learn_hrs": 60},
    {"skill": "Storytelling", "importance": 20, "learn_hrs": 10},
]


# ── compute_gap ───────────────────────────────────────────────────────────────

def test_compute_gap_flags_only_skills_the_user_has():
    df = compute_gap(ROLE, ["SQL", "Storytelling"])
    have = dict(zip(df["Skill"], df["Have"]))
    assert have == {"SQL": True, "Python": False, "Storytelling": True}


def test_compute_gap_sorts_by_importance_descending():
    df = compute_gap(ROLE, [])
    assert list(df["Skill"]) == ["SQL", "Python", "Storytelling"]


def test_compute_gap_matching_is_exact_not_substring():
    # "SQL" must not be credited by owning "NoSQL" — substring matching here
    # would silently inflate every readiness score in the app.
    df = compute_gap(ROLE, ["NoSQL"])
    assert not df["Have"].any()


def test_compute_gap_defaults_missing_learn_hours():
    df = compute_gap([{"skill": "Excel", "importance": 50}], [])
    assert df.iloc[0]["LearnHrs"] == 10


# ── gap_score ─────────────────────────────────────────────────────────────────

def test_gap_score_is_importance_weighted_not_a_simple_count():
    """One high-importance skill should beat two low-importance ones."""
    one_important = gap_score(compute_gap(ROLE, ["SQL"]))          # 100 / 200
    two_trivial = gap_score(compute_gap(ROLE, ["Storytelling"]))   # 20 / 200
    assert one_important > two_trivial
    assert one_important == 50.0


def test_gap_score_bounds():
    assert gap_score(compute_gap(ROLE, [])) == 0.0
    assert gap_score(compute_gap(ROLE, ["SQL", "Python", "Storytelling"])) == 100.0


def test_gap_score_does_not_divide_by_zero_when_all_importances_are_zero():
    zeroed = [{"skill": "A", "importance": 0, "learn_hrs": 5}]
    assert gap_score(compute_gap(zeroed, ["A"])) == 0.0


# ── roi_score ─────────────────────────────────────────────────────────────────

def test_roi_score_prefers_cheaper_skills_of_equal_importance():
    assert roi_score(80, 10) > roi_score(80, 40)


def test_roi_score_clamps_zero_hours_instead_of_dividing_by_zero():
    assert roi_score(50, 0) == 50.0


# ── role matching ─────────────────────────────────────────────────────────────

def test_fuzzy_match_is_case_insensitive_and_partial():
    assert _fuzzy_match_role("data analyst") == "Data Analyst"
    assert _fuzzy_match_role("Data Analyst") == "Data Analyst"


def test_fuzzy_match_returns_none_for_unknown_role():
    assert _fuzzy_match_role("Underwater Basket Weaver") is None


# ── agent tools ───────────────────────────────────────────────────────────────

def test_gap_analysis_orders_missing_skills_by_roi():
    role = next(iter(CURATED_ROLES))
    result = compute_gap_analysis(role, [])
    rois = [s["roi"] for s in result["missing_skills"]]
    assert rois == sorted(rois, reverse=True)


def test_gap_analysis_reports_unknown_role_instead_of_raising():
    assert "error" in compute_gap_analysis("Nonexistent Role", ["SQL"])


def test_find_closest_roles_ranks_by_readiness_descending():
    ranked = find_closest_roles(["SQL", "Python", "Excel"])["ranked_roles"]
    scores = [r["readiness_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_find_closest_roles_requires_at_least_one_skill():
    assert "error" in find_closest_roles([])


# ── skill categorisation ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "skill,expected",
    [
        ("Programming", "Technical"),
        ("Mathematics", "Technical"),
        ("Active Listening", "Soft"),
        ("Negotiation", "Soft"),
        ("Repairing", "Domain"),
    ],
)
def test_categorise_skill(skill, expected):
    assert categorise_skill(skill) == expected


def test_categorise_skill_is_case_insensitive():
    assert categorise_skill("PROGRAMMING") == categorise_skill("programming")
