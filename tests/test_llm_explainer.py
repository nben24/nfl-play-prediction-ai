import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from llm_explainer import _build_reasoning


# --- Down and distance ---

def test_first_down_run_mentions_balanced():
    points = _build_reasoning(down=1, ydstogo=10, yardline_100=50,
                               score_diff=0, qtr=1, seconds=3600, prediction="run")
    assert any("1st" in p and "balanced" in p for p in points)


def test_first_down_pass_mentions_balanced():
    points = _build_reasoning(down=1, ydstogo=10, yardline_100=50,
                               score_diff=0, qtr=1, seconds=3600, prediction="pass")
    assert any("1st" in p and "balanced" in p for p in points)


def test_third_and_long_is_passing_situation():
    points = _build_reasoning(down=3, ydstogo=8, yardline_100=40,
                               score_diff=0, qtr=2, seconds=1800, prediction="pass")
    assert any("passing" in p.lower() or "clear" in p.lower() for p in points)


def test_third_and_short_mentions_run_or_quick_pass():
    points = _build_reasoning(down=3, ydstogo=2, yardline_100=40,
                               score_diff=0, qtr=2, seconds=1800, prediction="run")
    assert any("short" in p.lower() or "run" in p.lower() for p in points)


def test_fourth_and_short_viable():
    points = _build_reasoning(down=4, ydstogo=1, yardline_100=35,
                               score_diff=0, qtr=4, seconds=600, prediction="run")
    assert any("short" in p.lower() or "run" in p.lower() for p in points)


# --- Field position ---

def test_goal_line_mentioned_inside_5():
    points = _build_reasoning(down=1, ydstogo=3, yardline_100=3,
                               score_diff=0, qtr=2, seconds=1800, prediction="run")
    assert any("goal-line" in p.lower() or "goal line" in p.lower() for p in points)


def test_red_zone_mentioned_inside_20():
    points = _build_reasoning(down=2, ydstogo=8, yardline_100=15,
                               score_diff=0, qtr=3, seconds=2000, prediction="pass")
    assert any("red zone" in p.lower() for p in points)


def test_deep_own_territory_mentioned():
    points = _build_reasoning(down=1, ydstogo=10, yardline_100=80,
                               score_diff=0, qtr=1, seconds=3400, prediction="run")
    assert any("own territory" in p.lower() or "turnovers" in p.lower() for p in points)


def test_midfield_no_field_position_point():
    # Between own 35 and opponent's 35 — no special field position note expected
    points = _build_reasoning(down=1, ydstogo=10, yardline_100=50,
                               score_diff=0, qtr=2, seconds=2400, prediction="run")
    field_points = [p for p in points if "zone" in p.lower() or "territory" in p.lower()
                    or "goal" in p.lower() or "end zone" in p.lower()]
    assert len(field_points) == 0


# --- Score and time ---

def test_game_winning_td_attempt_described():
    points = _build_reasoning(down=4, ydstogo=3, yardline_100=3,
                               score_diff=-2, qtr=4, seconds=24, prediction="pass")
    assert any("wins the game" in p.lower() or "touchdown" in p.lower() for p in points)


def test_game_winning_td_explicitly_says_not_tied():
    points = _build_reasoning(down=4, ydstogo=3, yardline_100=3,
                               score_diff=-2, qtr=4, seconds=24, prediction="pass")
    full_text = " ".join(points).lower()
    assert "not tied" in full_text


def test_large_deficit_triggers_pass_urgency():
    points = _build_reasoning(down=2, ydstogo=6, yardline_100=40,
                               score_diff=-14, qtr=3, seconds=1800, prediction="pass")
    assert any("trailing" in p.lower() or "pass" in p.lower() for p in points)


def test_large_lead_second_half_run_clock():
    points = _build_reasoning(down=1, ydstogo=10, yardline_100=40,
                               score_diff=17, qtr=3, seconds=1800, prediction="run")
    assert any("clock" in p.lower() or "leading" in p.lower() for p in points)


def test_tied_early_no_urgency_point():
    # Q1 tied — no score/time point should be added
    points = _build_reasoning(down=1, ydstogo=10, yardline_100=70,
                               score_diff=0, qtr=1, seconds=3400, prediction="run")
    score_time_points = [p for p in points if "tied" in p.lower() or "trailing" in p.lower()
                         or "leading" in p.lower() or "clock" in p.lower()]
    assert len(score_time_points) == 0


def test_two_minute_warning_halftime():
    points = _build_reasoning(down=2, ydstogo=7, yardline_100=45,
                               score_diff=0, qtr=2, seconds=90, prediction="pass")
    assert any("half" in p.lower() for p in points)


# --- At least one point always generated ---

def test_always_returns_at_least_one_point():
    points = _build_reasoning(down=2, ydstogo=5, yardline_100=45,
                               score_diff=3, qtr=2, seconds=2000, prediction="pass")
    assert len(points) >= 1
