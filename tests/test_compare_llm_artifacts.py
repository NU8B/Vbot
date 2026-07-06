import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from compare_llm_artifacts import benchmark_table, build_report, evaluate_policy, judge_table


def _benchmark(path, brevity, safety, versions=None):
    return {
        "_path": path,
        "prompt_versions": versions,
        "characters": {
            "Amelia": {
                "aggregate": {
                    "tts_safety_rate": safety,
                    "brevity_rate": brevity,
                    "persona_adherence_rate": 0.8,
                    "character_break_rate": 0.1,
                    "avg_word_count": 42.0,
                    "avg_latency_s": 0.6,
                }
            }
        },
    }


def _benchmark_with_persona(path, persona):
    artifact = _benchmark(path, brevity=0.7, safety=0.9)
    artifact["characters"]["Amelia"]["aggregate"]["persona_adherence_rate"] = persona
    return artifact


def _judge(path, voice, break_rate):
    return {
        "_path": path,
        "characters": {
            "Amelia": {
                "aggregate": {
                    "avg_persona_voice": voice,
                    "avg_engagement": 4.5,
                    "avg_kayfabe": 4.0,
                    "kayfabe_break_rate": break_rate,
                }
            }
        },
    }


def test_benchmark_table_formats_percent_and_numeric_deltas():
    table = benchmark_table(
        _benchmark("old.json", brevity=0.3, safety=0.8),
        _benchmark("new.json", brevity=0.6, safety=1.0),
    )

    assert "| Amelia | brevity | 30% | 60% | +30% |" in table
    assert "| Amelia | TTS-safe | 80% | 100% | +20% |" in table
    assert "| Amelia | avg words | 42.0 | 42.0 | +0.0 |" in table


def test_judge_table_reports_missing_pair_gracefully():
    assert "_No complete old/new judge pair supplied._" in judge_table(None, None)


def test_judge_table_formats_score_and_break_rate_deltas():
    table = judge_table(_judge("old.json", voice=4.0, break_rate=0.2), _judge("new.json", voice=4.5, break_rate=0.0))

    assert "| Amelia | voice | 4.00 | 4.50 | +0.50 |" in table
    assert "| Amelia | judge breaks | 20% | 0% | -20% |" in table


def test_report_includes_prompt_versions_and_input_names():
    report = build_report(
        _benchmark("old.json", brevity=0.3, safety=0.8),
        _benchmark("new.json", brevity=0.6, safety=0.9, versions={"Amelia": 2}),
    )

    assert "- old benchmark: `old.json`" in report
    assert "- old prompt versions: not recorded" in report
    assert "- new prompt versions: Amelia=v2" in report
    assert "Verdict: `NEEDS REVIEW`" in report


def test_policy_rejects_large_tts_safety_or_brevity_regression():
    result = evaluate_policy(
        _benchmark("old.json", brevity=0.6, safety=0.9),
        _benchmark("new.json", brevity=0.4, safety=0.7),
        _judge("old_judge.json", voice=4.0, break_rate=0.0),
        _judge("new_judge.json", voice=4.2, break_rate=0.0),
    )

    assert result["verdict"] == "REJECT"
    assert any("TTS-safety dropped" in failure for failure in result["failures"])
    assert any("brevity dropped" in failure for failure in result["failures"])


def test_policy_rejects_judge_kayfabe_break_increase():
    result = evaluate_policy(
        _benchmark("old.json", brevity=0.6, safety=0.9),
        _benchmark("new.json", brevity=0.7, safety=0.9),
        _judge("old_judge.json", voice=4.0, break_rate=0.0),
        _judge("new_judge.json", voice=4.2, break_rate=0.2),
    )

    assert result["verdict"] == "REJECT"
    assert result["failures"] == ["Amelia: judge kayfabe break rate increased by +20%"]


def test_policy_warns_on_persona_keyword_drop_without_rejecting():
    result = evaluate_policy(
        _benchmark_with_persona("old.json", persona=0.9),
        _benchmark_with_persona("new.json", persona=0.7),
        _judge("old_judge.json", voice=4.0, break_rate=0.0),
        _judge("new_judge.json", voice=4.2, break_rate=0.0),
    )

    assert result["verdict"] == "NEEDS REVIEW"
    assert result["failures"] == []
    assert result["warnings"] == ["Amelia: persona keyword hits dropped by -20%"]


def test_policy_promotes_when_contract_and_judge_metrics_do_not_regress():
    result = evaluate_policy(
        _benchmark("old.json", brevity=0.6, safety=0.9),
        _benchmark("new.json", brevity=0.7, safety=0.9),
        _judge("old_judge.json", voice=4.0, break_rate=0.1),
        _judge("new_judge.json", voice=4.2, break_rate=0.0),
    )

    assert result == {
        "verdict": "PROMOTE",
        "failures": [],
        "warnings": [],
        "policy": {
            "max_tts_safety_drop": 0.10,
            "max_brevity_drop": 0.10,
            "max_persona_hit_drop_warning": 0.10,
        },
    }
