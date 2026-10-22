from __future__ import annotations

import json
from typing import cast

import pytest

from deltamem.eval.locomo_protocol import (
    build_official_question_prompt,
    canonicalize_locomo_prediction,
    prepare_locomo_question,
)
from deltamem.eval.locomo_delta import (
    DEFAULT_LOCOMO_CATEGORIES,
    build_question_prompt,
    evaluate_locomo_conditions,
    load_locomo_samples,
    score_locomo_prediction,
    validate_locomo_question_count,
)


def make_sample() -> dict:
    return {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "9:00 am on 1 Jan, 2024",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I bought oranges."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "I went to the library."},
            ],
            "session_2_date_time": "10:00 am on 2 Jan, 2024",
            "session_2": [
                {
                    "speaker": "Alice",
                    "dia_id": "D2:1",
                    "text": "I shared a picture.",
                    "blip_caption": "a sunset over the water",
                }
            ],
        },
        "qa": [],
    }


def test_build_question_prompt_includes_no_info_instruction_for_adversarial() -> None:
    prompt = build_question_prompt({"question": "What did Alice realize?", "category": 5})
    assert "Select the correct answer by writing (a) or (b)." in prompt


def test_score_locomo_prediction_matches_category_rules() -> None:
    assert score_locomo_prediction(
        {"category": 1, "answer": "adoption agencies"},
        "Adoption agencies",
    ) == 1.0
    assert score_locomo_prediction(
        {"category": 2, "answer": "7 May 2023"},
        "7 May, 2023",
    ) == 1.0
    assert score_locomo_prediction(
        {"category": 4, "answer": "library"},
        "libraries",
    ) == 1.0
    assert score_locomo_prediction(
        {"category": 5},
        "No information available",
    ) == 1.0
    assert score_locomo_prediction(
        {"category": 5},
        "She realized self-care is important.",
    ) == 0.0


def test_prepare_locomo_question_builds_deterministic_adversarial_options() -> None:
    spec = prepare_locomo_question(
        {"question": "What was the plan?", "category": 5, "answer": "Go hiking"},
        sample_id="sample-1",
        question_index=0,
        seed=42,
    )
    prompt = build_official_question_prompt(spec)
    assert "(a)" in prompt and "(b)" in prompt
    assert "Select the correct answer by writing (a) or (b)." in prompt
    normalized = canonicalize_locomo_prediction("(a)", spec)
    assert normalized in {"No information available", "Go hiking"}


def test_load_locomo_samples_excludes_adversarial_by_default_category_set(tmp_path) -> None:
    data_file = tmp_path / "locomo_100q.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    **make_sample(),
                    "qa": [
                        {"question": "Q1", "answer": "A1", "category": 4},
                        {"question": "Q2", "answer": "A2", "category": 5},
                        {"question": "Q3", "answer": "A3", "category": 2},
                    ],
                }
            ]
        )
    )

    filtered = load_locomo_samples(
        data_file,
        max_conversations=None,
        max_questions_per_conversation=None,
        categories=list(DEFAULT_LOCOMO_CATEGORIES),
    )

    assert [qa["category"] for qa in filtered[0]["qa"]] == [4, 2]


def test_validate_locomo_question_count_skips_filename_check_when_categories_filtered(tmp_path) -> None:
    data_file = tmp_path / "locomo_100q.json"
    samples = [{"qa": [{"category": 4}] * 3}]
    data_file.write_text("[]")

    validate_locomo_question_count(data_file, samples, categories=[1, 2, 3, 4])


def test_evaluate_locomo_conditions_uses_official_prompt_only(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = make_sample()
    sample["qa"] = [{"question": "What did Alice buy?", "answer": "oranges", "category": 1}]
    calls = []
    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return "oranges", "oranges"
    monkeypatch.setattr("deltamem.eval.locomo_delta.generate_official_full_history_answer", fake_generate)
    args = dict(
        model="model", tokenizer="tokenizer", device="cpu", samples=[sample],
        question_tasks=[(0, 0, 0)], condition_names=["official_prompt"],
        max_new_tokens=16, seed=42, answer_reserve_tokens=50,
        official_do_sample=False, official_temperature=1.0,
        official_top_p=1.0, official_top_k=0, eval_batch_size=1,
    )
    records = evaluate_locomo_conditions(**args)
    assert len(calls) == 1
    assert calls[0]["do_sample"] is False
    conditions = cast(dict[str, dict[str, object]], records[0][2]["conditions"])
    assert conditions["official_prompt"]["score"] == 1.0
    with pytest.raises(ValueError, match="Only the official LoCoMo prompt"):
        evaluate_locomo_conditions(**{**args, "condition_names": ["unsupported"]})
