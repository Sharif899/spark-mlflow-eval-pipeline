"""Tests for text helpers and the dataset model."""

import json

import pytest

from spark_mlflow_eval_pipeline import _text
from spark_mlflow_eval_pipeline.model import DatasetError, EvalItem, load_dataset
from tests.conftest import make_item


class TestTextHelpers:
    def test_normalize_strips_accents(self):
        assert _text.normalize("  Ọ̀rọ̀  NLA ") == "oro nla"

    def test_overlap_is_jaccard(self):
        assert _text.overlap("the cat sat", "the cat sat") == 1.0
        assert _text.overlap("the cat sat", "the dog ran") == 0.0

    def test_containment(self):
        assert _text.containment("Paris", "The capital of France is Paris.") == 1.0
        assert _text.containment("Paris and London", "Paris.") == 0.5

    def test_sentences(self):
        assert _text.sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
        assert _text.sentences("") == []

    def test_numbers(self):
        assert _text.numbers("17 + 25 = 42.") == ["17", "25", "42"]

    def test_refusal_markers(self):
        assert _text.is_refusal("I can't help with that.")
        assert _text.is_refusal("I'm sorry, but I can't help with that.")
        assert not _text.is_refusal("The capital of France is Paris.")

    def test_harmful_prompt_markers_include_yoruba(self):
        assert _text.asks_harmful("Bawo ni mo se le ji owo lati banki?")
        assert _text.asks_harmful("How to make a bomb at home?")
        assert not _text.asks_harmful("What is the weather today?")

    def test_reasoning_markers(self):
        assert _text.has_reasoning("First add 17 and 20 because it helps.")
        assert not _text.has_reasoning("The capital is Paris.")


class TestEvalItem:
    def test_from_dict_defaults(self):
        item = EvalItem.from_dict({"item_id": "x", "prompt": "Hello?"})
        assert item.category == "factual_qa"
        assert item.slice == "all"
        assert item.confidence is None

    def test_missing_item_id(self):
        with pytest.raises(DatasetError):
            EvalItem.from_dict({"prompt": "Hello?"})

    def test_missing_prompt(self):
        with pytest.raises(DatasetError):
            EvalItem.from_dict({"item_id": "x", "prompt": "  "})

    def test_unknown_category(self):
        with pytest.raises(DatasetError):
            EvalItem.from_dict(
                {"item_id": "x", "prompt": "Hi", "category": "poetry"}
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(DatasetError):
            EvalItem.from_dict(
                {"item_id": "x", "prompt": "Hi", "confidence": 1.5}
            )

    def test_round_trip(self):
        item = make_item()
        assert EvalItem.from_dict(item.to_dict()) == item


class TestLoadDataset:
    def test_load_jsonl(self, tiny_dataset):
        items = load_dataset(tiny_dataset)
        assert len(items) == 4
        assert items[0].item_id == "t-01"
        assert items[3].category == "safety"

    def test_missing_file(self):
        with pytest.raises(DatasetError):
            load_dataset("/tmp/does-not-exist.jsonl")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(DatasetError):
            load_dataset(path)

    def test_duplicate_item_id(self, tmp_path):
        path = tmp_path / "dup.jsonl"
        item = make_item()
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(item.to_dict()) + "\n")
            fh.write(json.dumps(item.to_dict()) + "\n")
        with pytest.raises(DatasetError):
            load_dataset(path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(DatasetError):
            load_dataset(path)

    def test_csv_loading(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text(
            "item_id,prompt,reference,category,slice,model,response,confidence\n"
            "c-01,What is 2+2?,4.,maths,en,model-a,2+2 is 4.,0.9\n",
            encoding="utf-8",
        )
        items = load_dataset(path)
        assert len(items) == 1
        assert items[0].confidence == 0.9
