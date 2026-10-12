import copy
import json
from pathlib import Path

import pytest

from satellite_retrieval.config import load_config, validate_config


@pytest.fixture
def valid_config():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment.json"
    )
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_project_config_loads():
    config = load_config()
    assert config["project_name"] == "satellite-scene-retrieval"
    assert config["dataset"]["num_classes"] == 10
    assert config["encoder"]["embedding_dimension"] == 512


def test_valid_config_passes_validation(valid_config):
    validate_config(valid_config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("dataset", "num_classes", 0, "num_classes"),
        ("dataset", "image_size", 0, "image_size"),
        ("encoder", "embedding_dimension", -1, "embedding_dimension"),
        ("retrieval", "top_k", 0, "top_k"),
        ("reranking", "reciprocal_k", 0, "reciprocal_k"),
        ("retrieval", "evaluation_k", [], "evaluation_k"),
        ("retrieval", "evaluation_k", [1, 5, 5], "duplicates"),
        ("encoder", "normalize_embeddings", False, "normalize_embeddings"),
    ],
)
def test_invalid_settings_raise(valid_config, section, key, value, message):
    config = copy.deepcopy(valid_config)
    config[section][key] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_invalid_alpha_raises(valid_config):
    config = copy.deepcopy(valid_config)
    config["reranking"]["alpha"] = 1.5

    with pytest.raises(ValueError, match="alpha"):
        validate_config(config)


def test_missing_section_raises(valid_config):
    config = copy.deepcopy(valid_config)
    del config["reranking"]

    with pytest.raises(ValueError, match="Missing configuration sections"):
        validate_config(config)
