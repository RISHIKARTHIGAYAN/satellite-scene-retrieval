"""Load and validate experiment configuration."""

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load experiment settings from a JSON file and validate them."""
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = json.load(file)

    validate_config(config)
    return config


def _positive_integer(value: Any, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{field}' must be a positive integer.")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the settings required by the retrieval experiments."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    required_sections = {"dataset", "encoder", "retrieval", "reranking"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("'seed' must be an integer.")

    dataset = config["dataset"]
    encoder = config["encoder"]
    retrieval = config["retrieval"]
    reranking = config["reranking"]

    for name, section in (
        ("dataset", dataset),
        ("encoder", encoder),
        ("retrieval", retrieval),
        ("reranking", reranking),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"'{name}' must be an object.")

    _positive_integer(dataset.get("num_classes"), "dataset.num_classes")
    _positive_integer(dataset.get("image_size"), "dataset.image_size")
    _positive_integer(
        encoder.get("embedding_dimension"), "encoder.embedding_dimension"
    )
    _positive_integer(retrieval.get("top_k"), "retrieval.top_k")
    _positive_integer(reranking.get("reciprocal_k"), "reranking.reciprocal_k")

    if retrieval.get("similarity") != "cosine":
        raise ValueError(
            "This implementation currently supports cosine similarity only."
        )

    evaluation_k = retrieval.get("evaluation_k")
    if (
        not isinstance(evaluation_k, list)
        or not evaluation_k
        or any(
            not isinstance(k, int) or isinstance(k, bool) or k <= 0
            for k in evaluation_k
        )
    ):
        raise ValueError(
            "'retrieval.evaluation_k' must be a non-empty list of positive integers."
        )

    if len(set(evaluation_k)) != len(evaluation_k):
        raise ValueError("'retrieval.evaluation_k' must not contain duplicates.")

    alpha = reranking.get("alpha")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 <= alpha <= 1
    ):
        raise ValueError("'reranking.alpha' must be between 0 and 1.")

    if encoder.get("normalize_embeddings") is not True:
        raise ValueError(
            "'encoder.normalize_embeddings' must be true for cosine retrieval."
        )
