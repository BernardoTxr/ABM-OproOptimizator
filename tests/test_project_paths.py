from pathlib import Path

import yaml

from lob_sim.paths import find_project_root


ROOT = Path(__file__).resolve().parents[1]


def test_nested_paper_config_finds_repository_root():
    assert find_project_root(ROOT / "configs/paper/optimizer.yaml") == ROOT


def test_checked_in_config_references_exist():
    references = {
        "configs/paper/scenario.yaml": [
            "agents_yaml", "environment_yaml",
        ],
        "configs/paper/optimizer.yaml": [
            "base_config", "real_config", "loss_config",
        ],
        "configs/paper/later-evaluation.yaml": [
            "base_config", "real_config", "loss_config",
        ],
        "configs/examples/scenario.yaml": [
            "agents_yaml", "environment_yaml",
        ],
    }
    for relative_path, keys in references.items():
        config_path = ROOT / relative_path
        values = yaml.safe_load(config_path.read_text())
        for key in keys:
            target = (config_path.parent / values[key]).resolve()
            assert target.is_file(), f"{relative_path}:{key} -> {target}"
