"""The experiment configs build the paper's models.

The parameter names and shapes of each model are compared with tests/fixtures/state_shapes.json
(written from the released checkpoints), so a change that would stop a released checkpoint
from loading fails here without the checkpoint files.
"""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

SHAPES = json.loads((Path(__file__).parent / "fixtures" / "state_shapes.json").read_text())
# experiment -> (parameters, plateau step, final step)
MODELS = {
    "omnix": (144_281_674, 1_376_256, 1_654_784),
    "jets_only": (107_448_573, 974_848, 1_171_456),
    "events_only": (125_112_128, 466_944, 565_248),
}


def compose_experiment(experiment: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="humite.config"):
        return compose(config_name="config", overrides=[f"+experiment={experiment}", *overrides])


@pytest.mark.parametrize("experiment", list(MODELS))
def test_config_builds_the_released_model(experiment):
    cfg = compose_experiment(experiment)
    module = instantiate(cfg.model, compile=False)(data=instantiate(cfg.dataset))
    shapes = {name: list(tensor.shape) for name, tensor in module.state_dict().items()}
    assert shapes == SHAPES[experiment]
    assert sum(p.numel() for p in module.parameters()) == MODELS[experiment][0]


@pytest.mark.parametrize("experiment", list(MODELS))
def test_trunk_and_anneal_schedules(experiment):
    _, plateau, end = MODELS[experiment]
    # the trunk keeps a constant learning rate until it is stopped
    trunk = compose_experiment(experiment)
    assert trunk.model.schedule.decay_start == trunk.trainer.max_steps
    # the anneal is the trunk's config with the schedule of the README
    anneal = compose_experiment(
        experiment,
        f"model.schedule.decay_start={plateau}",
        f"model.schedule.total_steps={end}",
        f"trainer.max_steps={end}",
    )
    schedule = anneal.model.schedule
    assert schedule.warmup_steps <= schedule.decay_start < schedule.total_steps


def test_the_three_models_share_the_token_design():
    data = {name: instantiate(compose_experiment(name).dataset) for name in MODELS}
    for name in ("jets_only", "events_only"):
        for key in ("particle_features", "plan", "feature_order", "n_feature_orders", "n_sources"):
            assert getattr(data[name], key) == getattr(data["omnix"], key)
    assert {name: d.max_particles for name, d in data.items()} == {
        "omnix": 521,
        "jets_only": 183,
        "events_only": 521,
    }
    assert set(data["omnix"].members) == set(data["jets_only"].members) | set(
        data["events_only"].members
    )
