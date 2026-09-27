"""The single precision exception must leave the historical path untouched."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from subject_context.stage_c import backbone_features


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Identity()
        self.encoder = SimpleNamespace(layers=nn.ModuleList([nn.Identity() for _ in range(12)]))


def test_historical_forward_rounding_is_unchanged():
    x = torch.linspace(-1.234567, 2.876543, 800).reshape(1, 2, 400)
    expected = x.reshape(1, 2, 2, 200).half().float().mean(2).flatten(1)
    torch.testing.assert_close(backbone_features(Core(), x, [0, 1]), expected, rtol=0, atol=0)


def test_fp32_exception_avoids_fp16_forward_and_backward_overflow():
    core = Core()
    large = torch.full((1, 1, 200), 70000., requires_grad=True)
    assert not torch.isfinite(backbone_features(core, large, [0])).all()
    core.budget_retry_fp32 = True
    stable = backbone_features(core, large, [0])
    assert stable.dtype == torch.float32 and torch.isfinite(stable).all()
    (stable.sum()*100000.).backward()
    assert torch.isfinite(large.grad).all()
    torch.testing.assert_close(stable, large.reshape(1, 200), rtol=0, atol=0)


def test_exception_survives_personal_deepcopy_but_does_not_modify_source():
    source = Core()
    population = copy.deepcopy(source)
    population.budget_retry_fp32 = True
    personal = copy.deepcopy(population)
    x = torch.full((1, 1, 200), .123456)
    assert not getattr(source, 'budget_retry_fp32', False)
    torch.testing.assert_close(backbone_features(population, x, [0]), backbone_features(personal, x, [0]), rtol=0, atol=0)


def test_retry_authorization_rejects_other_runs_and_repeat_attempts():
    path = Path(__file__).resolve().parents[1]/'scripts/population_budget_run.py'
    spec = importlib.util.spec_from_file_location('budget_runner_for_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    auth = dict(user_confirmed=True, scope='population_budget_CB_retry_once', array_index=57,
                maximum_attempts=1, numerical_policy='fp32_layer8_no_rounding')
    assert module.retry_authorized(auth, 57)
    for wrong, index in [({**auth,'maximum_attempts':2},57),(auth,56),({**auth,'user_confirmed':False},57)]:
        with pytest.raises(ValueError):
            module.retry_authorized(wrong,index)
    assert not module.retry_authorized(dict(user_confirmed=True,scope='population_budget_75_runs'),0)
