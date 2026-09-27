"""Stage C3: neighbour selection and parameter averaging over C2 personal parameters."""
import numpy as np
import torch
import yaml
from .stage_a_common import ROOT


def config_c3():
    return yaml.safe_load((ROOT / "configs/stage_c3.yaml").read_text())


def average_personal(variant, params):
    """FiLM offsets: plain mean. LoRA: mean of B·A per module, as one rank-(r·k) LoRA."""
    k = len(params)
    if variant == "film_offset":
        return [torch.stack([p[0] for p in params]).mean(0)]
    merged = []
    for index in range(0, len(params[0]), 2):  # PersonalModel order per module: a, b
        merged.append(torch.cat([p[index] for p in params], dim=1))
        merged.append(torch.cat([p[index + 1] for p in params], dim=2) / k)
    return merged


def nearest(features, subject, candidates, k):
    """k nearest candidates by Euclidean distance; ties go to the smaller key, e.g. (dataset, subject id)."""
    order = sorted(candidates, key=lambda c: (float(np.linalg.norm(features[subject] - features[c])), c))
    return order[:k]


def random_sets(rng, candidates, k, draws):
    """Draw indices, not items: Generator.choice would turn tuples into rows of strings."""
    pool = sorted(candidates)
    return [sorted(pool[i] for i in rng.choice(len(pool), size=k, replace=False)) for _ in range(draws)]
