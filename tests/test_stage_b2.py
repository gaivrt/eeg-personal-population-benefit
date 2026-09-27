import numpy as np
import torch
from subject_context.model import load_backbone, state_hash
from subject_context.stage_a_common import ROOT
from subject_context.stage_b import Readout
from subject_context.stage_b2 import FullAdapter, config_b2


def test_full_forward_zero_identity_gradient_scope_and_parameter_counts():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    backbone = load_backbone(ROOT / 'weights/cbramod.pth', 4)
    readout = Readout({'head': torch.nn.Linear(600, 2).state_dict(),
        'feature_mean': np.zeros(600), 'feature_std': np.ones(600)}, 'linear')
    eeg = torch.randn(2, 3, 400)
    expected = readout(backbone.suffix(backbone.prefix(eeg).half().float()).mean(2).flatten(1))
    frozen = state_hash(backbone), state_hash(readout)
    counts = {'film_first2': 800, 'film_all': 4800, 'lora4': 76800, 'lora8': 153600, 'scratch_head': 1202}
    for variant in config_b2()['variants']:
        model = FullAdapter(backbone, readout, [0, 1, 2], variant)
        initial = model(eeg)
        if variant == 'scratch_head':
            assert not torch.equal(initial, expected)
        else:
            with torch.no_grad():  # scoring path: bit-identical
                torch.testing.assert_close(model(eeg), expected, atol=0, rtol=0)
            # Grad-mode CPU kernels can differ at rounding level across machines and thread counts.
            torch.testing.assert_close(initial, expected, atol=1e-5, rtol=1e-5)
        assert sum(p.numel() for p in model.trainable()) == counts[variant]
        before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW(model.trainable(), lr=.01)
        torch.nn.functional.cross_entropy(initial, torch.tensor([0, 1])).backward()
        opt.step()
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
        assert (state_hash(backbone), state_hash(readout)) == frozen
        assert all(p.grad is None for p in backbone.parameters())
        if variant.startswith('lora'):
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            assert len(trainable) == 12 * 2 * 2 * 2
            assert all('parametrizations' in n and 'self_attn_' in n for n in trainable)
            assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)


def test_scratch_initialization_reproducible_and_independent_of_b0_head():
    backbone = load_backbone(ROOT / 'weights/cbramod.pth', 4)
    readout = Readout({'head': torch.nn.Linear(600, 2).state_dict(),
        'feature_mean': np.zeros(600), 'feature_std': np.ones(600)}, 'linear')
    torch.manual_seed(11)
    first = FullAdapter(backbone, readout, [0, 1, 2], 'scratch_head')
    with torch.no_grad():
        readout.head.weight.add_(100)
        readout.head.bias.add_(100)
    torch.manual_seed(11)
    second = FullAdapter(backbone, readout, [0, 1, 2], 'scratch_head')
    assert state_hash(first.readout) == state_hash(second.readout)
