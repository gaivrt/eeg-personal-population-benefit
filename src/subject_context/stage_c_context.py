"""Label-free context features, episode sampling helpers and the T3A control."""
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from scipy.signal import butter, sosfiltfilt
import torch
from torch.nn import functional as F
from .stage_a_features import covariances


def band_covariances(segments, bands, order=4, sfreq=200):
    """segments: (n, channels, samples) of the readout channels; one covariance set per band."""
    x = np.asarray(segments, dtype=np.float64)
    result = []
    for low, high in bands:
        sos = butter(order, [low, high], fs=sfreq, btype="bandpass", output="sos")
        result.append(covariances(sosfiltfilt(sos, x, axis=-1)))
    return result


class TangentFeatures:
    """Reference point and scaling are fitted on training-subject rest only."""
    def fit(self, band_sets):
        self.references = [mean_riemann(np.concatenate(covs)) for covs in zip(*band_sets)]
        vectors = np.concatenate([self._raw(bands) for bands in band_sets])
        self.mean, self.std = vectors.mean(0), vectors.std(0) + 1e-8
        return self

    def _raw(self, bands):
        return np.concatenate([tangent_space(c, r) for c, r in zip(bands, self.references)], 1)

    def __call__(self, bands):
        return ((self._raw(bands) - self.mean) / self.std).astype(np.float32)


def context_window(valid_count, rng, min_segments):
    """Contiguous run of usable segments with a uniformly drawn length."""
    assert valid_count >= min_segments, "usable eyes-open rest shorter than the minimum"
    n = int(rng.integers(min_segments, valid_count + 1))
    start = int(rng.integers(0, valid_count - n + 1))
    return start, n


def derangement(people, rng):
    """Same-dataset donor for every test subject, never the subject itself."""
    donors = {}
    for dataset in sorted({d for d, _ in people}):
        group = sorted(p for p in people if p[0] == dataset)
        assert len(group) > 1, f"{dataset}: no other test subject for shuffled context"
        while True:
            order = rng.permutation(len(group))
            if all(i != j for i, j in enumerate(order)):
                break
        donors.update({group[i]: group[j] for i, j in enumerate(order)})
    return donors


def head_parts(readout):
    """Split the B0 head into T3A's feature map and final linear classifier."""
    head = readout.head
    if isinstance(head, torch.nn.Linear):
        return lambda f: readout.normalize(f), head
    return lambda f: head[1](head[0](readout.normalize(f))), head[2]


@torch.no_grad()
def t3a_predict(readout, support_features, query_features, filter_size):
    """T3A with rest segments as the only unlabeled supports; queries never update prototypes."""
    features, classifier = head_parts(readout)
    supports = torch.cat([classifier.weight, features(support_features)])
    logits = classifier(supports)
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    labels = logits.argmax(1)
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for k in range(classifier.out_features):
        members = torch.nonzero(labels == k).flatten()
        if filter_size != "all":
            members = members[entropy[members].argsort()[:filter_size]]
        keep[members] = True
    one_hot = F.one_hot(labels[keep], classifier.out_features).float()
    prototypes = F.normalize(F.normalize(supports[keep], dim=1).T @ one_hot, dim=0)
    return (features(query_features) @ prototypes).argmax(1).cpu().numpy()
