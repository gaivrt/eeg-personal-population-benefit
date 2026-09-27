from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, lfilter, resample


@dataclass(frozen=True)
class Window:
    dataset: str
    subject: int
    session: int
    start: float
    end: float
    trial_id: str

    @property
    def person(self):
        return self.dataset, self.subject


def assert_split(train, val, test):
    train, val, test = set(train), set(val), set(test)
    assert train and val and test, "empty split"
    assert not (train & val or train & test or val & test), "subject leakage"


def assert_episode(context, query, *, cross_session=False):
    assert context and query
    assert len({w.person for w in context + query}) == 1
    assert len({w.session for w in context}) == len({w.session for w in query}) == 1
    if cross_session:
        assert context[0].session < query[0].session
    else:
        assert context[0].session == query[0].session
    assert max(w.end for w in context) < min(w.start for w in query), "temporal leakage"
    assert not ({w.trial_id for w in context} & {w.trial_id for w in query}), "trial leakage"


def make_splits(people, folds, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for dataset, subject in sorted(people):
        groups.setdefault(dataset, []).append((dataset, subject))
    partitions = {}
    for dataset, group in groups.items():
        order = rng.permutation(len(group))
        partitions[dataset] = [list(map(tuple, np.array(group, dtype=object)[idx]))
                               for idx in np.array_split(order, folds)]
    result = []
    for fold in range(folds):
        test, val, train = [], [], []
        for dataset, parts in partitions.items():
            test.extend(parts[fold])
            remaining = [p for i, part in enumerate(parts) if i != fold for p in part]
            rng.shuffle(remaining)
            nval = max(1, round(len(remaining) * validation_fraction))
            val.extend(remaining[:nval])
            train.extend(remaining[nval:])
        assert_split(train, val, test)
        assert set(train + val + test) == set(people)
        result.append((train, val, test))
    return result


def other_subject(person, pool, rng):
    candidates = sorted(p for p in pool if p[0] == person[0] and p != person)
    if not candidates:
        raise ValueError("No different subject in the same dataset and split")
    return candidates[int(rng.integers(len(candidates)))]


def preprocess_synthetic(x, sfreq=200):
    """BCIC-IV-2a-style common average + order-5 causal 0.3–40 Hz.

    x is already isolated synthetic EEG in microvolts. No real-data protocol
    is inferred here. Context and query are always filtered independently.
    """
    x = x - x.mean(axis=-2, keepdims=True)
    b, a = butter(5, [0.3, 40], btype="bandpass", fs=sfreq)
    x = lfilter(b, a, x, axis=-1)
    if sfreq != 200:
        x = resample(x, round(x.shape[-1] * 200 / sfreq), axis=-1)
    return (x / 100).astype(np.float32)


def synthetic_sessions(cfg):
    """Only this function creates training signals. Never loads audit files."""
    c = cfg["synthetic"]
    rng = np.random.default_rng(cfg["data_seed"])
    result = {}
    nch, fs = len(c["channels"]), c["sfreq"]
    assert nch == 3, "Stage 0a synthetic generator uses C3/Cz/C4"
    for dataset in c["datasets"]:
        for subject in range(1, c["subjects_per_dataset"] + 1):
            gain = rng.uniform(0.6, 1.5, (nch, 1))
            for session in range(1, c["sessions"] + 1):
                base = (session - 1) * 86400.0
                nctx = c["context_seconds"] // c["segment_seconds"]
                tc = np.arange(c["segment_seconds"] * fs) / fs
                context = gain * (10 * rng.standard_normal((nctx, nch, len(tc))) +
                                  15 * np.sin(2 * np.pi * 10 * tc + rng.uniform(0, 6.28, (nctx, nch, 1))))
                nquery = c["query_trials"]
                y = np.tile([0, 1], nquery // 2)
                rng.shuffle(y)
                tq = np.arange(c["query_seconds"] * fs) / fs
                amplitude = np.array([[30, 10, 8] if label == 0 else [8, 10, 30] for label in y])
                query = gain * (10 * rng.standard_normal((nquery, nch, len(tq))) +
                                amplitude[..., None] * np.sin(2 * np.pi * 10 * tq +
                                rng.uniform(0, 6.28, (nquery, nch, 1))))
                cw = [Window(dataset, subject, session, base + i * c["segment_seconds"],
                             base + (i + 1) * c["segment_seconds"], f"{dataset}:{subject}:{session}:rest:{i}")
                      for i in range(nctx)]
                qw = [Window(dataset, subject, session, base + c["context_seconds"] + 2 + 6 * i,
                             base + c["context_seconds"] + 2 + 6 * i + c["query_seconds"],
                             f"{dataset}:{subject}:{session}:trial:{i}") for i in range(nquery)]
                assert_episode(cw, qw)
                result[(dataset, subject, session)] = {
                    "context": preprocess_synthetic(context, fs), "query": preprocess_synthetic(query, fs),
                    "labels": y, "context_windows": cw, "query_windows": qw,
                }
    return result
