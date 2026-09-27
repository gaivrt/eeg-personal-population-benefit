import importlib.util
from pathlib import Path
import pytest
from subject_context.stage_a_common import ROOT


def loader():
    spec = importlib.util.spec_from_file_location("lee_prepare", ROOT/"scripts/stage_w_lee_prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_excludes_other_sessions_and_paradigms():
    module = loader()
    xml = (ROOT/"docs/data_access_audit_2026-09-25/sources/lee_object_list.xml").read_bytes()
    rows = module.manifest(xml)
    assert len(rows) == 54 and sum(r["bytes"] for r in rows) == 32738728671
    assert all("/session1/" in r["url"] and r["url"].endswith("_EEG_MI.mat") for r in rows)
    with pytest.raises(AssertionError):
        module.manifest(xml.replace(b"<IsTruncated>false</IsTruncated>",b"<IsTruncated>true</IsTruncated>"))
