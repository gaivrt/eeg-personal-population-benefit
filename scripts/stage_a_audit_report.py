"""Aggregate every subject, apply the fixed stop gate, and write section-12 report."""
import csv
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import ROOT, config, failure_stop, participants, save_json, scratch_root, sha256


def main():
    root = scratch_root()
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    rows, gates, metrics = [], {}, []
    accounting_path = out / "slurm_accounting.psv"
    accounting = list(csv.DictReader(accounting_path.open(), delimiter="|")) if accounting_path.exists() else []
    structure_path = root / "receipts/dreyer_raw_structure.json"
    dreyer_structure = {r["subject"]: r for r in json.loads(structure_path.read_text())["subjects"]} if structure_path.exists() else {}
    threshold = config()["audit"]["max_failed_subject_fraction"]
    for dataset in config()["datasets"]:
        path = root / "receipts/preprocess" / f"{dataset}.json"
        receipt = json.loads(path.read_text()) if path.exists() else {"subjects": [], "status": "not_run"}
        subjects = {r["subject"]: r for r in receipt["subjects"]}
        # Include completed smoke audits if a stop interrupts a dataset before
        # its full pass starts. A not-yet-written participant remains not_run.
        for audit_file in (root / "processed" / dataset).glob("sub-*/audit.json"):
            row = json.loads(audit_file.read_text())
            subjects.setdefault(row["subject"], row)
        failed = [s for s, r in subjects.items() if r["status"] != "pass"]
        complete = receipt["status"] == "complete" and set(subjects) == set(participants(dataset))
        gates[dataset] = {"expected": len(participants(dataset)), "audited": len(subjects),
                          "failed": len(failed), "failed_subjects": failed, "complete": complete,
                          "continued_audit_by_user": receipt.get("complete_audit_after_threshold_authorized", False),
                          "failed_fraction_of_cohort": len(failed) / len(participants(dataset)),
                          "stop": failure_stop(len(failed), len(participants(dataset)), threshold)}
        for s in participants(dataset):
            row = subjects.get(s, {"status": "not_run", "failures": []})
            alpha = row.get("alpha", {})
            rest = row.get("rest", {})
            original = dreyer_structure.get(s, {}) if dataset == "Dreyer2023" else {}
            raw_count_source = "processed_trial_labels" if "left_trials" in row else None
            if original and "left_trials" not in row:
                row["left_trials"], row["right_trials"] = original["left_trials"], original["right_trials"]
                raw_count_source = "original_BIDS_events_only"
            if not alpha and dataset == "Dreyer2023":
                alpha = {"status": "not_applicable", "reason": "No occipital O1/Oz/O2 channels"}
            rows.append({"dataset": dataset, "subject": s, "status": row["status"],
                         "failures": ";".join(row["failures"]),
                         "preprocessing_completed": "processed_signal_bytes" in row,
                         "raw_trial_count_source": raw_count_source,
                         "left_trials": row.get("left_trials"), "right_trials": row.get("right_trials"),
                         "usable_left": row.get("usable_left"), "usable_right": row.get("usable_right"),
                         "class_imbalance": row.get("imbalance"),
                         "open_boundary_status": original.get("rest", {}).get("OE", {}).get("boundary_status", "pass" if "open" in rest else None),
                         "closed_boundary_status": original.get("rest", {}).get("CE", {}).get("boundary_status", "pass" if "closed" in rest else None),
                         "open_raw_seconds": rest.get("open", {}).get("duration_seconds", original.get("rest", {}).get("OE", {}).get("duration_seconds")),
                         "open_usable_seconds": rest.get("open", {}).get("usable_seconds"),
                         "closed_raw_seconds": rest.get("closed", {}).get("duration_seconds", original.get("rest", {}).get("CE", {}).get("duration_seconds")),
                         "closed_usable_seconds": rest.get("closed", {}).get("usable_seconds"),
                         "alpha_status": alpha.get("status"), "alpha_reason": alpha.get("reason"),
                         "alpha_open_uv2": alpha.get("open_power_uv2"),
                         "alpha_closed_uv2": alpha.get("closed_power_uv2"),
                         "alpha_closed_open_ratio": alpha.get("closed_open_ratio"),
                         "cho_order": row.get("checks", {}).get("cho_order_mapping_and_signal_roundtrip"),
                         "error": row.get("error", "")})
        download_path = root / "receipts/download" / dataset / "run.json"
        download = json.loads(download_path.read_text()) if download_path.exists() else {}
        raw_dir = root / "raw" / {"PhysionetMI": "MNE-eegbci-data", "Dreyer2023": "MNE-dreyer2023-data",
                                 "Cho2017": "MNE-gigadb-data"}[dataset]
        raw_files = [p for p in raw_dir.rglob("*") if p.is_file()]
        if dataset == "Dreyer2023":
            payload = [p for p in raw_files if p.parent == raw_dir]
        else:
            payload = [p for p in raw_files if p.suffix == (".edf" if dataset == "PhysionetMI" else ".mat")]
        completed = len(list(download_path.parent.glob("subject-*.json")))
        attempt_files = list(download_path.parent.glob("attempt-*.json"))
        attempts = [json.loads(p.read_text()) for p in attempt_files] + ([download] if download else [])
        starts = [datetime.fromisoformat(a["utc"]) for a in attempts if "utc" in a]
        completed_at = (datetime.fromisoformat(download["utc"]) + timedelta(seconds=download["elapsed_seconds"])) if download.get("status") == "complete" else None
        array_index = list(config()["datasets"]).index(dataset)
        allocation_rows = [r for r in accounting if "." not in r["JobID"] and r["JobID"].endswith(f"_{array_index}")]
        download_jobs = [r for r in allocation_rows if r["JobName"] == "eeg-A-download"]
        preprocess_jobs = [r for r in allocation_rows if r["JobName"] == "eeg-A-preprocess"]
        signal_files = [p for p in (root / "processed" / dataset).rglob("*.npy") if p.parent.name.startswith("sub-")]
        task_bytes = sum(p.stat().st_size for p in signal_files if p.name == "tasks.npy")
        rest_bytes = sum(p.stat().st_size for p in signal_files if p.name in ("rest_open.npy", "rest_closed.npy"))
        metrics.append({"dataset": dataset, "download_status": download.get("status", "not_run"),
                        "completed_subject_download_receipts": completed,
                        "download_payload_bytes": sum(p.stat().st_size for p in payload),
                        "raw_disk_bytes": sum(p.stat().st_size for p in raw_files),
                        "download_first_attempt_utc": min(starts).isoformat() if starts else None,
                        "download_completed_utc": completed_at.isoformat() if completed_at else None,
                        "download_wall_seconds_first_to_complete": (completed_at - min(starts)).total_seconds() if completed_at and starts else None,
                        "wall_seconds_first_download_to_report": (datetime.now(timezone.utc) - min(starts)).total_seconds() if starts else None,
                        "download_elapsed_seconds_latest_attempt": download.get("elapsed_seconds"),
                        "download_slurm_elapsed_seconds_sum": sum(int(r["ElapsedRaw"]) for r in download_jobs),
                        "download_slurm_job_ids": [r["JobID"] for r in download_jobs],
                        "preprocessing_elapsed_seconds": receipt.get("elapsed_seconds"),
                        "preprocessing_slurm_elapsed_seconds_sum": sum(int(r["ElapsedRaw"]) for r in preprocess_jobs),
                        "preprocessing_slurm_job_ids": [r["JobID"] for r in preprocess_jobs],
                        "preprocessing_subject_seconds_sum": sum(r.get("seconds", 0) for r in receipt.get("subjects", [])),
                        "task_signal_bytes": task_bytes, "rest_signal_bytes": rest_bytes,
                        "npy_total_bytes": sum(p.stat().st_size for p in signal_files),
                        "processed_bytes": sum(r.get("processed_signal_bytes", 0) for r in subjects.values()),
                        "download_git_commit": download.get("git_commit"),
                        "preprocessing_git_commit": receipt.get("git_commit")})
        card = [f"# {dataset} 阶段 A 数据卡", "",
                f'预定 {len(participants(dataset))} 人；下载回执 {completed} 人；已检查 {len(subjects)} 人。',
                f'原始 {config()["datasets"][dataset]["sfreq"]} Hz、{config()["datasets"][dataset]["channels"]} EEG 通道；预处理 200 Hz、float16、μV/100。',
                "", "| 被试 | 状态 | 左/右原始试次 | 左/右有效试次 | 睁眼原始/有效秒 | alpha | 失败原因 |",
                "|---|---|---|---|---|---|---|"]
        for s, r in sorted(subjects.items()):
            rest = r.get("rest", {}).get("open", {})
            card.append(f'| {s} | {r["status"]} | {r.get("left_trials", "—")}/{r.get("right_trials", "—")} | '
                        f'{r.get("usable_left", "—")}/{r.get("usable_right", "—")} | '
                        f'{rest.get("duration_seconds", "—")}/{rest.get("usable_seconds", "—")} | '
                        f'{r.get("alpha", {}).get("status", "—")} | {"; ".join(r["failures"])} |')
        card += ["", "完整未检查名单、功率与试次数见 ../subject_audit.csv；未检查被试不得视为通过。",
                 "原始边界、来源 SHA256、完整片段/尾段/质量筛选时长见逐被试 metadata.json。"]
        metadata_files = sorted((root / "processed" / dataset).glob("sub-*/metadata.json"))
        if metadata_files:
            metadata = json.loads(metadata_files[0].read_text())
            card += ["", "## 通道与来源", "", "最终通道（保存顺序）：" + ", ".join(metadata["channels"]) + "。",
                     "单 session；任务提示、原始试次编号和质量掩码保存在每人的 trials.csv。",
                     "全部原始文件路径、SHA256、静息起止、采样率、单位和配置哈希保存在每人的 metadata.json。"]
        (out / "data_cards").mkdir(exist_ok=True)
        (out / "data_cards" / f"{dataset}.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    with (out / "subject_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stop = any(g["stop"] for g in gates.values())
    complete = all(g["complete"] for g in gates.values())
    gate = {"datasets": gates, "threshold": threshold, "stop": stop,
            "audit_complete": complete, "training_allowed": complete and not stop,
            "continued_audit_by_user": any(g["continued_audit_by_user"] for g in gates.values()),
            "config_sha256": sha256(ROOT / "configs/stage_a.yaml")}
    save_json(out / "audit_gate.json", gate)
    save_json(out / "resource_metrics.json", metrics)
    lines = ["# 阶段 A：数据检查报告", "", "## 做了什么", "",
             "按固定起步池逐文件处理，结果以 float16 保存于 scratch。下表按实际回执填写；未运行项不填估算值。",
             "", "## 与规格的偏离", "",
             "- 本地没有远程仓库，服务器通过 Git bundle fast-forward 拉取源码提交。",
             "- 侧边终端不可用，使用工具终端；Slurm、tmux 与逐项 JSON 日志保留。",
             "- Dreyer/Cho 没有 CBraMod 官方专用预处理脚本，项目扩展见预处理协议。",
             "- Cho 缺独立连续事件时间线及 rest→MI 绝对间隔；全量顺序检查不消除此限制。",
             "", "## 结果表", "",
             "| 数据集 | 预定人数 | 已检查 | 未通过 | 未通过/预定人数 | 超过 10% |",
             "|---|---:|---:|---:|---:|---|"]
    for d, g in gates.items():
        lines.append(f'| {d} | {g["expected"]} | {g["audited"]} | {g["failed"]} | {g["failed_fraction_of_cohort"]:.2%} | {g["stop"]} |')
    lines += ["", "[逐被试检查与试次数 CSV](subject_audit.csv)。",
              "未检查行明确标为 not_run；有效样本已不足时，不对不存在的分类结果计算均值与标准差。",
              "", "| 数据集 | 已完成下载回执 | 完整下载文件 GiB | 原始磁盘占用 GiB | 结果数组 MiB | 逐被试处理秒合计 |",
              "|---|---:|---:|---:|---:|---:|"]
    for m in metrics:
        lines.append(f'| {m["dataset"]} | {m["completed_subject_download_receipts"]} | '
                     f'{m["download_payload_bytes"]/2**30:.3f} | {m["raw_disk_bytes"]/2**30:.3f} | '
                     f'{m["npy_total_bytes"]/2**20:.3f} | {m["preprocessing_subject_seconds_sum"]:.2f} |')
    lines += ["", "下载体积按实际文件计；Dreyer 原始磁盘占用包含 ZIP 和解压文件。中断与重试的实际运行时间另见 Slurm 记账。",
              "", "| 条件 | PhysionetMI 均值±SD | Dreyer2023 均值±SD | Cho2017 均值±SD | 合并 |",
              "|---|---|---|---|---|", "| B0 线性 / MLP | 未训练 | 未训练 | 未训练 | 未训练 |",
              "| B1a / B1b | 未训练 | 未训练 | 未训练 | 未训练 |",
              "| EA + 切空间 + LR | 未训练 | 未训练 | 未训练 | 未训练 |",
              "", "下载、预处理体积和耗时见 [resource_metrics.json](resource_metrics.json)。",
              "特征缓存未运行；缓存体积、耗时、显存峰值均不可报告。", "",
              "## 统计检验", "", "尚无基线预测，未运行配对 Wilcoxon 或 Holm 校正。",
              "", "## 发现的问题", ""]
    for d, g in gates.items():
        lines.append(f'- {d} 未通过被试：{g["failed_subjects"]}。详见 CSV 中失败原因和原始错误。')
    lines += ["", "## 下一步建议", "",
              "原 10% 数据质量门槛已触发；按用户后续指令继续补齐数据检查，失败标记与阈值保留。该报告器不提交缓存或训练，不进入阶段 B。"
              if stop else ("数据检查已完成且未触发停止条件，可继续本次已授权的阶段 A；不得进入阶段 B。"
                            if complete else "检查尚未完成；不能将待处理被试计为通过，也不能启动训练。")]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
