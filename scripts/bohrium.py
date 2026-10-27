#!/usr/bin/env python3
"""Submit/fetch prepared V100 jobs; credentials are read only from --machine."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from repro_common import ROOT, load_json, safe_extract, sha256, write_json

LOG = "bohrium_reproduction.log"
ERR = "bohrium_reproduction.err"


def machine_for(machine_path, batch):
    from dpdispatcher import Machine

    environment = load_json(ROOT / "metadata/environment.json")
    data = load_json(machine_path)
    profile = data["remote_profile"]
    profile["ignore_exit_code"] = False
    settings = profile.setdefault("input_data", {})
    settings.update(
        machine_type=environment["gpu"],
        image_name=environment["image"],
        grouped=True,
        output_log=False,
        job_name="synth_classification_reproduction",
    )
    settings.pop("backward_files", None)
    os.environ.pop("LBG_CLI_DEBUG_PRINT", None)
    machine = Machine.load_from_dict(data)
    group = batch / "task_group.json"
    if group.exists():
        machine.group_id = int(load_json(group)["job_group_id"])
    original = machine.serialize
    # Submission also serializes internally; never persist remote credentials.
    machine.serialize = lambda if_empty_remote_profile=False: original(if_empty_remote_profile=True)
    return machine


def save(directory, submission):
    state = submission.serialize()
    write_json(directory / "submission_state.json", state)
    job = submission.belonging_jobs[0]
    raw = str(job.job_id)
    job_id, group = raw.split(":job_group_id:") if ":job_group_id:" in raw else (raw, None)
    result = {
        "job_id": job_id,
        "job_group_id": int(group) if group else None,
        "url": f"https://bohrium.dp.tech/jobs/{job_id}",
        "state": getattr(job.job_state, "name", str(job.job_state)),
        "log_file": LOG,
        "credentials_serialized": False,
    }
    write_json(directory / "submission_status.json", result)
    return result


def process(action, directory, machine_path, batch):
    from dpdispatcher import Resources, Submission, Task

    spec = load_json(directory / "submission_spec.json")
    machine = machine_for(machine_path, batch)
    if action == "submit":
        attempt = directory / "submission_attempt.json"
        if attempt.exists():
            raise RuntimeError(f"Prior submission attempt exists: {directory}; use status/fetch")
        for name, expected in spec["forward_sha256"].items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Input changed: {name}")
        resources = Resources.load_from_dict(
            {
                "number_node": 1,
                "cpu_per_node": 12,
                "gpu_per_node": 1,
                "queue_name": "",
                "group_size": 1,
                "source_list": [],
            }
        )
        task = Task(
            command="bash run_job.sh",
            task_work_path=".",
            forward_files=spec["forward_files"],
            backward_files=spec["backward_files"],
            outlog=LOG,
            errlog=ERR,
        )
        submission = Submission(
            work_base=str(directory), machine=machine, resources=resources, task_list=[task]
        )
        write_json(
            attempt,
            {
                "submission_hash": submission.submission_hash,
                "mode": spec["mode"],
                "experiment": spec["experiment"],
            },
        )
        submission.run_submission(exit_on_submit=True, clean="on_success", check_interval=20)
        result = save(directory, submission)
        if result["job_group_id"] is None:
            raise RuntimeError("Bohrium did not return a task group ID")
        group_file = batch / "task_group.json"
        if group_file.exists() and load_json(group_file)["job_group_id"] != result["job_group_id"]:
            raise RuntimeError("Batch tasks must share the same group")
        write_json(group_file, {"job_group_id": result["job_group_id"]})
    else:
        submission = Submission.deserialize(
            load_json(directory / "submission_state.json"), machine=machine
        )
        job = submission.belonging_jobs[0]
        job_id = str(job.job_id).split(":job_group_id:")[0]
        detail = machine.api.get_job_detail(job_id)
        if not detail:
            raise RuntimeError(f"No details for job {job_id}")
        job.job_state = machine.map_dp_job_state(detail["status"], detail.get("exitCode", 0), False)
        result = save(directory, submission)
        if action == "fetch" and result["state"] in ("finished", "terminated"):
            machine._download_job(job)
            archive = directory / "reproduction_result.tar.gz"
            if archive.exists():
                safe_extract(archive, directory / "downloaded")
                result["downloaded"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "status", "fetch"))
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument(
        "--batch", type=Path, required=True, help="Shared state directory for one batch"
    )
    args = parser.parse_args()
    batch = args.batch.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / ".submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = [
            process(args.action, p.resolve(), args.machine.resolve(), batch) for p in args.jobs
        ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
