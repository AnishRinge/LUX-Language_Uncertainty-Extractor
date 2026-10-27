#!/usr/bin/env python
"""Autonomous production evaluation runner for LUX Phase 2.

Processes ONE generation at a time using the existing production CLI,
inspects each result, and stops immediately on quota exhaustion or
persistent errors. Preserves all existing records and checkpoints.

Progress log:  data/production/evaluation_runner_log.json
Report:        docs/production_evaluation_runner_report.md

Usage:
    python run_eval_autonomous.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.hashing import compute_sha256

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
LOG_PATH = ROOT / "data" / "production" / "evaluation_runner_log.json"
REPORT_PATH = ROOT / "docs" / "production_evaluation_runner_report.md"
EVAL_DIR = ROOT / "data" / "production" / "evaluations"
GEN_PATH = ROOT / "data" / "production" / "generations.json"
MAX_EVALS_PER_BATCH = 1
SPACING_SECONDS = 12.0
# No iteration cap: the governing bound is the wall-clock session budget below.
# A hard iteration cap would silently truncate a long queue; the time budget is
# the only safe universal bound for a 2-3 hour production session.
MAX_ITERATIONS = None
# Preferred hard session budget: 2h 30m. Overridable via --max-seconds.
SESSION_BUDGET_SECONDS = 2 * 3600 + 30 * 60
BATCH_STOP_ERROR_TYPES = {
    "QuotaExhaustedError",
    "PersistentEvaluationError",
    "EvaluatorConfigError",
}
# Error types that indicate persistent non-retryable failures after
# the built-in retry policy exhausted its budget.
PERSISTENT_ROOT_CAUSES = {"401", "403", "400", "404", "InvalidArgument", "NotFound"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------
def _count_eval_statuses() -> dict:
    counts = {"SUCCESS": 0, "FAILED": 0, "MANUAL_REVIEW": 0, "UNKNOWN": 0}
    if not EVAL_DIR.exists():
        return counts
    for ef in sorted(EVAL_DIR.glob("*.json")):
        try:
            data = json.load(open(ef, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            counts["UNKNOWN"] += 1
            continue
        s = data.get("evaluation_status", "UNKNOWN")
        counts[s] = counts.get(s, 0) + 1
    return counts


def _count_pending() -> int:
    """Count generations without a SUCCESS or MANUAL_REVIEW eval file."""
    if not GEN_PATH.exists():
        return 0
    artifact = json.load(open(GEN_PATH, "r", encoding="utf-8"))
    gens = artifact.get("generations", [])
    pending = 0
    from src.evaluation.schemas import EvaluationStatus
    terminal = {s.value for s in [
        EvaluationStatus.SUCCESS, EvaluationStatus.MANUAL_REVIEW
    ]}
    for gen in gens:
        from run_production import _generation_evaluation_path
        eval_path = _generation_evaluation_path(gen["generation_id"])
        if eval_path.exists():
            try:
                data = json.load(open(eval_path, "r", encoding="utf-8"))
                if data.get("evaluation_status") in terminal:
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        pending += 1
    return pending


def _snapshot_eval_files() -> dict:
    """Snapshot (path -> mtime) of every eval file currently on disk.

    Used to distinguish a *new* evaluation produced by the current subprocess
    invocation from a pre-existing record.  Without this, the runner can mistake
    the most-recently-modified historical file (e.g. the controlled SUCCESS
    record) for the result of a run that actually crashed with exit_code=1.
    """
    snapshot: dict = {}
    if EVAL_DIR.exists():
        for f in EVAL_DIR.glob("*.json"):
            try:
                snapshot[str(f)] = f.stat().st_mtime
            except OSError:
                pass
    return snapshot


def _get_newest_eval_after(snapshot: dict) -> dict:
    """Return the content of the newest eval file created/modified after *snapshot*.

    Returns an empty dict when no new eval file was produced — which is the
    correct signal that the subprocess did not persist a result.
    """
    if not EVAL_DIR.exists():
        return {}
    files = [
        f for f in EVAL_DIR.glob("*.json")
        if str(f) not in snapshot or f.stat().st_mtime > snapshot[str(f)]
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            return json.load(open(f, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return {}


def _attempts_this_run(log: dict, generation_id: str, attempts: list) -> list:
    """Exclude prior persisted attempts when reporting this invocation's calls."""
    previous_counts = [
        entry.get("attempts_recorded", entry.get("attempts", 0))
        for entry in log.get("iterations", [])
        if entry.get("gen_id") == generation_id
    ]
    previous_count = max(previous_counts, default=0)
    return attempts[previous_count:]


def _get_most_recent_eval() -> dict:
    """Return the content of the most recently modified eval file.

    Kept for backward compatibility with the startup quota/service check, which
    inspects the latest artifact regardless of when it was written.
    """
    if not EVAL_DIR.exists():
        return {}
    files = sorted(EVAL_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            return json.load(open(f, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return {}


def _get_most_recent_eval_gen_id() -> str:
    """Return the source_generation_id of the most recent eval file."""
    if not EVAL_DIR.exists():
        return "?"
    files = sorted(EVAL_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            data = json.load(open(f, "r", encoding="utf-8"))
            return data.get("source_generation_id", f.stem)
        except (json.JSONDecodeError, OSError):
            continue
    return "?"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _resolve_venv_python() -> str:
    """Return the path to the virtual-environment Python interpreter.

    The production CLI imports ``google.genai`` and other project dependencies
    that live ONLY in the project venv.  ``sys.executable`` is unreliable: when
    the runner is launched under the system interpreter (e.g. ``python`` on
    PATH), the spawned subprocess inherits that interpreter and fails with
    ``EvaluatorConfigError: google.genai SDK is not installed`` even though the
    venv is fully populated.  Prefer the venv interpreter explicitly, falling
    back to ``sys.executable`` only when no venv is present.
    """
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return sys.executable


VENV_PYTHON = _resolve_venv_python()


def _update_unusable_slots(record: dict, known_slots: dict) -> dict:
    """Track credential-specific failures and clear slots after later success."""
    updated = dict(known_slots)
    for attempt in record.get("attempts", []):
        slot = attempt.get("key_slot")
        if not slot:
            continue
        if attempt.get("status") == "SUCCESS":
            updated.pop(slot, None)
            continue
        error_type = attempt.get("error_type", "")
        message = (attempt.get("error_message") or "").lower()
        credential_specific = (
            error_type in ("QuotaExhaustedError", "CredentialAccessError")
            or (
                error_type == "PersistentEvaluationError"
                and any(code in message for code in ("401", "403"))
            )
            or "no longer available to new users" in message
        )
        if credential_specific:
            updated[slot] = error_type or "CredentialAccessError"
    return updated


def _configured_credential_slots() -> set:
    """Load repository env settings and return configured slot names only."""
    from run_production import _load_env_file
    from src.evaluation.gemini_evaluator import discover_api_credentials

    _load_env_file()
    return {slot for slot, _ in discover_api_credentials()}


def run_one_evaluation(known_unusable_slots=None) -> dict:
    """Run one evaluation batch (max 1 generation) via the production CLI."""
    cmd = [
        VENV_PYTHON, str(ROOT / "run_production.py"),
        "evaluate",
        "--max-evals", str(MAX_EVALS_PER_BATCH),
        "--spacing", str(SPACING_SECONDS),
    ]
    child_env = os.environ.copy()
    if known_unusable_slots:
        child_env["GEMINI_SKIP_CREDENTIAL_SLOTS"] = ",".join(
            sorted(known_unusable_slots)
        )
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
        env=child_env,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:] if proc.stdout else "",
        "stderr": proc.stderr[-2000:] if proc.stderr else "",
    }


def should_stop_for_error(eval_record: dict) -> tuple:
    """Check if the eval record indicates we should stop the autonomous run.

    Returns (should_stop, reason).
    """
    error_type = eval_record.get("error_type", "")
    status = eval_record.get("evaluation_status", "")
    attempts = eval_record.get("attempts", [])

    if status in ("SUCCESS", "MANUAL_REVIEW"):
        return False, ""
    if status != "FAILED":
        return False, ""

    last_attempt = attempts[-1] if attempts else {}
    last_error_type = last_attempt.get("error_type", "")
    last_error_message = last_attempt.get("error_message", "").lower()

    # A successful fallback may contain earlier quota errors in attempts. Only
    # classify the terminal error, so an earlier key's quota cannot mask it.
    if error_type == "QuotaExhaustedError" or last_error_type == "QuotaExhaustedError":
        return True, "quota_exhausted"

    if error_type in PERSISTENT_ROOT_CAUSES:
        return True, "persistent_error"

    if last_error_type in ("PersistentEvaluationError", "EvaluatorConfigError"):
        return True, "persistent_error"
    if any(code in last_error_message for code in ("401", "403", "400", "404")):
        return True, "persistent_error"

    # Server 503/504 errors that exhausted the bounded retry policy should stop
    # the autonomous run. The evaluator already retried up to max_retries=3.
    if error_type in ("ServerError", "ServiceUnavailable", "InternalServerError"):
        # If all attempts show failures (retries exhausted), stop.
        if attempts and all(a.get("status") == "RETRY" for a in attempts):
            return True, "service_unavailable"

    return False, ""


def load_log() -> dict:
    if LOG_PATH.exists():
        try:
            return json.load(open(LOG_PATH, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "started_at": _now_iso(),
        "iterations": [],
        "stop_reason": None,
        "stopped_at": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous production evaluation runner for LUX Phase 2."
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=SESSION_BUDGET_SECONDS,
        help="Hard wall-clock session budget in seconds (default: %(default)s)",
    )
    args = parser.parse_args()
    session_budget = args.max_seconds

    log = load_log()
    log["started_at"] = _now_iso()
    log["session_budget_seconds"] = session_budget
    known_unusable_slots = dict(log.get("unusable_credential_slots", {}))
    configured_slots = _configured_credential_slots()

    print("=" * 72)
    print("Autonomous Production Evaluation Runner")
    print("=" * 72)

    # Initial counts
    counts = _count_eval_statuses()
    pending = _count_pending()
    log["starting_counts"] = dict(counts)
    print(f"Starting state: {counts}")
    print(f"Pending evaluations: {pending}")

    if pending == 0:
        log["stop_reason"] = "no_pending"
        log["stopped_at"] = _now_iso()
        _atomic_write_json(LOG_PATH, log)
        print("No pending evaluations. Nothing to do.")
        _write_report(log, counts, 0, "no_pending")
        return

    # Legacy quota records lack credential-slot provenance, so allow one run
    # with the current bounded failover configuration. Once a record includes
    # slot IDs for the configured attempts, a subsequent invocation can avoid
    # repeating the same known-exhausted credentials.
    recent = _get_most_recent_eval()
    if recent:
        known_unusable_slots = _update_unusable_slots(
            recent, known_unusable_slots
        )
    log["unusable_credential_slots"] = known_unusable_slots
    if recent:
        err = recent.get("error_type", "")
        attempts = recent.get("attempts", [])
        is_quota = (err == "QuotaExhaustedError" or
                    any("QuotaExhausted" in a.get("error_type", "") for a in attempts))
        attempted_slots = {
            a.get("key_slot")
            for a in attempts
            if a.get("key_slot") and a.get("error_type") == "QuotaExhaustedError"
        }
        if (
            recent.get("evaluation_status") == "FAILED"
            and err == "QuotaExhaustedError"
            and is_quota
            and attempted_slots
        ):
            print("STOPPING: Previous run ended with quota_exhausted.")
            print("The previous record already contains credential-slot failures.")
            log["stop_reason"] = "quota_exhausted"
            log["stopped_at"] = _now_iso()
            _atomic_write_json(LOG_PATH, log)
            _write_report(log, counts, 0, "quota_exhausted")
            return

    # If the most recent eval has a service error (503/504), warn but proceed —
    # the model may have recovered since the previous attempt.
    if recent:
        err = recent.get("error_type", "")
        if err in ("ServerError", "ServiceUnavailable", "InternalServerError"):
            print(f"WARNING: Previous eval ended with {err} (may have recovered). Proceeding with fresh attempts.")

    iterations = 0
    stop_reason = None
    session_start = time.monotonic()

    while True:
        # Enforce the wall-clock session budget (2h 30m default). This is the
        # only safe universal bound for a long queue; an iteration cap would
        # silently truncate work without warning.
        elapsed = time.monotonic() - session_start
        if elapsed >= session_budget:
            stop_reason = "session_budget_exhausted"
            print(
                f"\nSession budget ({session_budget}s) reached "
                f"after {elapsed:.0f}s. Stopping."
            )
            break

        # Re-check pending count
        pending = _count_pending()
        if pending == 0:
            stop_reason = "completed"
            print(f"\nAll evaluations complete (pending={pending}).")
            break
        if configured_slots and configured_slots.issubset(known_unusable_slots):
            stop_reason = "credentials_exhausted"
            print(
                "\nAll configured credential slots are known exhausted or "
                "unable to access the frozen model."
            )
            break

        print(f"\n--- Iteration {iterations + 1} ---")
        print(f"Pending: {pending}")
        print(f"Elapsed: {elapsed:.0f}s / {session_budget}s")

        # Run one evaluation
        print(f"Calling: run_production.py evaluate --max-evals {MAX_EVALS_PER_BATCH}")
        snapshot = _snapshot_eval_files()
        result = run_one_evaluation(known_unusable_slots)
        print(f"Exit code: {result['returncode']}")

        # Inspect ONLY a genuinely new eval file produced by this invocation.
        # Inspecting the most-recently-modified file overall can return a
        # pre-existing historical record (e.g. the controlled SUCCESS file) and
        # falsely report success when the subprocess actually crashed.
        recent = _get_newest_eval_after(snapshot)
        if not recent:
            print("WARNING: No new eval file produced by this invocation.")
            iterations += 1
            log["iterations"].append({
                "timestamp": _now_iso(),
                "iteration": iterations,
                "gen_id": "unknown",
                "status": "no_new_record",
                "error_type": "no_eval_file",
                "exit_code": result["returncode"],
                "stdout_tail": result["stdout"][-500:],
                "stderr_tail": result["stderr"][-500:],
            })
            _atomic_write_json(LOG_PATH, log)
            stop_reason = "no_new_record"
            break

        gen_id = recent.get("source_generation_id", "?")
        status = recent.get("evaluation_status", "UNKNOWN")
        error_type = recent.get("error_type", "")
        attempts = recent.get("attempts", [])

        print(f"Gen: {gen_id}")
        print(f"Status: {status}")
        if error_type:
            print(f"Error type: {error_type}")
        current_attempts = _attempts_this_run(log, gen_id, attempts)
        print(
            f"Attempts this run: {len(current_attempts)} "
            f"(record total: {len(attempts)})"
        )
        credential_slots = sorted({
            a.get("key_slot") for a in current_attempts if a.get("key_slot")
        })
        known_unusable_slots = _update_unusable_slots(
            recent, known_unusable_slots
        )
        log["unusable_credential_slots"] = known_unusable_slots
        if credential_slots:
            print(f"Credential slots: {', '.join(credential_slots)}")

        # A non-zero subprocess exit means the invocation did not complete
        # cleanly.  A stale/historical record's status must never override this.
        if result["returncode"] != 0:
            stderr_tail = result["stderr"][-500:]
            print(f"WARNING: Non-zero exit code ({result['returncode']}). Stopping.")
            log["iterations"].append({
                "timestamp": _now_iso(),
                "iteration": iterations + 1,
                "gen_id": gen_id,
                "status": status,
                "error_type": error_type or "subprocess_error",
                "attempts": len(current_attempts),
                "attempts_recorded": len(attempts),
                "exit_code": result["returncode"],
                "stderr_tail": stderr_tail,
            })
            iterations += 1
            _atomic_write_json(LOG_PATH, log)
            stop_reason = "subprocess_error"
            break

        # Log progress
        iterations += 1
        log["iterations"].append({
            "timestamp": _now_iso(),
            "iteration": iterations,
            "gen_id": gen_id,
            "status": status,
            "error_type": error_type,
            "attempts": len(current_attempts),
            "attempts_recorded": len(attempts),
            "credential_slots": credential_slots,
            "unusable_credential_slots": dict(known_unusable_slots),
            "exit_code": result["returncode"],
        })
        _atomic_write_json(LOG_PATH, log)

        # Check stop conditions
        stop, reason = should_stop_for_error(recent)
        if stop:
            stop_reason = reason
            print(f"\nSTOPPING: {reason}.")
            break

        # Brief pause between iterations (additional to configured spacing)
        time.sleep(2)

    log["stop_reason"] = stop_reason or "unknown"
    log["stopped_at"] = _now_iso()
    log["session_budget_seconds"] = session_budget
    log["elapsed_seconds"] = round(time.monotonic() - session_start, 1)
    log["venv_python"] = VENV_PYTHON
    log["runner_interpreter"] = sys.executable
    _atomic_write_json(LOG_PATH, log)

    # Final counts
    final_counts = _count_eval_statuses()
    final_pending = _count_pending()
    print(f"\n=== Final State ===")
    print(f"Counts: {final_counts}")
    print(f"Pending: {final_pending}")
    print(f"Iterations: {iterations}")
    print(f"Elapsed: {log['elapsed_seconds']}s")
    print(f"Stop reason: {log['stop_reason']}")

    _write_report(log, final_counts, iterations, log["stop_reason"], final_pending)


def _write_report(log: dict, counts: dict, iterations: int, reason: str, remaining: int = None):
    if remaining is None:
        remaining = _count_pending()
    starting_counts = log.get("starting_counts", counts)
    current_run_entries = [
        entry
        for entry in log.get("iterations", [])
        if entry.get("timestamp", "") >= log.get("started_at", "")
    ]
    newly_resolved = sum(
        entry.get("status") == "SUCCESS" for entry in current_run_entries
    )
    newly_failed = sum(
        entry.get("status") == "FAILED" for entry in current_run_entries
    )
    newly_manual_review = sum(
        entry.get("status") == "MANUAL_REVIEW"
        for entry in current_run_entries
    )

    lines = [
        "# Production Evaluation Runner Report",
        "",
        f"Generated: {_now_iso()}",
        "",
        "## Session Budget",
        f"- Configured budget: {log.get('session_budget_seconds', SESSION_BUDGET_SECONDS)}s",
        f"- Actual elapsed: {log.get('elapsed_seconds', '?')}s",
        "",
        "## Interpreter / Dependency",
        f"- Runner interpreter: {log.get('runner_interpreter', '?')}",
        f"- Subprocess Python (venv): {log.get('venv_python', '?')}",
        "",
        "## Starting Counts",
        f"- SUCCESS: {starting_counts.get('SUCCESS', 0)}",
        f"- FAILED: {starting_counts.get('FAILED', 0)}",
        f"- MANUAL_REVIEW: {starting_counts.get('MANUAL_REVIEW', 0)}",
        f"- UNKNOWN: {starting_counts.get('UNKNOWN', 0)}",
        "",
        "## Run Summary",
        f"- Iterations (evaluations attempted): {iterations}",
        f"- Stop reason: {reason}",
        "- Process at report generation: stopped",
        f"- Newly resolved evaluations: {newly_resolved}",
        f"- Newly failed evaluations: {newly_failed}",
        f"- Newly marked manual review: {newly_manual_review}",
        "",
        "## End-of-Run Counts",
        f"- SUCCESS: {counts.get('SUCCESS', 0)}",
        f"- FAILED: {counts.get('FAILED', 0)}",
        f"- MANUAL_REVIEW: {counts.get('MANUAL_REVIEW', 0)}",
        f"- UNKNOWN: {counts.get('UNKNOWN', 0)}",
        f"- Remaining pending: {remaining}",
        "",
        "## Progress Log",
    ]

    for index, entry in enumerate(log.get("iterations", [])):
        prior_entries = [
            prior
            for prior in log.get("iterations", [])[:index]
            if prior.get("gen_id") == entry.get("gen_id")
        ]
        prior_count = max(
            (
                prior.get("attempts_recorded", prior.get("attempts", 0))
                for prior in prior_entries
            ),
            default=0,
        )
        total_attempts = entry.get(
            "attempts_recorded", entry.get("attempts", 0)
        )
        attempts_this_run = (
            entry.get("attempts", 0)
            if "attempts_recorded" in entry
            else max(0, total_attempts - prior_count)
        )
        slots = set(entry.get("credential_slots", []))
        if "attempts_recorded" not in entry and prior_count:
            prior_slots = {
                slot
                for prior in prior_entries
                for slot in prior.get("credential_slots", [])
            }
            slots -= prior_slots
        slot_summary = ", ".join(sorted(slots)) or "not recorded"
        lines.append(
            f"- [{entry.get('timestamp', '?')}] Gen {entry.get('gen_id', '?')}: "
            f"status={entry.get('status', '?')}, "
            f"error={entry.get('error_type', '-')}, "
            f"attempts this run={attempts_this_run}, "
            f"total recorded attempts={total_attempts}, "
            f"credential slots={slot_summary}, "
            f"exit code={entry.get('exit_code', '?')}"
        )

    latest_iteration = log.get("iterations", [])[-1:] or []
    if latest_iteration:
        generation_id = latest_iteration[0].get("gen_id")
        latest_record = {}
        if generation_id and EVAL_DIR.exists():
            for path in sorted(
                EVAL_DIR.glob("*.json"),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            ):
                try:
                    candidate_record = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if candidate_record.get("source_generation_id") == generation_id:
                    latest_record = candidate_record
                    break
        if latest_record.get("attempts"):
            lines.extend([
                "",
                "## Latest Credential Outcomes",
                f"- Generation: {generation_id}",
                f"- Status: {latest_record.get('evaluation_status', 'UNKNOWN')}",
                f"- Label: {latest_record.get('generation_label') or 'none'}",
            ])
            for attempt in latest_record["attempts"]:
                slot = attempt.get("key_slot", "unknown slot")
                error_type = attempt.get("error_type", "success")
                message = (attempt.get("error_message") or "").lower()
                if error_type == "QuotaExhaustedError":
                    summary = "daily quota exhausted (HTTP 429)"
                elif "no longer available to new users" in message:
                    summary = "model access denied for this project (HTTP 404)"
                elif error_type == "GlobalModelConfigurationError":
                    summary = "global model/configuration error"
                elif attempt.get("status") == "SUCCESS":
                    summary = "successful evaluation response"
                else:
                    summary = error_type
                lines.append(f"- {slot}: {summary}")

    lines.extend([
        "",
        "## Quota / Service Status",
        f"- Stopped due to: {reason}",
        "",
        "## Artifact Paths",
        f"- Progress log: `data/production/evaluation_runner_log.json`",
        f"- This report: `docs/production_evaluation_runner_report.md`",
        f"- Evaluation records: `data/production/evaluations/`",
        f"- Generation artifacts: `data/production/generations.json`",
        "",
        "## Existing Records Preserved",
        "Yes — the runner only appends to or updates eval files; SUCCESS and "
        "MANUAL_REVIEW records are never overwritten.",
        "",
        "## Next Action",
    ])

    if reason == "quota_exhausted":
        lines.append("Quota is exhausted. Wait for the daily quota window to reset, then re-run:")
        lines.append("```")
        lines.append("python run_eval_autonomous.py")
        lines.append("```")
    elif reason == "completed":
        lines.append("All pending evaluations complete. Rebuild targets:")
        lines.append("```")
        lines.append("python run_production.py build")
        lines.append("```")
    elif reason == "persistent_error":
        lines.append("A persistent error was encountered. Inspect the eval record at:")
        lines.append("  data/production/evaluations/")
        lines.append("Do not re-run automatically.")
    elif reason == "no_pending":
        lines.append("No pending evaluations found. Rebuild targets:")
        lines.append("```")
        lines.append("python run_production.py build")
        lines.append("```")
    elif reason == "subprocess_error":
        lines.append("The evaluation subprocess exited non-zero. Inspect the captured")
        lines.append("stderr_tail in the progress log; the likely cause is a missing")
        lines.append("dependency in the spawned interpreter. Re-run under the venv:")
        lines.append("```")
        lines.append("python run_eval_autonomous.py")
        lines.append("```")
    elif reason == "no_new_record":
        lines.append("The subprocess produced no new evaluation record. Inspect the")
        lines.append("captured stdout/stderr in the progress log before re-running.")
    else:
        lines.append("Manual inspection required. See the progress log above.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
