from fastapi import FastAPI, Request, BackgroundTasks
from slack_sdk import WebClient
import os, ast, difflib, shutil, requests, tarfile, zipfile, logging, time, threading
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()
slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

# ── Slack event deduplication ──────────────────────────────────────────────────
# Slack retries events after ~3s if it doesn't get a fast response.
# We track seen event IDs (TTL=60s) to prevent duplicate pipeline triggers.
_seen_events: dict[str, float] = {}
_seen_lock = threading.Lock()
_EVENT_TTL = 60  # seconds


def _is_duplicate_event(event_id: str) -> bool:
    """Return True if this event_id was already processed recently."""
    now = time.time()
    with _seen_lock:
        # Prune expired entries
        expired = [k for k, t in _seen_events.items() if now - t > _EVENT_TTL]
        for k in expired:
            del _seen_events[k]
        if event_id in _seen_events:
            return True
        _seen_events[event_id] = now
    return False

# ── Pipeline orchestrator ──────────────────────────────────────────────────────
try:
    from orchestrator.pipeline import run_pipeline as _run_pipeline
    logger.info("Pipeline orchestrator loaded successfully.")
except Exception as _e:
    _run_pipeline = None
    logger.warning("Pipeline orchestrator could not be loaded: %s", _e)

# ─────────────────────────────────────────
# CONFIG — change these
PACKAGE_NAME = "endee"   # package to compare
VERSION_1    = "0.1.9"   # old version
VERSION_2    = "0.1.13"    # new version
REPORT_FILE  = "comparison_report.txt"
DOWNLOAD_DIR = Path("downloaded_packages")
# ─────────────────────────────────────────


def download_package(package: str, version: str) -> Path:
    folder = DOWNLOAD_DIR / f"{package}-{version}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    res = requests.get(f"https://pypi.org/pypi/{package}/{version}/json")
    res.raise_for_status()
    data = res.json()

    urls   = data["urls"]
    sdist  = next((u for u in urls if u["packagetype"] == "sdist"), None)
    wheel  = next((u for u in urls if u["packagetype"] == "bdist_wheel"), None)
    target = sdist or wheel

    if not target:
        raise Exception(f"No downloadable file found for {package}=={version}")

    filename = folder / target["filename"]
    with requests.get(target["url"], stream=True) as r:
        with open(filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    if filename.suffix == ".gz":
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(folder, filter="data")
    elif filename.suffix == ".whl":
        with zipfile.ZipFile(filename, "r") as z:
            z.extractall(folder)

    return folder


def get_python_files(folder: Path) -> dict:
    result = {}
    for f in folder.rglob("*.py"):
        rel = f.relative_to(folder)
        # Strip the first component (e.g. "endee-0.1.13/") so paths match across versions
        normalized = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel
        result[str(normalized)] = f.read_text(errors="ignore")
    return result


def compare_code_diff(files_v1: dict, files_v2: dict) -> str:
    lines = ["*LINE-BY-LINE CODE DIFFERENCES*"]
    for filename in sorted(set(files_v1) | set(files_v2)):
        if filename in ("setup.py", "__init__.py") or filename.endswith("__init__.py"):
            continue
        v1 = files_v1.get(filename, "").splitlines()
        v2 = files_v2.get(filename, "").splitlines()
        if v1 == v2:
            continue
            
        file_lines = [f"\nFile: {filename}", "```diff"]
        for group in difflib.SequenceMatcher(None, v1, v2).get_grouped_opcodes(3):
            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    for line in v1[i1:i2]:
                        if line.strip(): # Skip empty context lines
                            file_lines.append(f"  {line}")
                elif tag in ("replace", "delete"):
                    for line in v1[i1:i2]:
                        file_lines.append(f"- {line}")
                if tag in ("replace", "insert"):
                    for line in v2[j1:j2]:
                        file_lines.append(f"+ {line}")
        
        file_lines.append("```\n") # Close code block
        lines.extend(file_lines)
    if len(lines) == 3:
        lines.append("No code differences found.")
    return "\n".join(lines)


def extract_functions(source: str) -> dict:
    functions = {}
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                functions[node.name] = f"def {node.name}({', '.join(args)})"
    except Exception:
        pass
    return functions


def compare_function_signatures(files_v1: dict, files_v2: dict, v1: str = "", v2: str = "") -> str:
    v1_label = f"v{v1}" if v1 else "v1"
    v2_label = f"v{v2}" if v2 else "v2"
    lines = ["\n*FUNCTION SIGNATURE CHANGES*"]
    lines.append(f"_(NEW = added in {v2_label} | REMOVED = absent in {v2_label} compared to {v1_label})_")
    for filename in sorted(set(files_v1) | set(files_v2)):
        f1 = extract_functions(files_v1.get(filename, ""))
        f2 = extract_functions(files_v2.get(filename, ""))
        new     = set(f2) - set(f1)   # in v2 but not v1
        removed = set(f1) - set(f2)   # in v1 but not v2
        changed = {f for f in set(f1) & set(f2) if f1[f] != f2[f]}
        if new or removed or changed:
            lines.append(f"\nFile: {filename}")
            for f in sorted(new):
                lines.append(f"  [NEW] ({v2_label}):     {f2[f]}")
            for f in sorted(removed):
                lines.append(f"  [REMOVED] ({v2_label}): {f1[f]}")
            for f in sorted(changed):
                lines.append(f"  [MODIFIED]:\n      {v1_label}: {f1[f]}\n      {v2_label}: {f2[f]}")
    if len(lines) == 4:
        lines.append("No function signature changes found.")
    return "\n".join(lines)


def get_requirements(folder: Path) -> set:
    for req_file in folder.rglob("requirements*.txt"):
        return set(req_file.read_text(errors="ignore").splitlines())
    return set()


def compare_dependencies(folder_v1: Path, folder_v2: Path, v1: str = "", v2: str = "") -> str:
    v1_label = f"v{v1}" if v1 else "v1"
    v2_label = f"v{v2}" if v2 else "v2"
    lines   = ["\n*DEPENDENCIES COMPARISON*"]
    lines.append(f"_(Added = new in {v2_label} | Removed = absent in {v2_label} compared to {v1_label})_")
    r1, r2  = get_requirements(folder_v1), get_requirements(folder_v2)
    added   = r2 - r1   # in v2 but not v1
    removed = r1 - r2   # in v1 but not v2
    if added:
        lines.append(f"\nAdded in {v2_label}:")
        lines.extend(f"   + {r}" for r in sorted(added))
    if removed:
        lines.append(f"\nRemoved in {v2_label}:")
        lines.extend(f"   - {r}" for r in sorted(removed))
    if not added and not removed:
        lines.append("No dependency changes found.")
    return "\n".join(lines)


def run_comparison(channel: str, v1: str = VERSION_1, v2: str = VERSION_2):
    """Main agent — triggered when user types 'run [v1] [v2]' in Slack."""
    try:
        # Notify Slack that the task has started
        resp = slack_client.chat_postMessage(
            channel=channel,
            text=f"Agent process initiated. Comparing *{PACKAGE_NAME}* v{v1} vs v{v2}. Please wait."
        )
        init_ts = resp.get("ts")

        # Step 1: Download both versions
        slack_client.chat_postMessage(channel=channel, thread_ts=init_ts, text="Downloading specified versions from PyPI...")
        folder_v1 = download_package(PACKAGE_NAME, v1)
        folder_v2 = download_package(PACKAGE_NAME, v2)

        # Step 2: Read all python files
        files_v1 = get_python_files(folder_v1)
        files_v2 = get_python_files(folder_v2)

        # Step 3: Run comparisons
        slack_client.chat_postMessage(channel=channel, thread_ts=init_ts, text="Analyzing codebase differences...")
        header      = f"\nPACKAGE : {PACKAGE_NAME}\nv1      : {v1}\nv2      : {v2}\n"
        diff_report = compare_code_diff(files_v1, files_v2)
        func_report = compare_function_signatures(files_v1, files_v2, v1, v2)
        deps_report = compare_dependencies(folder_v1, folder_v2, v1, v2)

        # Step 4: Save report
        full_report = header + diff_report + func_report + deps_report
        with open(REPORT_FILE, "w") as f:
            f.write(full_report)

        # Step 5: Send full report to Slack as plain text (chunked to stay within 4000-char limit)
        slack_client.chat_postMessage(
            channel=channel,
            thread_ts=init_ts,
            text=(
                f"*Comparison Completed*\n"
                f"Package: `{PACKAGE_NAME}`\n"
                f"Versions: `{v1}` -> `{v2}`\n"
                f"Full report saved to: `{REPORT_FILE}`"
            )
        )

        def send_in_chunks(text: str):
            chunk_size = 3900
            for i in range(0, len(text), chunk_size):
                slack_client.chat_postMessage(
                    channel=channel,
                    thread_ts=init_ts,
                    text=text[i:i + chunk_size]
                )

        send_in_chunks(func_report)
        send_in_chunks(deps_report)
        send_in_chunks(diff_report)

    except Exception as e:
        slack_client.chat_postMessage(channel=channel, thread_ts=init_ts, text=f"Error encountered: {str(e)}")


# ─────────────────────────────────────────
# FASTAPI ROUTES
# ─────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Hello, FastAPI is running!"}


@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    # Slack URL verification
    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    # Deduplicate retried events
    event_id = body.get("event_id", "")
    if event_id and _is_duplicate_event(event_id):
        logger.debug("[slack] Skipping duplicate event %s", event_id)
        return {"status": "ok"}

    event  = body.get("event", {})
    if event.get("type") == "message":
        text   = event.get("text", "").strip().lower()
        channel = event.get("channel")
        bot_id  = event.get("bot_id")

        # ── Existing: trigger package comparison ──────────────────────────────
        # Trigger agent when user types "run <version1> <version2>"
        if text.startswith("run") and not bot_id:
            parts = text.split()
            v1 = parts[1] if len(parts) > 1 else VERSION_1
            v2 = parts[2] if len(parts) > 2 else VERSION_2
            background_tasks.add_task(run_comparison, channel, v1, v2)

        # ── New: trigger pipeline via message ─────────────────────────────────
        # Trigger when user types "automate" in a channel message
        elif text.startswith("automate") and not bot_id:
            background_tasks.add_task(
                _run_pipeline_background,
                channel,
                event.get("user", ""),
                "auto/endee-update",
                "all",
            )

    return {"status": "ok"}


# ─────────────────────────────────────────
# SLASH COMMAND:  /sync-clients
# ─────────────────────────────────────────
# Register this URL in Slack:
#   Slash Commands → Request URL → https://your-host/slack/slash/sync-clients
#
# Usage:
#   /sync-clients
#   /sync-clients --branch feature/my-branch --scope all
#   /sync-clients --branch hotfix/x --scope crewai,langchain

import shlex as _shlex


def _parse_slash_args(text: str) -> tuple[str, str]:
    """
    Parse /sync-clients command arguments.

    Supported flags:
      --branch <name>    Git branch to commit to (default: auto/endee-update)
      --scope  <list>    Comma-separated clients or "all" (default: all)

    Returns:
        (branch, scope)
    """
    branch = "auto/endee-update"
    scope = "all"

    try:
        tokens = _shlex.split(text or "")
    except ValueError:
        tokens = (text or "").split()

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--branch" and i + 1 < len(tokens):
            branch = tokens[i + 1]
            i += 2
        elif tok == "--scope" and i + 1 < len(tokens):
            scope = tokens[i + 1]
            i += 2
        else:
            i += 1

    return branch, scope


@app.post("/slack/slash/sync-clients")
async def slash_sync_clients(request: Request, background_tasks: BackgroundTasks):
    """
    Handle the /sync-clients Slack slash command.

    Slack requires a response within 3 seconds, so we acknowledge immediately
    and run the pipeline in a background task.

    Slack slash command payload (form-encoded):
        command     = /sync-clients
        text        = --branch feature/x --scope all
        channel_id  = C0123456
        user_id     = U0123456
        response_url= https://hooks.slack.com/commands/…
    """
    form = await request.form()
    channel = form.get("channel_id", "")
    user = form.get("user_id", "")
    text = form.get("text", "")

    branch, scope = _parse_slash_args(text)

    background_tasks.add_task(_run_pipeline_background, channel, user, branch, scope)

    return {
        "response_type": "ephemeral",
        "text": (
            f"Initializing `sync-clients` pipeline.\n"
            f"Branch: `{branch}` | Scope: `{scope}`\n"
            "Progress updates will appear in this channel."
        ),
    }


# ─────────────────────────────────────────
# SHARED PIPELINE RUNNER (background task)
# ─────────────────────────────────────────

def _run_pipeline_background(
    channel: str,
    trigger_user: str,
    branch: str,
    scope: str,
) -> None:
    """
    Background task that runs the three-stage pipeline and streams progress
    back to Slack via chat_postMessage after each stage.

    Pipeline:
      Stage 1 — Claude subagent → parse diff → structured JSON
      Stage 2 — Claude subagents → Read/Edit integration files (×N parallel)
      Stage 3 — pytest → if fail → Claude subagent fixes → retry (up to 3×)
    """
    if _run_pipeline is None:
        slack_client.chat_postMessage(
            channel=channel,
            text=(
                "Error: Pipeline orchestrator is not available.\n"
                "Ensure the `claude` CLI is installed (`npm install -g @anthropic-ai/claude-code`) "
                "and `pip install -r requirements.txt` has been run."
            ),
        )
        return

    report_path = Path(os.getenv("COMPARISON_REPORT_PATH", REPORT_FILE))
    if not report_path.exists():
        slack_client.chat_postMessage(
            channel=channel,
            text=(
                f"Error: Comparison report not found at `{report_path}`.\n"
                "Run a comparison first (`run <v1> <v2>`), then re-trigger automation."
            ),
        )
        return

    report_content = report_path.read_text(encoding="utf-8")

    # Starter message — all stage updates will be threaded under it
    resp = slack_client.chat_postMessage(
        channel=channel,
        text=(
            f"<@{trigger_user}> initiated *sync-clients* process "
            f"(branch: `{branch}`, scope: `{scope}`)"
        ),
    )
    thread_ts = resp.get("ts")

    def _notify(msg: str) -> None:
        """Post a progress update to the Slack thread."""
        slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=msg,
        )

    import asyncio

    try:
        results = asyncio.run(
            _run_pipeline(
                report_content=report_content,
                branch=branch,
                scope=scope,
                notify=_notify,
            )
        )

        # Final summary
        errors = results.get("errors", [])
        success = results.get("success", False)
        cost = results.get("cost", 0.0)
        usage = results.get("usage", {})

        summary_lines = ["─" * 40]
        if success:
            summary_lines.append("*Pipeline execution completed successfully.*")
        else:
            summary_lines.append("*Pipeline execution completed with errors.*")
            
        # Append cost and usage if available
        if cost > 0:
            master = usage.get("master_usage", {})
            subagent = usage.get("subagent_usage", {})

            m_in = master.get("input_tokens", 0) + master.get("cache_read_input_tokens", 0) + master.get("cache_creation_input_tokens", 0)
            m_out = master.get("output_tokens", 0)
            # Calculate total subagent tokens across all tracked subagents
            s_total = sum(d.get("total_tokens", 0) for d in subagent.values())
            total_tokens = m_in + m_out + s_total

            summary_lines.append(f"*Cost / Usage Summary:*")
            summary_lines.append(f"  • Total Cost: `${cost:.4f}` ({total_tokens:,} total tokens)")
            summary_lines.append(f"  • Master (Haiku): Input: {m_in:,} | Output: {m_out:,}")
            if s_total > 0:
                summary_lines.append(f"  • Subagents (Sonnet): ~{s_total:,} total tokens")
                for sub_name, sub_data in sorted(subagent.items()):
                    st = sub_data.get("total_tokens", 0)
                    if st > 0:
                        summary_lines.append(f"      ◦ _{sub_name}_: {st:,} tokens")

        if errors:
            summary_lines.append("\n*Errors Encountered:*")
            summary_lines.extend(f"  • {e}" for e in errors[:5])

        slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="\n".join(summary_lines),
        )

        logger.info(
            "[pipeline] Done. success=%s errors=%s",
            success, errors,
        )

    except Exception as exc:
        logger.exception("[pipeline] Unhandled exception.")
        slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=f"Pipeline encountered an unexpected error: `{exc}`",
        )