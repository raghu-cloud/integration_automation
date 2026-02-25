from fastapi import FastAPI, Request, BackgroundTasks
from slack_sdk import WebClient
import os, ast, difflib, shutil, requests, tarfile, zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

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
            tar.extractall(folder)
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
    lines = ["=" * 50, "📄 LINE BY LINE CODE DIFF", "=" * 50]
    for filename in sorted(set(files_v1) | set(files_v2)):
        v1 = files_v1.get(filename, "").splitlines()
        v2 = files_v2.get(filename, "").splitlines()
        if v1 == v2:
            continue
        file_lines = [f"\n📄 {filename}"]
        for group in difflib.SequenceMatcher(None, v1, v2).get_grouped_opcodes(3):
            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    for line in v1[i1:i2]:
                        file_lines.append(f"     {line}")
                elif tag in ("replace", "delete"):
                    for line in v1[i1:i2]:
                        file_lines.append(f"➖  {line}")
                if tag in ("replace", "insert"):
                    for line in v2[j1:j2]:
                        file_lines.append(f"➕  {line}")
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
    lines = ["\n" + "=" * 50, "🔍 CHANGED FUNCTION SIGNATURES", "=" * 50]
    lines.append(f"(NEW = added in {v2_label} | REMOVED = absent in {v2_label} compared to {v1_label})")
    for filename in sorted(set(files_v1) | set(files_v2)):
        f1 = extract_functions(files_v1.get(filename, ""))
        f2 = extract_functions(files_v2.get(filename, ""))
        new     = set(f2) - set(f1)   # in v2 but not v1
        removed = set(f1) - set(f2)   # in v1 but not v2
        changed = {f for f in set(f1) & set(f2) if f1[f] != f2[f]}
        if new or removed or changed:
            lines.append(f"\n📁 {filename}")
            for f in sorted(new):
                lines.append(f"  ✅ NEW ({v2_label}):     {f2[f]}")
            for f in sorted(removed):
                lines.append(f"  ❌ REMOVED ({v2_label}): {f1[f]}")
            for f in sorted(changed):
                lines.append(f"  ✏️  CHANGED:\n      {v1_label}: {f1[f]}\n      {v2_label}: {f2[f]}")
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
    lines   = ["\n" + "=" * 50, "📦 DEPENDENCIES COMPARISON", "=" * 50]
    lines.append(f"(Added = new in {v2_label} | Removed = absent in {v2_label} compared to {v1_label})")
    r1, r2  = get_requirements(folder_v1), get_requirements(folder_v2)
    added   = r2 - r1   # in v2 but not v1
    removed = r1 - r2   # in v1 but not v2
    if added:
        lines.append(f"\n✅ Added in {v2_label}:")
        lines.extend(f"   + {r}" for r in sorted(added))
    if removed:
        lines.append(f"\n❌ Removed in {v2_label}:")
        lines.extend(f"   - {r}" for r in sorted(removed))
    if not added and not removed:
        lines.append("No dependency changes found.")
    return "\n".join(lines)


def run_comparison(channel: str, v1: str = VERSION_1, v2: str = VERSION_2):
    """Main agent — triggered when user types 'run [v1] [v2]' in Slack."""
    try:
        # Notify Slack that the task has started
        slack_client.chat_postMessage(
            channel=channel,
            text=f"🤖 Agent started! Comparing *{PACKAGE_NAME}* v{v1} vs v{v2}... Please wait ⏳"
        )

        # Step 1: Download both versions
        slack_client.chat_postMessage(channel=channel, text="⬇️ Downloading both versions from PyPI...")
        folder_v1 = download_package(PACKAGE_NAME, v1)
        folder_v2 = download_package(PACKAGE_NAME, v2)

        # Step 2: Read all python files
        files_v1 = get_python_files(folder_v1)
        files_v2 = get_python_files(folder_v2)

        # Step 3: Run comparisons
        slack_client.chat_postMessage(channel=channel, text="🔍 Comparing changes...")
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
            text=(
                f"✅ *Comparison Complete!*\n"
                f"📦 Package: `{PACKAGE_NAME}`\n"
                f"🔁 Versions: `{v1}` → `{v2}`\n"
                f"📄 Full report saved to: `{REPORT_FILE}`"
            )
        )

        def send_in_chunks(text: str):
            chunk_size = 3900
            for i in range(0, len(text), chunk_size):
                slack_client.chat_postMessage(channel=channel, text=text[i:i + chunk_size])

        send_in_chunks(func_report)
        send_in_chunks(deps_report)
        send_in_chunks(diff_report)

    except Exception as e:
        slack_client.chat_postMessage(channel=channel, text=f"❌ Error: {str(e)}")


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

    event  = body.get("event", {})
    if event.get("type") == "message":
        text   = event.get("text", "").strip().lower()
        channel = event.get("channel")
        bot_id  = event.get("bot_id")

        # Trigger agent when user types "run <version1> <version2>"
        if text.startswith("run") and not bot_id:
            parts = text.split()
            v1 = parts[1] if len(parts) > 1 else VERSION_1
            v2 = parts[2] if len(parts) > 2 else VERSION_2
            print(v1,v2)
            background_tasks.add_task(run_comparison, channel, v1, v2)

    return {"status": "ok"}