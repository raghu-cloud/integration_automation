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


def extract_classes(source: str) -> dict:
    """Returns {class_name: [base_names]} for every class in source."""
    classes = {}
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                        bases.append(f"{base.value.id}.{base.attr}")
                classes[node.name] = bases
    except Exception:
        pass
    return classes


def extract_constants(source: str) -> set:
    """Returns set of module-level ALL_CAPS constant names."""
    constants = set()
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id.isupper():
                    constants.add(node.target.id)
    except Exception:
        pass
    return constants


def compare_structural_changes(files_v1: dict, files_v2: dict, v1: str = "", v2: str = "") -> str:
    v1_label = f"v{v1}" if v1 else "v1"
    v2_label = f"v{v2}" if v2 else "v2"

    class_added, class_removed = [], []
    const_added, const_removed = [], []

    for filename in sorted(set(files_v1) | set(files_v2)):
        c1 = extract_classes(files_v1.get(filename, ""))
        c2 = extract_classes(files_v2.get(filename, ""))
        for cls in sorted(set(c2) - set(c1)):
            class_added.append(f"{cls}  [{filename}]")
        for cls in sorted(set(c1) - set(c2)):
            class_removed.append(f"{cls}  [{filename}]")

        k1 = extract_constants(files_v1.get(filename, ""))
        k2 = extract_constants(files_v2.get(filename, ""))
        for k in sorted(k2 - k1):
            const_added.append(f"{k}  [{filename}]")
        for k in sorted(k1 - k2):
            const_removed.append(f"{k}  [{filename}]")

    lines = ["\n" + "=" * 50, "🏗️  STRUCTURAL CHANGES", "=" * 50]
    lines.append(f"Comparing {v1_label} → {v2_label}\n")

    if class_added:
        lines.append(f"✅ NEW CLASSES ({len(class_added)}):")
        lines.extend(f"   + {c}" for c in sorted(class_added))
    else:
        lines.append("✅ NEW CLASSES: none")

    lines.append("")

    if class_removed:
        lines.append(f"❌ REMOVED CLASSES ({len(class_removed)}):")
        lines.extend(f"   - {c}" for c in sorted(class_removed))
    else:
        lines.append("❌ REMOVED CLASSES: none")

    lines.append("")

    if const_added:
        lines.append(f"✅ NEW CONSTANTS ({len(const_added)}):")
        lines.extend(f"   + {c}" for c in sorted(const_added))
    else:
        lines.append("✅ NEW CONSTANTS: none")

    lines.append("")

    if const_removed:
        lines.append(f"❌ REMOVED CONSTANTS ({len(const_removed)}):")
        lines.extend(f"   - {c}" for c in sorted(const_removed))
    else:
        lines.append("❌ REMOVED CONSTANTS: none")

    return "\n".join(lines)


def extract_functions(source: str) -> dict:
    """Returns {qualified_name: signature_str} for every function/method in source."""
    functions = {}
    try:
        tree = ast.parse(source)
        # Build a parent map so we can resolve class context
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node  # type: ignore[attr-defined]

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                sig  = f"def {node.name}({', '.join(args)})"
                # Walk up to find enclosing class (if any)
                parent = getattr(node, "_parent", None)
                if isinstance(parent, ast.ClassDef):
                    qualified = f"{parent.name}.{node.name}"
                else:
                    qualified = node.name
                functions[qualified] = sig
    except Exception:
        pass
    return functions


def compare_function_signatures(files_v1: dict, files_v2: dict, v1: str = "", v2: str = "") -> str:
    v1_label = f"v{v1}" if v1 else "v1"
    v2_label = f"v{v2}" if v2 else "v2"

    # Collect per-file diffs
    per_file = {}
    all_added, all_removed, all_changed = [], [], []
    for filename in sorted(set(files_v1) | set(files_v2)):
        f1 = extract_functions(files_v1.get(filename, ""))
        f2 = extract_functions(files_v2.get(filename, ""))
        new     = sorted(set(f2) - set(f1))
        removed = sorted(set(f1) - set(f2))
        changed = sorted(f for f in set(f1) & set(f2) if f1[f] != f2[f])
        if new or removed or changed:
            per_file[filename] = (f1, f2, new, removed, changed)
            all_added.extend(f"{f}  [{filename}]" for f in new)
            all_removed.extend(f"{f}  [{filename}]" for f in removed)
            all_changed.extend(f"{f}  [{filename}]" for f in changed)

    lines = ["\n" + "=" * 50, "🔍 FUNCTION CHANGES SUMMARY", "=" * 50]
    lines.append(f"Comparing {v1_label} → {v2_label}\n")

    if all_added:
        lines.append(f"✅ ADDED ({len(all_added)} function{'s' if len(all_added) != 1 else ''}):")
        lines.extend(f"   + {f}" for f in sorted(all_added))
    else:
        lines.append("✅ ADDED: none")

    lines.append("")

    if all_removed:
        lines.append(f"❌ REMOVED ({len(all_removed)} function{'s' if len(all_removed) != 1 else ''}):")
        lines.extend(f"   - {f}" for f in sorted(all_removed))
    else:
        lines.append("❌ REMOVED: none")

    lines.append("")

    if all_changed:
        lines.append(f"✏️  SIGNATURE CHANGED ({len(all_changed)} function{'s' if len(all_changed) != 1 else ''}):")
        lines.extend(f"   ~ {f}" for f in sorted(all_changed))
    else:
        lines.append("✏️  SIGNATURE CHANGED: none")

    if not per_file:
        lines.append("\nNo function changes found.")
        return "\n".join(lines)

    # Per-file detail
    lines.append("\n" + "-" * 50)
    lines.append("📂 DETAILS BY FILE")
    lines.append("-" * 50)
    for filename, (f1, f2, new, removed, changed) in per_file.items():
        lines.append(f"\n📁 {filename}")
        for f in new:
            lines.append(f"  ✅ ADDED:   {f2[f]}")
        for f in removed:
            lines.append(f"  ❌ REMOVED: {f1[f]}")
        for f in changed:
            lines.append(f"  ✏️  CHANGED:")
            lines.append(f"      {v1_label}: {f1[f]}")
            lines.append(f"      {v2_label}: {f2[f]}")

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
        header       = f"\nPACKAGE : {PACKAGE_NAME}\nv1      : {v1}\nv2      : {v2}\n"
        struct_report = compare_structural_changes(files_v1, files_v2, v1, v2)
        func_report   = compare_function_signatures(files_v1, files_v2, v1, v2)
        deps_report   = compare_dependencies(folder_v1, folder_v2, v1, v2)

        # Step 4: Save report
        full_report = header + struct_report + func_report + deps_report
        with open(REPORT_FILE, "w") as f:
            f.write(full_report)

        # Step 5: Send to Slack (chunked to stay within 4000-char limit)
        def send_in_chunks(text: str):
            chunk_size = 3900
            for i in range(0, len(text), chunk_size):
                slack_client.chat_postMessage(channel=channel, text=text[i:i + chunk_size])

        send_in_chunks(struct_report)
        send_in_chunks(func_report)
        send_in_chunks(deps_report)

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