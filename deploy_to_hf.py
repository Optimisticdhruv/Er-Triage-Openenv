import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_HF_USERNAME")
SPACE_NAME = "er-triage-openenv"
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

HF_README_HEADER = """---
title: ERTriageEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - medical
  - triage
  - healthcare
  - emergency
  - ai-agent
short_description: Hospital ER Triage - OpenEnv AI Agent Training Environment

---
"""

def check_prerequisites():
    print("Pre-flight checks...")
    checks = [
        ("HF_TOKEN set", bool(HF_TOKEN)),
        ("app/ directory exists", Path("app").exists()),
        ("openenv.yaml exists", Path("openenv.yaml").exists()),
        ("Dockerfile exists", Path("Dockerfile").exists()),
        ("inference.py in root", Path("inference.py").exists()),
        ("requirements.txt exists", Path("requirements.txt").exists()),
        ("static/dashboard.html exists", Path("static/dashboard.html").exists()),
    ]
    all_pass = True
    for name, status in checks:
        icon = "OK" if status else "FAIL"
        print(f"  [{icon}] {name}")
        if not status: all_pass = False
    if not all_pass:
        print("\nFix failed checks before deploying.")
        exit(1)
    print("All checks passed.\n")

def prepare_readme():
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding='utf-8')
        if not content.startswith("---"):
            content = HF_README_HEADER + "\n" + content
            readme_path.write_text(content, encoding='utf-8')
            print("README.md updated with HF Space YAML header.")
    else:
        readme_path.write_text(HF_README_HEADER + "\n# ERTriageEnv\n", encoding='utf-8')
        print("README.md created.")

def deploy():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Run: pip install huggingface_hub")
        exit(1)
    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="space")
        print(f"Space exists: https://huggingface.co/spaces/{REPO_ID}")
    except Exception:
        print(f"Creating space: {REPO_ID}")
        api.create_repo(repo_id=REPO_ID, repo_type="space",
            space_sdk="docker", private=False, exist_ok=True)
    upload_files = ["pyproject.toml","inference.py","openenv.yaml",
                    "Dockerfile","requirements.txt","README.md",
                    ".dockerignore",".env.example"]
    upload_dirs = ["app","scenarios","static"]
    print("\nUploading files...")
    for fname in upload_files:
        if Path(fname).exists():
            api.upload_file(path_or_fileobj=fname, path_in_repo=fname,
                repo_id=REPO_ID, repo_type="space")
            print(f"  Uploaded: {fname}")
    for dname in upload_dirs:
        if Path(dname).exists():
            api.upload_folder(folder_path=dname, path_in_repo=dname,
                repo_id=REPO_ID, repo_type="space")
            print(f"  Uploaded folder: {dname}/")
    print(f"""
Upload complete!
NEXT: Go to https://huggingface.co/spaces/{REPO_ID}
Click Settings tab and add these Repository Secrets:
  OPENAI_API_KEY  = your key
  API_BASE_URL    = https://api.openai.com/v1
  MODEL_NAME      = gpt-4o-mini
  HF_TOKEN        = your hf token
Wait 2-3 min for build, then click App tab to verify.
Test: curl https://{HF_USERNAME}-er-triage-openenv.hf.space/health
""")

if __name__ == "__main__":
    check_prerequisites()
    prepare_readme()
    deploy()
