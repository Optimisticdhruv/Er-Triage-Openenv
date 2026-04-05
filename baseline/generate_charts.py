import json, sys
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

RESULTS_DIR = Path("baseline/results")

COLORS = {
    "gpt-4o": "#1d4ed8", "gpt-4o-mini": "#0f766e",
    "llama-3.1-70b": "#7c3aed", "mistral-7b": "#b45309", "default": "#374151"
}

def load_all_results():
    results = []
    for f in RESULTS_DIR.glob("results_*.json"):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list): results.extend(data)
        except Exception as e:
            print(f"Skip {f}: {e}")
    return results

def plot_reward_curves(results, out_path="baseline/results/reward_curves.png"):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed: pip install matplotlib")
        return
    tasks = ["task_easy", "task_medium", "task_hard"]
    titles = ["Task 1: Single Triage (Easy)",
              "Task 2: Shift Management (Medium)",
              "Task 3: Mass Casualty (Hard)"]
    pass_scores = {"task_easy": 0.70, "task_medium": 0.60, "task_hard": 0.50}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ERTriageEnv Agent Performance Benchmark", fontsize=14, fontweight="bold")
    for i, (tid, title) in enumerate(zip(tasks, titles)):
        ax = axes[i]
        seen = set()
        for r in results:
            if r.get("task_id") != tid: continue
            mk = r.get("model_key", r.get("model", "unknown"))
            if mk in seen: continue
            seen.add(mk)
            rw = r.get("step_rewards", [])
            if not rw: continue
            cum = [sum(rw[:j+1])/(j+1) for j in range(len(rw))]
            ax.plot(range(1, len(cum)+1), cum,
                label=r.get("model_display", mk),
                color=COLORS.get(mk, COLORS["default"]), linewidth=2.5,
                marker='o', markersize=3, alpha=0.85)
        ax.axhline(y=pass_scores[tid], linestyle='--', color='#dc2626',
            alpha=0.6, label=f"Pass: {pass_scores[tid]}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("Mean Reward")
        ax.set_ylim(-0.15, 1.10); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {out_path}")
    plt.close()

def update_readme_scores(results):
    readme = Path("README.md")
    if not readme.exists(): return
    try:
        content = readme.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        print("README.md has encoding issues, skipping update")
        return
    by_model = {}
    for r in results:
        display = r.get("model_display", r.get("model_key", "unknown"))
        if display not in by_model: by_model[display] = {}
        by_model[display][r["task_id"]] = r["task_score"]
    for display, scores in by_model.items():
        mean = round(sum(scores.values())/len(scores),4) if scores else "TBD"
        old = f"| {display} | TBD | TBD | TBD | TBD |"
        new = (f"| {display} | {scores.get('task_easy','TBD')} | "
               f"{scores.get('task_medium','TBD')} | "
               f"{scores.get('task_hard','TBD')} | {mean} |")
        content = content.replace(old, new)
    readme.write_text(content, encoding='utf-8')
    print("README.md updated with baseline scores.")

if __name__ == "__main__":
    results = load_all_results()
    if not results:
        print("No results. Run: python inference.py or python baseline/run_baseline.py first")
        sys.exit(1)
    print(f"Found {len(results)} result records.")
    plot_reward_curves(results)
    update_readme_scores(results)
    print("Done.")
