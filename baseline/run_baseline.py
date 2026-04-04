import os
import json
import time
import argparse
import requests
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_URL = os.environ.get("ENVIRONMENT_URL", "http://localhost:7860")
SEED = int(os.environ.get("SEED", "42"))
RESULTS_DIR = Path("baseline/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_CONFIG = {
    "gpt-4o": {"api": "openai", "model_id": "gpt-4o", "display": "GPT-4o"},
    "gpt-4o-mini": {"api": "openai", "model_id": "gpt-4o-mini", "display": "GPT-4o-mini"},
    "qwen-plus": {"api": "openrouter", "model_id": "qwen/qwen3.6-plus:free", "display": "Qwen 3.6 Plus"},
    "step-3.5": {"api": "openrouter", "model_id": "stepfun/step-3.5-flash:free", "display": "Step 3.5 Flash"},
    "gpt-oss-120b": {"api": "openrouter", "model_id": "openai/gpt-oss-120b:free", "display": "GPT-OSS 120B"},
    "nemotron-super": {"api": "openrouter", "model_id": "nvidia/nemotron-3-super-120b-a12b:free", "display": "Nemotron 3 Super"},
    "gemma-3-4b": {"api": "openrouter", "model_id": "google/gemma-3-4b-it:free", "display": "Gemma 3 4B"},
    "deepseek-v3": {"api": "openrouter", "model_id": "deepseek/deepseek-v3.2", "display": "DeepSeek V3.2"},
    "deepseek-chat": {"api": "openrouter", "model_id": "deepseek/deepseek-chat", "display": "DeepSeek Chat"},
    "deepseek-r1": {"api": "openrouter", "model_id": "deepseek/deepseek-r1", "display": "DeepSeek R1"},
}

SYSTEM_PROMPT = """You are an emergency room triage nurse.
ESI Levels: 1=Immediate(life-threat), 2=Emergent(high-risk), 3=Urgent, 4=Less-urgent.
ESI-1 signs: SpO2 less than or equal to 85%, SBP less than 80, GCS less than or equal to 8, HR greater than 150 with cardiac/respiratory, cardiac arrest.
ESI-2 signs: SpO2 86-92%, SBP 80-100, GCS 9-13, HR 120+, sepsis.
Beds: resus=ESI-1, acute=ESI-2/3, minor=ESI-3/4, waiting_area=ESI-4/5.
Escalate=true for ESI-1 and ESI-2.
RESPOND ONLY VALID JSON:
{"patient_id":"P001","priority":2,"bed_type":"acute","escalate":true,"reasoning":"one sentence"}"""

def get_client(model_key: str):
    cfg = MODELS_CONFIG.get(model_key, {"api":"openai","model_id":model_key,"display":model_key})
    if cfg["api"] == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set. Free signup: openrouter.ai")
        return OpenAI(api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"), cfg["model_id"]
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL), cfg["model_id"]

def fmt_patient(obs):
    v = obs.get("vitals", {})
    alerts = obs.get("active_alerts", [])
    alert_str = " ALERTS: " + "; ".join(alerts) if alerts else ""
    return (f"ID:{obs['patient_id']} Age:{obs['age']} Sex:{obs['sex']} Mode:{obs['arrival_mode']}\n"
            f"Complaint: {obs['chief_complaint']}\n"
            f"HR={v.get('heart_rate')} BP={v.get('bp_systolic')}/{v.get('bp_diastolic')} "
            f"SpO2={v.get('spo2')}% RR={v.get('respiratory_rate')} "
            f"Temp={v.get('temperature')}C Pain={v.get('pain_score')}/10 GCS={v.get('gcs')}\n"
            f"Waiting:{obs.get('waiting_minutes',0):.1f}min{alert_str}")

def get_action(client, model_id, obs):
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": fmt_patient(obs)}
            ],
            temperature=0,
            max_tokens=200
        )
        raw = r.choices[0].message.content.strip()
        # Strip markdown code blocks if model wraps in ```json
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        data = json.loads(raw.strip())
        
        # Sanitize and validate all fields
        valid_beds = {'resus', 'acute', 'minor', 'waiting_area'}
        priority = int(data.get('priority', 3))
        priority = max(1, min(4, priority))  # clamp to 1-4
        
        bed_type = str(data.get('bed_type', 'minor')).lower().strip()
        if bed_type not in valid_beds:
            bed_type = 'minor'  # safe default
            
        escalate = bool(data.get('escalate', False))
        
        return {
            'patient_id': obs['patient_id'],  # always use obs patient_id, not model's
            'priority': priority,
            'bed_type': bed_type,
            'escalate': escalate,
            'reasoning': str(data.get('reasoning', 'model decision'))
        }
    except Exception as e:
        print(f'    Action parse error: {e}')
        return {
            'patient_id': obs['patient_id'],
            'priority': 3,
            'bed_type': 'minor',
            'escalate': False,
            'reasoning': 'parse_error_fallback'
        }

def run_task(task_id, model_key, seed=42, client_override=None):
    if client_override is not None:
        if isinstance(client_override, tuple):
            client, model_id = client_override
        else:
            client = client_override
            model_id = MODELS_CONFIG.get(model_key, {}).get('model_id', model_key)
    else:
        client, model_id = get_client(model_key)
    resp = requests.get(f"{ENV_URL}/reset", params={"task_id":task_id,"seed":seed})
    resp.raise_for_status()
    state = resp.json()
    rewards, step_n = [], 0
    start = time.time()
    while not state.get("episode_done") and state.get("patients_waiting") and step_n < 100:
        try:
            patient = state["patients_waiting"][0]
            action = get_action(client, model_id, patient)
            sr = requests.post(f"{ENV_URL}/step", json=action,
                headers={"Content-Type": "application/json"})
            if sr.status_code != 200:
                print(f'    Step {step_n} error: {sr.status_code} - {sr.text[:100]}')
                break
            result = sr.json()
            rewards.append(result["reward"]["total"])
            state = result["observation"]
            step_n += 1
        except Exception as e:
            print(f'    Step {step_n} exception: {e}')
            break
    mean = round(sum(rewards)/len(rewards), 4) if rewards else 0.0
    return {
        "task_id": task_id, "model_key": model_key,
        "model_display": MODELS_CONFIG.get(model_key,{}).get("display",model_key),
        "seed": seed, "steps": step_n, "mean_reward": mean,
        "task_score": round(max(0.0,min(1.0,mean)),4),
        "step_rewards": rewards,
        "duration_seconds": round(time.time()-start,2),
    }

def run_all(model_key, seed=42, tasks=None):
    if tasks is None: tasks = ["task_easy","task_medium","task_hard"]
    results = []
    print(f"\n{'='*60}\nERTriageEnv Baseline | Model: {model_key} | Seed: {seed}\n{'='*60}")
    for tid in tasks:
        print(f"\nRunning {tid}...", flush=True)
        r = run_task(tid, model_key, seed)
        results.append(r)
        print(f"  Score: {r['task_score']:.4f} | Steps: {r['steps']} | Time: {r['duration_seconds']}s")
    print_table(results)
    fname = RESULTS_DIR/f"results_{model_key.replace('/','_')}_{int(time.time())}.json"
    fname.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {fname}")
    return results

def run_leaderboard(seed=42):
    all_results = []
    for mk, cfg in MODELS_CONFIG.items():
        print(f"\nRunning: {cfg['display']}")
        try:
            client, model_id = get_client(mk)
            for tid in ["task_easy","task_medium","task_hard"]:
                r = run_task(tid, mk, seed, client_override=(client, model_id))
                all_results.append(r)
        except ValueError as e:
            print(f"  SKIPPED: {e}")
    if all_results:
        md = gen_leaderboard_md(all_results)
        (RESULTS_DIR/"LEADERBOARD.md").write_text(md)
        print(f"\nLeaderboard saved: {RESULTS_DIR/'LEADERBOARD.md'}")
        if HAS_MATPLOTLIB:
            gen_reward_curves(all_results)
    return all_results

def gen_leaderboard_md(results):
    by_model = {}
    for r in results:
        mk = r["model_key"]
        display = r.get("model_display", mk)
        if display not in by_model: by_model[display] = {}
        by_model[display][r["task_id"]] = r["task_score"]
    rows = []
    for display, scores in by_model.items():
        mean = round(sum(scores.values())/len(scores),4) if scores else 0
        rows.append((display, scores.get("task_easy","TBD"),
            scores.get("task_medium","TBD"), scores.get("task_hard","TBD"), mean))
    rows.sort(key=lambda x: x[4], reverse=True)
    lines = ["# ERTriageEnv Model Leaderboard\n",
             "| Model | Task 1 Easy | Task 2 Medium | Task 3 Hard | Mean |",
             "|-------|-------------|---------------|-------------|------|"]
    for d, e, m, h, mean in rows:
        lines.append(f"| {d} | {e} | {m} | {h} | **{mean}** |")
    lines.append("\n*Seed=42 for reproducibility.*")
    return "\n".join(lines)

def gen_reward_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    tasks = ["task_easy","task_medium","task_hard"]
    titles = ["Task 1 Easy","Task 2 Medium","Task 3 Hard"]
    colors = {"gpt-4o":"#1d4ed8","gpt-4o-mini":"#0f766e","llama-3.1-70b":"#7c3aed","mistral-7b":"#b45309"}
    for i,(tid,title) in enumerate(zip(tasks,titles)):
        ax = axes[i]
        seen = set()
        for r in [x for x in results if x["task_id"]==tid]:
            mk = r["model_key"]
            if mk in seen: continue
            seen.add(mk)
            rw = r.get("step_rewards",[])
            if rw:
                cum = [sum(rw[:j+1])/(j+1) for j in range(len(rw))]
                ax.plot(cum, label=r.get("model_display",mk),
                    color=colors.get(mk,"#374151"), linewidth=2)
        ax.axhline(y=0.5, linestyle='--', color='#dc2626', alpha=0.5, label='Pass threshold')
        ax.set_title(title,fontsize=11,fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("Mean Reward")
        ax.set_ylim(-0.15,1.10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("ERTriageEnv Agent Performance",fontsize=13,fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR/"reward_curves.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Chart saved: {out}")

def print_table(results):
    print(f"\n{'='*65}")
    print(f" {'Task':<22} {'Model':<16} {'Score':<8} {'Steps':<8} {'Time':>6}")
    print(f"{'-'*65}")
    for r in results:
        print(f" {r['task_id']:<22} {r.get('model_key','?')[:15]:<16} "
              f"{r['task_score']:<8.4f} {r['steps']:<8} {r['duration_seconds']:>5}s")
    if results:
        mean = sum(r['task_score'] for r in results)/len(results)
        print(f"{'-'*65}\n {'MEAN':<38} {mean:.4f}\n{'='*65}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ERTriageEnv Baseline")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--task", default="all")
    ap.add_argument("--leaderboard", action="store_true")
    args = ap.parse_args()
    if args.leaderboard:
        run_leaderboard(seed=args.seed)
    else:
        tasks = None if args.task=="all" else [args.task]
        run_all(args.model, seed=args.seed, tasks=tasks)
