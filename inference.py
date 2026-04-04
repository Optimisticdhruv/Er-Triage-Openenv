"""ERTriageEnv inference module for AI agent evaluation.

Uses OpenAI API to perform triage decisions on synthetic patients.
Follows strict JSON logging format for hackathon evaluation.
"""
import os
import json
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

# MANDATORY env variables per problem statement
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEED = int(os.environ.get("SEED", "42"))
ENV_URL = os.environ.get("ENVIRONMENT_URL", "http://localhost:7860")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an experienced emergency room triage nurse.
Assess patients and make a triage decision.

ESI Priority Guidelines:
- Priority 1 (IMMEDIATE): SpO2 less than or equal to 85%, SBP less than 80, GCS less than or equal to 8, cardiac arrest, HR greater than 150 with cardiac/respiratory
- Priority 2 (EMERGENT): SpO2 86-92%, SBP 80-100, GCS 9-13, HR 120+, sepsis signs (fever + tachycardia)
- Priority 3 (URGENT): Stable but needs workup. Multiple borderline vitals.
- Priority 4 (LESS URGENT): Near-normal vitals, one simple intervention.
- Priority 5 (NON-URGENT): Minor issues, can wait.

Bed assignment: resus for P1, acute for P2-3, minor for P3-4, waiting_area for P4-5.
Always escalate (request physician) for all P1 and P2 patients.

RESPOND ONLY WITH VALID JSON — NO TEXT BEFORE OR AFTER:
{
  "patient_id": "P001",
  "priority": 2,
  "bed_type": "acute",
  "escalate": true,
  "reasoning": "Brief one-sentence clinical reasoning"
}"""

def format_patient(obs: dict) -> str:
    """Format patient observation for AI prompt."""
    v = obs.get("vitals", {})
    alerts = obs.get("active_alerts", [])
    alert_str = " | ALERTS: " + ", ".join(alerts) if alerts else ""
    return (
        f"Patient ID: {obs['patient_id']}\n"
        f"Age: {obs['age']} | Sex: {obs['sex']} | Arrival: {obs['arrival_mode'].replace('_',' ')}\n"
        f"Chief Complaint: {obs['chief_complaint']}\n"
        f"Vitals: HR={v.get('heart_rate')}bpm | "
        f"BP={v.get('bp_systolic')}/{v.get('bp_diastolic')}mmHg | "
        f"SpO2={v.get('spo2')}% | RR={v.get('respiratory_rate')}/min | "
        f"Temp={v.get('temperature')}C | Pain={v.get('pain_score')}/10 | "
        f"GCS={v.get('gcs')}\n"
        f"Waiting: {obs.get('waiting_minutes', 0):.1f} min{alert_str}"
    )

def get_action(patient_obs: dict) -> dict:
    """Get AI triage decision for a patient."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_patient(patient_obs)}
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return {
            "patient_id": patient_obs["patient_id"],
            "priority": 3,
            "bed_type": "minor",
            "escalate": False,
            "reasoning": "parse_error_fallback"
        }

def run_task(task_id: str) -> dict:
    """Run a single task and return results."""
    resp = requests.get(f"{ENV_URL}/reset", params={"task_id": task_id, "seed": SEED})
    resp.raise_for_status()
    state = resp.json()
    step_num = 0
    step_rewards = []
    start_ts = time.time()
    while not state.get("episode_done", False) and state.get("patients_waiting"):
        patient = state["patients_waiting"][0]
        action = get_action(patient)
        step_resp = requests.post(f"{ENV_URL}/step", json=action,
            headers={"Content-Type": "application/json"})
        step_resp.raise_for_status()
        result = step_resp.json()
        reward_total = result["reward"]["total"]
        step_rewards.append(reward_total)
        step_num += 1
        # MANDATORY [STEP] log format
        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "step": step_num,
            "patient_id": action["patient_id"],
            "action": {
                "priority": action["priority"],
                "bed_type": action["bed_type"],
                "escalate": action["escalate"],
            },
            "reward": round(reward_total, 4),
            "cumulative_reward": round(result["observation"]["cumulative_reward"], 4),
            "done": result["done"],
        }), flush=True)
        state = result["observation"]
        if step_num >= 100:
            break
    duration = round(time.time() - start_ts, 2)
    mean_r = round(sum(step_rewards)/len(step_rewards), 4) if step_rewards else 0.0
    return {
        "task_id": task_id, "model": MODEL_NAME, "seed": SEED,
        "steps": step_num, "mean_reward": mean_r,
        "task_score": round(max(0.0, min(1.0, mean_r)), 4),
        "step_rewards": step_rewards, "duration_seconds": duration,
    }

def main():
    """Main inference runner."""
    run_id = f"run_{int(time.time())}"
    tasks = ["task_easy", "task_medium", "task_hard"]
    # MANDATORY [START] log
    print(json.dumps({
        "type": "[START]",
        "run_id": run_id,
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "environment_url": ENV_URL,
        "seed": SEED,
        "tasks": tasks,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)
    all_results = []
    overall_start = time.time()
    for task_id in tasks:
        print(json.dumps({"type": "[STEP]", "event": "task_start", "task_id": task_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        result = run_task(task_id)
        all_results.append(result)
        print(json.dumps({"type": "[STEP]", "event": "task_complete",
            "task_id": task_id, "task_score": result["task_score"],
            "steps": result["steps"], "duration_seconds": result["duration_seconds"]}), flush=True)
    mean_score = round(sum(r["task_score"] for r in all_results)/len(all_results), 4)
    # MANDATORY [END] log
    print(json.dumps({
        "type": "[END]",
        "run_id": run_id,
        "model": MODEL_NAME,
        "seed": SEED,
        "total_duration_seconds": round(time.time()-overall_start, 2),
        "results": {r["task_id"]: {"task_score": r["task_score"],
            "mean_reward": r["mean_reward"], "steps": r["steps"],
            "duration_seconds": r["duration_seconds"]} for r in all_results},
        "mean_score_all_tasks": mean_score,
        "status": "completed",
    }), flush=True)
    return all_results

if __name__ == "__main__":
    results = main()
    print(f"\nBaseline complete. Mean score: {sum(r['task_score'] for r in results)/len(results):.4f}", file=sys.stderr)
