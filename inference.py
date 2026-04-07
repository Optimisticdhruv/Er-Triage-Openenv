"""ERTriageEnv inference module for AI agent evaluation.

Uses OpenAI API to perform triage decisions on synthetic patients.
Follows strict OpenEnv hackathon logging format.
"""
import os
import json
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

# Use EXACTLY these variable names - judges inject these
API_BASE_URL = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")

# Initialize client with judge's proxy
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# Other environment variables
SEED = int(os.environ.get("SEED", "42"))
ENV_URL = os.environ.get("ENVIRONMENT_URL", "http://localhost:7860")

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

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={rewards_str}", flush=True)

def format_action(action: dict) -> str:
    """Format action as P{priority}_{bed_type}_{escalate}"""
    priority = action["priority"]
    bed_type = action["bed_type"]
    escalate = "escalate" if action["escalate"] else "no_escalate"
    return f"P{priority}_{bed_type}_{escalate}"
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
            temperature=0.3,  # Allow some variation
            max_tokens=300,    # Increase tokens for better JSON
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        # Intelligent fallback based on vitals
        v = patient_obs.get("vitals", {})
        age = patient_obs.get("age", 30)
        
        # Calculate priority based on vitals
        priority = 3  # default to P3
        bed_type = "minor"
        escalate = False
        
        # ESI-1: Immediate life threats
        if (v.get('spo2', 100) <= 85 or 
            v.get('gcs', 15) <= 8 or
            (v.get('bp_systolic', 120) < 80 and age >= 18) or
            v.get('heart_rate', 80) > 150):
            priority = 1
            bed_type = "resus"
            escalate = True
        
        # ESI-2: High risk
        elif (v.get('spo2', 100) <= 92 or
              v.get('gcs', 15) <= 13 or
              (v.get('bp_systolic', 120) <= 100 and age >= 18) or
              v.get('heart_rate', 80) >= 120 or
              v.get('temperature', 37) > 39.5):
            priority = 2
            bed_type = "acute"
            escalate = True
        
        # ESI-4: Less urgent
        elif (v.get('spo2', 100) >= 96 and
              v.get('heart_rate', 80) >= 55 and v.get('heart_rate', 80) <= 100 and
              v.get('temperature', 37) <= 38.5 and
              v.get('pain_score', 0) < 5):
            priority = 4
            bed_type = "minor"
            escalate = False
        
        # ESI-5: Non-urgent
        elif (v.get('spo2', 100) >= 96 and
              v.get('pain_score', 0) < 3 and
              v.get('temperature', 37) < 38.0):
            priority = 5
            bed_type = "waiting_area"
            escalate = False
        
        return {
            "patient_id": patient_obs["patient_id"],
            "priority": priority,
            "bed_type": bed_type,
            "escalate": escalate,
            "reasoning": "fallback_vitals_based"
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
        
        # Format action string and log step
        action_str = format_action(action)
        log_step(step_num, action_str, reward_total, result["done"])
        
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
    tasks = ["task_easy", "task_medium", "task_hard"]
    
    # Log start for each task
    for task_id in tasks:
        log_start(task_id, "er_triage", MODEL_NAME)
        
    all_results = []
    all_rewards = []
    
    for task_id in tasks:
        result = run_task(task_id)
        all_results.append(result)
        all_rewards.extend(result["step_rewards"])
    
    # Calculate overall score
    overall_score = round(sum(all_rewards)/len(all_rewards), 4) if all_rewards else 0.0
    overall_success = overall_score >= 0.50
    total_steps = sum(r["steps"] for r in all_results)
    
    # Log end
    log_end(overall_success, total_steps, overall_score, all_rewards)
    
    return all_results

if __name__ == "__main__":
    results = main()
    print(f"\nBaseline complete. Mean score: {sum(r['task_score'] for r in results)/len(results):.4f}", file=sys.stderr)
