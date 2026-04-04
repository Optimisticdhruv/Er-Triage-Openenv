---
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

# ERTriageEnv Hospital
Hospital Emergency Room Triage — OpenEnv AI Agent Training Environment

## Overview
ERTriageEnv is a realistic hospital emergency room triage simulation that addresses the critical problem of triage errors, which cost healthcare systems approximately $4.7 billion annually. The environment provides AI agents with authentic patient scenarios, complete with vital signs, chief complaints, and clinical decision-making challenges that mirror real emergency department workflows.

AI agents interact with ERTriageEnv through a standard reset/step/state loop: they receive patient observations, make triage decisions using Emergency Severity Index (ESI) guidelines, and receive immediate feedback on their performance. The environment tracks cumulative rewards, episode completion, and provides detailed scoring based on established medical protocols.

What makes ERTriageEnv unique is its implementation of ESI v4—the gold standard for emergency department triage—using deterministic, published grading algorithms. This represents the first open interactive ER benchmark that combines medical authenticity with reproducible evaluation, enabling researchers to develop and compare triage AI agents with clinically meaningful metrics.

## Live Demo and API
**HuggingFace Space:** https://huggingface.co/spaces/YOURUSERNAME/er-triage-openenv

### API Endpoints
```bash
# Health check
curl https://YOURUSERNAME-er-triage-openenv.hf.space/health

# Reset environment
curl "https://YOURUSERNAME-er-triage-openenv.hf.space/reset?task_id=task_easy&seed=42"

# Submit triage action
curl -X POST https://YOURUSERNAME-er-triage-openenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","priority":2,"bed_type":"acute","escalate":true,"reasoning":"Chest pain with elevated HR"}'

# Get current state
curl https://YOURUSERNAME-er-triage-openenv.hf.space/state
```

## Quick Start
```bash
git clone https://github.com/YOURUSERNAME/er-triage-openenv
cd er-triage-openenv
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
uvicorn app.main:app --reload --port 7860
# Dashboard: http://localhost:7860
# API docs:  http://localhost:7860/docs
```

## Environment Variables (Mandatory)
| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Yes | OpenAI API key |
| API_BASE_URL | Yes | API base URL (default: https://api.openai.com/v1) |
| MODEL_NAME | Yes | Model identifier (default: gpt-4o-mini) |
| HF_TOKEN | Yes | Hugging Face token (for deployment) |
| OPENROUTER_API_KEY | Recommended | Free multi-model access |
| SEED | Optional | Random seed default 42 |

## Observation Space
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| patient_id | string | P001-P999 | Unique patient identifier |
| age | integer | 0-120 | Patient age |
| sex | string | M or F | Patient sex |
| chief_complaint | string | — | Presenting complaint text |
| arrival_mode | string | ambulance, walk_in, transfer | How patient arrived |
| vitals.heart_rate | integer | 20-250 | Heart rate BPM |
| vitals.bp_systolic | integer | 50-250 | Systolic BP mmHg |
| vitals.bp_diastolic | integer | 30-150 | Diastolic BP mmHg |
| vitals.spo2 | integer | 60-100 | Oxygen saturation percent |
| vitals.respiratory_rate | integer | 4-60 | Breaths per minute |
| vitals.temperature | float | 32.0-43.0 | Temperature Celsius |
| vitals.pain_score | integer | 0-10 | Pain scale |
| vitals.gcs | integer | 3-15 | Glasgow Coma Scale |
| waiting_minutes | float | 0+ | Minutes in queue |
| active_alerts | string[] | — | Deterioration or escalation alerts |

## Action Space
| Field | Type | Valid Values | Description |
|-------|------|--------------|-------------|
| patient_id | string | from waiting list | Patient to triage |
| priority | integer | 1-4 | ESI level: 1=immediate, 4=non-urgent |
| bed_type | string | resus, acute, minor, waiting_area | Bed assignment |
| escalate | boolean | true or false | Request immediate physician |
| reasoning | string | any | Optional, not used by grader |

## Reward Function
| Signal | Weight | How Earned |
|--------|--------|------------|
| priority_accuracy | 0.50 | Exact match=0.50, off-by-1=0.25, off-by-2=0.10 |
| bed_assignment | 0.30 | Correct bed type for ESI level |
| escalation_timing | 0.20 | Escalate=true for ESI-1 and ESI-2 |
| penalty | variable | P1-as-P3: -0.30, P1-as-P4: -0.50, loop guard: -0.10 |

## Tasks

### Task 1: Easy
**Objective:** Single critical patient requiring immediate ESI-1 triage
**Grader Rubric:** Perfect score (1.0) for correct ESI-1 with resus bed and escalation
**Max Steps:** 1 | **Passing Score:** 0.8
**Edge Cases:** SpO2 exactly 85% (ESI-1 threshold), cardiac arrest scenarios

### Task 2: Medium
**Objective:** 15 patients with mixed acuity levels requiring prioritization
**Grader Rubric:** Weighted scoring based on correct ESI levels and bed assignments
**Max Steps:** 15 | **Passing Score:** 0.6
**Edge Cases:** Multiple ESI-2 patients, sepsis screening, resource constraints

### Task 3: Hard
**Objective:** 40 patients simulating high-volume ED surge conditions
**Grader Rubric:** Complex multi-objective scoring with queue management
**Max Steps:** 40 | **Passing Score:** 0.5
**Edge Cases:** Resource depletion, simultaneous critical patients, time pressure

## Baseline Scores
| Model | Task 1 Easy | Task 2 Medium | Task 3 Hard | Mean Score |
|-------|-------------|---------------|-------------|------------|
| GPT-4o | TBD | TBD | TBD | TBD |
| GPT-4o-mini | TBD | TBD | TBD | TBD |
| Llama 3.1 70B | TBD | TBD | TBD | TBD |
| Mistral 7B | TBD | TBD | TBD | TBD |

Note: Run `python inference.py` to generate scores. Seed=42 for reproducibility.

## Running Inference
```bash
export OPENAI_API_KEY="your_key_here"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_hf_token"
python inference.py
```

## Docker
```bash
docker build -t er-triage-openenv .
docker run -p 7860:7860 --env-file .env er-triage-openenv
```

## Validate OpenEnv Spec
```bash
openenv validate --url http://localhost:7860
```

## Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Agent     │───▶│   ERTriage  │───▶│   Reward    │
│             │    │   Env       │    │  Function   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Action    │    │  Observation │    │   Score     │
│  (Triage)   │    │  (Patient)   │    │  (ESI v4)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Citation
Emergency Severity Index Implementation Handbook Version 4. Agency for Healthcare Research and Quality, U.S. Department of Health and Human Services. Public domain.
