# ERTriageEnv — Final Pre-Submission Checklist

## Critical (disqualification if any fail)
- [ ] inference.py is in ROOT directory (not in any subfolder)
- [ ] inference.py uses OpenAI client with API_BASE_URL and MODEL_NAME env vars
- [ ] inference.py emits [START], [STEP], [END] structured JSON to stdout
- [ ] API_BASE_URL, MODEL_NAME, HF_TOKEN are all defined as env variables
- [ ] python inference.py completes all 3 tasks in under 20 minutes
- [ ] docker build succeeds cleanly
- [ ] docker run starts server responding on port 7860
- [ ] GET /health returns HTTP 200
- [ ] GET /reset returns valid ERState JSON
- [ ] POST /step returns valid StepResult with reward in [-1.0, 1.0]
- [ ] GET /state returns valid ERState JSON
- [ ] openenv validate passes against live URL
- [ ] HF Space URL is public and accessible
- [ ] HF Space GET /reset returns 200 (automated ping by judges)

## Quality checks
- [ ] All 3 tasks loadable (task_easy, task_medium, task_hard)
- [ ] All rewards in [-1.0, 1.0] range
- [ ] Same seed=42 produces identical patients on every reset
- [ ] pytest tests/ all pass
- [ ] README has baseline scores filled in (not all TBD)
- [ ] Dashboard loads at /
- [ ] API docs load at /docs
- [ ] No esi_ground_truth or deterioration_risk in PatientObservation

## How to verify
Run: bash validate_local.sh
Expected output: READY TO SUBMIT (0 failures)

Then run:
python inference.py > /tmp/inference_out.txt 2>&1
grep "\[START\]" /tmp/inference_out.txt | wc -l   # should be 1
grep "\[END\]" /tmp/inference_out.txt | wc -l     # should be 1
grep "\[STEP\]" /tmp/inference_out.txt | wc -l    # should be > 1

Then run: python baseline/generate_charts.py
(fills TBD values in README.md with real scores)

Then run: python deploy_to_hf.py
(deploys to Hugging Face, add secrets in Settings tab)
