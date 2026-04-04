#!/bin/bash
set -e
echo "=== ERTriageEnv Local Validation ==="
echo ""
PASS=0; FAIL=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "  OK  $1"; PASS=$((PASS+1))
    else
        echo "  FAIL  $1"; FAIL=$((FAIL+1))
    fi
}

echo "--- File Structure ---"
check "inference.py in root" "test -f inference.py"
check "app/main.py" "test -f app/main.py"
check "app/models.py" "test -f app/models.py"
check "app/environment.py" "test -f app/environment.py"
check "app/grader.py" "test -f app/grader.py"
check "app/reward.py" "test -f app/reward.py"
check "app/patient_gen.py" "test -f app/patient_gen.py"
check "openenv.yaml" "test -f openenv.yaml"
check "Dockerfile" "test -f Dockerfile"
check "requirements.txt" "test -f requirements.txt"
check "static/dashboard.html" "test -f static/dashboard.html"

echo ""
echo "--- Python Imports ---"
check "app.models importable" "python -c 'from app.models import ERState, TriageAction, TriageReward'"
check "app.grader importable" "python -c 'from app.grader import calculate_esi'"
check "app.environment importable" "python -c 'from app.environment import env'"
check "inference.py importable" "python -c 'import inference'"

echo ""
echo "--- Unit Tests ---"
if pytest tests/ -v --tb=short -q > /tmp/pytest_out.txt 2>&1; then
    echo "  OK  All tests pass"; PASS=$((PASS+1))
else
    echo "  FAIL  Some tests failed:"; tail -20 /tmp/pytest_out.txt; FAIL=$((FAIL+1))
fi

echo ""
echo "--- Starting Server ---"
uvicorn app.main:app --port 7860 &
SERVER_PID=$!
sleep 4

echo ""
echo "--- API Endpoint Tests ---"
check "GET /health returns 200" "curl -sf http://localhost:7860/health"
check "GET /reset task_easy returns 200" "curl -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42'"
check "GET /reset task_medium returns 200" "curl -sf 'http://localhost:7860/reset?task_id=task_medium&seed=42'"
check "GET /reset task_hard returns 200" "curl -sf 'http://localhost:7860/reset?task_id=task_hard&seed=42'"
check "GET /state returns 200" "curl -sf http://localhost:7860/state"
check "GET /tasks returns 200" "curl -sf http://localhost:7860/tasks"
check "GET / dashboard returns 200" "curl -sf http://localhost:7860/"
check "response has episode_done field" "curl -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42' | python -c \"import sys,json; d=json.load(sys.stdin); assert 'episode_done' in d\""
check "reset returns patients_waiting" "curl -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42' | python -c \"import sys,json; d=json.load(sys.stdin); assert len(d['patients_waiting'])>0\""

echo ""
echo "--- Reproducibility Check ---"
S1=$(curl -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42' | python -c "import sys,json; d=json.load(sys.stdin); print(d['patients_waiting'][0]['chief_complaint'])")
S2=$(curl -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42' | python -c "import sys,json; d=json.load(sys.stdin); print(d['patients_waiting'][0]['chief_complaint'])")
if [ "$S1" = "$S2" ]; then
    echo "  OK  Reproducibility: same seed produces identical patients"; PASS=$((PASS+1))
else
    echo "  FAIL  Reproducibility failed"; FAIL=$((FAIL+1))
fi

echo ""
echo "--- openenv validate ---"
if command -v openenv &>/dev/null; then
    check "openenv validate passes" "openenv validate --url http://localhost:7860"
else
    echo "  SKIP  openenv not installed (pip install openenv)"
fi

echo ""
echo "--- Docker Build ---"
if command -v docker &>/dev/null; then
    if docker build -t er-triage-test . > /tmp/docker_out.txt 2>&1; then
        echo "  OK  Docker build succeeds"; PASS=$((PASS+1))
    else
        echo "  FAIL  Docker build failed:"; tail -10 /tmp/docker_out.txt; FAIL=$((FAIL+1))
    fi
else
    echo "  SKIP  Docker not installed"
fi

kill $SERVER_PID 2>/dev/null || true

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ $FAIL -eq 0 ]; then
    echo "READY TO SUBMIT"
    exit 0
else
    echo "FIX $FAIL FAILURES BEFORE SUBMITTING"
    exit 1
fi
