# ERTriageEnv Local Validation Script for PowerShell
Write-Host "=== ERTriageEnv Local Validation ===" -ForegroundColor Cyan
Write-Host ""
$PASS = 0; $FAIL = 0

function Check-Test {
    param($Name, $Command)
    try {
        if ($Command -like "*Test-Path*") {
            $result = Invoke-Expression $Command -ErrorAction Stop
            if ($result) {
                Write-Host "  OK  $Name" -ForegroundColor Green
                $script:PASS++
            } else {
                Write-Host "  FAIL  $Name" -ForegroundColor Red
                $script:FAIL++
            }
        } else {
            $result = Invoke-Expression $Command -ErrorAction Stop
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  OK  $Name" -ForegroundColor Green
                $script:PASS++
            } else {
                Write-Host "  FAIL  $Name" -ForegroundColor Red
                $script:FAIL++
            }
        }
    } catch {
        Write-Host "  FAIL  $Name" -ForegroundColor Red
        $script:FAIL++
    }
}

Write-Host "--- File Structure ---" -ForegroundColor Yellow
Check-Test "inference.py in root" "Test-Path inference.py"
Check-Test "app/main.py" "Test-Path app/main.py"
Check-Test "app/models.py" "Test-Path app/models.py"
Check-Test "app/environment.py" "Test-Path app/environment.py"
Check-Test "app/grader.py" "Test-Path app/grader.py"
Check-Test "app/reward.py" "Test-Path app/reward.py"
Check-Test "app/patient_gen.py" "Test-Path app/patient_gen.py"
Check-Test "openenv.yaml" "Test-Path openenv.yaml"
Check-Test "Dockerfile" "Test-Path Dockerfile"
Check-Test "requirements.txt" "Test-Path requirements.txt"
Check-Test "static/dashboard.html" "Test-Path static/dashboard.html"

Write-Host ""
Write-Host "--- Python Imports ---" -ForegroundColor Yellow
Check-Test "app.models importable" "python -c 'from app.models import ERState, TriageAction, TriageReward'"
Check-Test "app.grader importable" "python -c 'from app.grader import calculate_esi'"
Check-Test "app.environment importable" "python -c 'from app.environment import env'"
Check-Test "inference.py importable" "python -c 'import inference'"

Write-Host ""
Write-Host "--- Unit Tests ---" -ForegroundColor Yellow
$pytestOutput = pytest tests/ -v --tb=short -q 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK  All tests pass" -ForegroundColor Green
    $PASS++
} else {
    Write-Host "  FAIL  Some tests failed:" -ForegroundColor Red
    $pytestOutput | Select-Object -Last 20
    $FAIL++
}

Write-Host ""
Write-Host "--- Starting Server ---" -ForegroundColor Yellow
$serverProcess = Start-Process -FilePath "uvicorn" -ArgumentList "app.main:app", "--port", "7860" -PassThru
Start-Sleep -Seconds 4

Write-Host ""
Write-Host "--- API Endpoint Tests ---" -ForegroundColor Yellow
Check-Test "GET /health returns 200" "curl.exe -sf http://localhost:7860/health"
Check-Test "GET /reset task_easy returns 200" "curl.exe -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42'"
Check-Test "GET /reset task_medium returns 200" "curl.exe -sf 'http://localhost:7860/reset?task_id=task_medium&seed=42'"
Check-Test "GET /reset task_hard returns 200" "curl.exe -sf 'http://localhost:7860/reset?task_id=task_hard&seed=42'"
Check-Test "GET /state returns 200" "curl.exe -sf http://localhost:7860/state"
Check-Test "GET /tasks returns 200" "curl.exe -sf http://localhost:7860/tasks"
Check-Test "GET / dashboard returns 200" "curl.exe -sf http://localhost:7860/"

Write-Host ""
Write-Host "--- Reproducibility Check ---" -ForegroundColor Yellow
try {
    $S1 = curl.exe -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42' 2>$null | python -c "import sys,json; d=json.load(sys.stdin); print(d['patients_waiting'][0]['chief_complaint'])" 2>$null
    $S2 = curl.exe -sf 'http://localhost:7860/reset?task_id=task_easy&seed=42' 2>$null | python -c "import sys,json; d=json.load(sys.stdin); print(d['patients_waiting'][0]['chief_complaint'])" 2>$null
    if ($S1 -eq $S2) {
        Write-Host "  OK  Reproducibility: same seed produces identical patients" -ForegroundColor Green
        $PASS++
    } else {
        Write-Host "  FAIL  Reproducibility failed" -ForegroundColor Red
        $FAIL++
    }
} catch {
    Write-Host "  FAIL  Reproducibility test error" -ForegroundColor Red
    $FAIL++
}

Write-Host ""
Write-Host "--- openenv validate ---" -ForegroundColor Yellow
try {
    $openenvCheck = Get-Command openenv -ErrorAction SilentlyContinue
    if ($openenvCheck) {
        Check-Test "openenv validate passes" "openenv validate --url http://localhost:7860"
    } else {
        Write-Host "  SKIP  openenv not installed (pip install openenv)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  SKIP  openenv not installed (pip install openenv)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "--- Docker Build ---" -ForegroundColor Yellow
try {
    $dockerCheck = Get-Command docker -ErrorAction SilentlyContinue
    if ($dockerCheck) {
        $dockerOutput = docker build -t er-triage-test . 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK  Docker build succeeds" -ForegroundColor Green
            $PASS++
        } else {
            Write-Host "  FAIL  Docker build failed:" -ForegroundColor Red
            $dockerOutput | Select-Object -Last 10
            $FAIL++
        }
    } else {
        Write-Host "  SKIP  Docker not installed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  SKIP  Docker not installed" -ForegroundColor Yellow
}

# Stop server
if ($serverProcess) {
    Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Results: $PASS passed, $FAIL failed ===" -ForegroundColor Cyan
if ($FAIL -eq 0) {
    Write-Host "READY TO SUBMIT" -ForegroundColor Green
    exit 0
} else {
    Write-Host "FIX $FAIL FAILURES BEFORE SUBMITTING" -ForegroundColor Red
    exit 1
}
