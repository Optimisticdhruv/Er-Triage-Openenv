"""ERTriageEnv FastAPI server - Hospital ER Triage OpenEnv Environment.

Provides REST API endpoints for AI agent training and human interaction:
- /reset: Initialize new episode with task configuration
- /step: Execute triage action and get reward
- /state: Get current environment observation
- /health: Health check and status
- /tasks: List available task configurations
- /metrics: Episode performance statistics
- /: Interactive dashboard
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import uvicorn
from pathlib import Path
from app.models import TriageAction, ERState, StepResult, ResetRequest
from app.environment import env, TASK_CONFIGS


APP_NAME = "ERTriageEnv"
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    print("""
===================================================
  ERTriageEnv v1.0.0 - Hospital ER Triage OpenEnv
  Tasks: task_easy | task_medium | task_hard
  API docs:  http://localhost:7860/docs
  Dashboard: http://localhost:7860/
===================================================
    """)
    env.reset(task_id="task_easy", seed=42)
    print("[INIT] Environment ready. Seed=42, task=task_easy")
    yield
    print("[SHUTDOWN] ERTriageEnv stopping.")


app = FastAPI(
    title=APP_NAME, 
    version=APP_VERSION,
    description="Hospital ER Triage OpenEnv AI Agent Training Environment",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse, tags=["dashboard"])
async def dashboard():
    """Serve interactive dashboard."""
    f = Path("static/dashboard.html")
    if f.exists():
        return FileResponse("static/dashboard.html")
    return HTMLResponse("<h1>ERTriageEnv</h1><p>Dashboard loading...</p>")


@app.get("/health", tags=["openenv"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "version": APP_VERSION, 
        "environment": APP_NAME,
        "current_task": env.task_config.task_id if env.task_config else None,
        "episode_id": env.episode_id or None
    }


@app.get("/reset", response_model=ERState, tags=["openenv"])
async def reset_get(task_id: str = "task_easy", seed: int = 42):
    """Reset environment and start new episode (GET method).
    
    Args:
        task_id: Task configuration identifier
        seed: Random seed for reproducible episodes
        
    Returns:
        Initial ERState observation
        
    Raises:
        HTTPException: If task_id is invalid
    """
    if task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}"
        )
    return env.reset(task_id=task_id, seed=seed)

@app.post("/reset", response_model=ERState, tags=["openenv"])
async def reset_post(request: ResetRequest = None, 
                     task_id: str = Query(default="task_easy"),
                     seed: int = Query(default=42)):
    """Reset environment and start new episode (POST method).
    
    Args:
        request: Optional JSON body with task_id and seed
        task_id: Task configuration identifier (from query params if not in body)
        seed: Random seed for reproducible episodes (from query params if not in body)
        
    Returns:
        Initial ERState observation
        
    Raises:
        HTTPException: If task_id is invalid
    """
    # Accept task_id from body OR query params
    actual_task = (request.task_id if request else None) or task_id
    actual_seed = (request.seed if request else None) or seed
    
    if actual_task not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{actual_task}'. Valid: {list(TASK_CONFIGS.keys())}"
        )
    return env.reset(task_id=actual_task, seed=actual_seed)


@app.post("/step", response_model=StepResult, tags=["openenv"])
async def step(action: TriageAction):
    """Execute triage action and return result.
    
    Args:
        action: TriageAction to execute
        
    Returns:
        StepResult with observation, reward, done flag, and info
        
    Raises:
        HTTPException: If action is invalid
    """
    try:
        return env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=ERState, tags=["openenv"])
async def state():
    """Get current environment state without advancing time."""
    return env.state()


@app.get("/tasks", tags=["info"])
async def list_tasks():
    """List available task configurations."""
    return {"tasks": {k: v.model_dump() for k, v in TASK_CONFIGS.items()}}


@app.get("/metrics", tags=["info"])
async def metrics():
    """Get episode performance metrics."""
    history = env.rewards_history
    if not history:
        return {"message": "No actions yet", "steps": 0}
    
    totals = [r.total for r in history]
    return {
        "steps": len(history),
        "mean_reward": round(sum(totals)/len(totals), 4),
        "cumulative_reward": round(env.cumulative_reward, 4),
        "dangerous_errors": sum(1 for r in history if r.penalty <= -0.30),
        "perfect_actions": sum(1 for r in history if r.total >= 0.90),
        "task_score": round(max(0.0, min(1.0, sum(totals)/len(totals))), 4),
        "last_reward": history[-1].model_dump() if history else None,
    }


@app.get("/openenv.yaml", tags=["info"])
async def get_spec():
    """Serve OpenEnv specification file."""
    spec = Path("openenv.yaml")
    if spec.exists():
        return HTMLResponse(content=spec.read_text(), media_type="text/yaml")
    raise HTTPException(status_code=404, detail="openenv.yaml not found")


@app.get("/debug/reset/{task_id}", tags=["debug"])
async def debug_reset(task_id: str):
    """Debug endpoint for troubleshooting reset issues."""
    try:
        result = env.reset(task_id=task_id, seed=42)
        return {"status": "ok", "patients": len(result.patients_waiting)}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# OpenEnv compliance endpoints
@app.get("/metadata", tags=["openenv"])
async def get_metadata():
    """Get environment metadata for OpenEnv compliance."""
    return {
        "name": "ER Triage Environment",
        "description": "Hospital Emergency Room triage simulation where AI agents learn to prioritize patients using the Emergency Severity Index (ESI) system. Agents must assess patient vitals, chief complaints, and allocate appropriate bed resources while managing escalation decisions.",
        "version": "1.0.0",
        "environment_type": "medical_triage",
        "observation_space": {
            "type": "structured",
            "description": "ER state with patient queue, bed occupancy, vitals, and alerts"
        },
        "action_space": {
            "type": "discrete",
            "description": "Triage actions: patient selection, ESI priority (1-4), bed assignment, escalation"
        },
        "reward_range": [-1.0, 1.0],
        "max_episode_length": 100,
        "tags": ["medical", "healthcare", "triage", "emergency", "esi", "hospital"],
        "author": "optimisticdhruv",
        "license": "MIT"
    }


@app.get("/schema", tags=["openenv"])
async def get_schema():
    """Get OpenEnv schema definitions for action, observation, and state."""
    return {
        "action": {
            "type": "object",
            "description": "Triage action with patient selection and priority assignment",
            "required_fields": ["patient_id", "priority", "bed_type"],
            "optional_fields": ["escalate", "reasoning"],
            "constraints": {
                "priority": {"min": 1, "max": 4},
                "bed_type": ["resus", "acute", "minor", "waiting_area"]
            }
        },
        "observation": {
            "type": "object", 
            "description": "Current ER state with patients and beds",
            "key_fields": [
                "task_id", "episode_id", "shift_time",
                "patients_waiting", "beds", "patients_triaged",
                "total_patients", "cumulative_reward", "episode_done"
            ]
        },
        "state": {
            "type": "object",
            "description": "Complete step result with observation, reward, and done flag",
            "fields": ["observation", "reward", "done", "info"]
        }
    }


@app.post("/mcp", tags=["openenv"])
async def mcp_endpoint(request: dict = None):
    """Model Context Protocol endpoint for OpenEnv MCP compliance."""
    if request is None:
        request = {}
    
    # Handle JSON-RPC 2.0 requests
    if "jsonrpc" in request:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": False
                        },
                        "resources": {}
                    },
                    "serverInfo": {
                        "name": "er-triage-openenv",
                        "version": "1.0.0"
                    }
                }
            }
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0", 
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "reset_environment",
                            "description": "Reset the ER triage environment"
                        },
                        {
                            "name": "step_environment", 
                            "description": "Execute a triage action"
                        },
                        {
                            "name": "get_state",
                            "description": "Get current environment state"
                        }
                    ]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                }
            }
    
    # Default response for non-JSON-RPC requests
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "status": "ok",
            "mcp_version": "2024-11-05",
            "server": "er-triage-openenv",
            "message": "MCP endpoint available - use JSON-RPC 2.0 format for tool calls"
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
