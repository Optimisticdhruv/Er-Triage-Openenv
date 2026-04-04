import pytest
from app.environment import EREnvironment, TASK_CONFIGS
from app.models import TriageAction

def test_reset_task_easy_returns_1_patient():
    env = EREnvironment()
    state = env.reset("task_easy", seed=42)
    assert state.total_patients == 1
    assert len(state.patients_waiting) == 1
    assert state.episode_done == False
    assert state.task_id == "task_easy"

def test_reset_task_medium_returns_15():
    assert EREnvironment().reset("task_medium", seed=42).total_patients == 15

def test_reset_task_hard_returns_40():
    assert EREnvironment().reset("task_hard", seed=42).total_patients == 40

def test_step_returns_valid_step_result():
    env = EREnvironment()
    state = env.reset("task_easy", seed=42)
    p = state.patients_waiting[0]
    result = env.step(TriageAction(patient_id=p.patient_id,
        priority=2, bed_type="acute", escalate=True))
    assert -1.0 <= result.reward.total <= 1.0
    assert isinstance(result.done, bool)
    assert result.observation is not None
    assert isinstance(result.info, dict)

def test_episode_done_after_single_patient():
    env = EREnvironment()
    state = env.reset("task_easy", seed=42)
    result = env.step(TriageAction(
        patient_id=state.patients_waiting[0].patient_id,
        priority=1, bed_type="resus", escalate=True))
    assert result.done == True

def test_invalid_patient_id_raises():
    env = EREnvironment()
    env.reset("task_easy", seed=42)
    with pytest.raises(ValueError, match="Unknown patient_id"):
        env.step(TriageAction(patient_id="INVALID_XYZ",
            priority=1, bed_type="resus", escalate=True))

def test_invalid_task_raises():
    with pytest.raises(ValueError):
        EREnvironment().reset("task_nonexistent", seed=42)

def test_reproducibility_across_resets():
    env = EREnvironment()
    s1 = env.reset("task_medium", seed=42)
    ids1 = [p.patient_id for p in s1.patients_waiting]
    s2 = env.reset("task_medium", seed=42)
    ids2 = [p.patient_id for p in s2.patients_waiting]
    assert ids1 == ids2

def test_no_esi_ground_truth_in_observation():
    env = EREnvironment()
    state = env.reset("task_medium", seed=42)
    for obs in state.patients_waiting:
        assert not hasattr(obs, "esi_ground_truth"), "esi_ground_truth leaked to agent!"

def test_inference_py_at_root():
    from pathlib import Path
    assert Path("inference.py").exists(), (
        "inference.py NOT IN ROOT — DISQUALIFICATION CONDITION")

def test_cumulative_reward_accumulates():
    env = EREnvironment()
    state = env.reset("task_medium", seed=42)
    total = 0.0
    for _ in range(3):
        if not state.patients_waiting: break
        p = state.patients_waiting[0]
        result = env.step(TriageAction(patient_id=p.patient_id,
            priority=2, bed_type="acute", escalate=True))
        total += result.reward.total
        state = result.observation
    assert abs(state.cumulative_reward - round(total, 4)) < 0.01
