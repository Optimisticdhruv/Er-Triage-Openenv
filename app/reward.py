"""ERTriageEnv reward calculation system for triage decisions.

Implements structured 4-signal reward system:
- Priority accuracy (0.0-0.50): ESI level matching
- Bed assignment (0.0-0.30): Correct bed type selection  
- Escalation timing (0.0-0.20): Appropriate escalation for high-acuity
- Penalty (≤0.0): Dangerous errors and invalid actions

Total reward range: -1.0 to 1.0
"""
from app.models import Patient, TriageAction, TriageReward
from app.grader import grade_single_action, calculate_esi


PRIORITY_SCORES = {0: 0.50, 1: 0.25, 2: 0.10, 3: 0.0}

PENALTY_TABLE = {
    "p1_as_p3":   -0.30,
    "p1_as_p4":   -0.50,
    "loop_guard": -0.10,
    "invalid_bed": -0.05,
}

VALID_BEDS = {"resus", "acute", "minor", "waiting_area"}


def calculate_reward(patient: Patient, action: TriageAction,
                     already_triaged_ids: set) -> TriageReward:
    """Calculate 4-signal structured reward for one triage action.
    
    Reward components:
    - priority_accuracy: 0.0-0.50 (ESI matching)
    - bed_assignment: 0.0-0.30 (correct bed type)
    - escalation_timing: 0.0-0.20 (appropriate escalation)
    - penalty: ≤0.0 (dangerous errors, invalid actions)
    
    Total range: -1.0 to 1.0
    
    Args:
        patient: Patient with ground truth ESI level
        action: Agent's triage action to evaluate
        already_triaged_ids: Set of patient IDs already triaged this episode
        
    Returns:
        TriageReward with detailed breakdown and explanation
    """
    # Loop guard - prevent re-triaging same patient
    if action.patient_id in already_triaged_ids:
        return TriageReward(
            total=-0.10, priority_accuracy=0.0, bed_assignment=0.0,
            escalation_timing=0.0, penalty=PENALTY_TABLE["loop_guard"],
            explanation=f"LOOP GUARD: {action.patient_id} already triaged. Penalty: -0.10"
        )

    bed_invalid = action.bed_type not in VALID_BEDS
    bed_invalid_pen = PENALTY_TABLE["invalid_bed"] if bed_invalid else 0.0

    grade = grade_single_action(patient, action)
    esi_gt = grade["esi_ground_truth"]
    delta = grade["priority_delta"]

    # Signal 1: priority_accuracy
    priority_score = PRIORITY_SCORES.get(min(delta, 3), 0.0)

    # Signal 2: bed_assignment
    if bed_invalid:
        bed_score = 0.0
    elif grade["bed_correct"]:
        bed_score = 0.30
    elif delta <= 1:
        bed_score = 0.10
    else:
        bed_score = 0.0
    bed_score = max(0.0, bed_score + bed_invalid_pen)

    # Signal 3: escalation_timing
    if grade["escalation_correct"]:
        esc_score = 0.20
    elif esi_gt <= 2 and not action.escalate:
        esc_score = 0.05
    else:
        esc_score = 0.0

    # Signal 4: life-threat penalty
    life_penalty = 0.0
    if grade["is_fatal_miss"]:
        life_penalty = PENALTY_TABLE["p1_as_p4"]
    elif grade["is_p1_miss"]:
        life_penalty = PENALTY_TABLE["p1_as_p3"]

    total_penalty = life_penalty + bed_invalid_pen
    total = round(max(-1.0, min(1.0, priority_score + bed_score + esc_score + life_penalty)), 4)

    explanation = (
        f"Priority: agent={action.priority} truth={esi_gt} delta={delta} -> {priority_score:.2f} | "
        f"Bed: {action.bed_type} (correct={grade['correct_bed']}) -> {bed_score:.2f} | "
        f"Escalate: {action.escalate} -> {esc_score:.2f} | "
        f"Penalty: {total_penalty:.2f} | Total: {total:.4f}"
    )

    return TriageReward(
        total=total,
        priority_accuracy=round(priority_score, 4),
        bed_assignment=round(bed_score, 4),
        escalation_timing=round(esc_score, 4),
        penalty=round(total_penalty, 4),
        explanation=explanation
    )


def calculate_episode_summary(rewards: list) -> dict:
    """Generate summary statistics for a complete episode.
    
    Provides performance metrics for episode evaluation and logging.
    Includes step counts, reward statistics, and error tracking.
    
    Args:
        rewards: List of TriageReward objects from the episode
        
    Returns:
        Dictionary with episode summary statistics
    """
    if not rewards:
        return {"message": "no actions taken", "steps": 0}
    totals = [r.total for r in rewards]
    return {
        "steps": len(rewards),
        "mean_reward": round(sum(totals)/len(totals), 4),
        "total_reward": round(sum(totals), 4),
        "task_score": round(max(0.0, min(1.0, sum(totals)/len(totals))), 4),
        "dangerous_errors": sum(1 for r in rewards if r.penalty <= -0.30),
        "perfect_actions": sum(1 for r in rewards if r.total >= 0.90),
        "pass_threshold_met": (sum(totals)/len(totals) >= 0.5),
    }
