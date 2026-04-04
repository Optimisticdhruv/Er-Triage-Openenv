"""ERTriageEnv grading system implementing AHRQ Emergency Severity Index (ESI) Version 4.

Citation: AHRQ Emergency Severity Index (ESI) Version 4 — public domain.
This implementation follows the official ESI algorithm for emergency department triage.
"""
from app.models import Patient, TriageAction, TaskConfig


CORRECT_BED_MAP = {1: "resus", 2: "acute", 3: "acute", 4: "minor", 5: "waiting_area"}


def calculate_esi(patient: Patient) -> int:
    """ESI Version 4 algorithm. Returns int 1-5.
    
    Implements the AHRQ ESI v4 algorithm with the following steps:
    1. Check for immediate life-threatening conditions (ESI-1)
    2. Check for high-risk conditions (ESI-2)
    3. Count resources needed (ESI-3/4/5)
    
    Args:
        patient: Patient object with vitals and complaint information
        
    Returns:
        ESI level (1-5, where 1 is most urgent)
    """
    v = patient.vitals
    category = patient.complaint_category.upper()
    complaint_lower = patient.chief_complaint.lower()

    # STEP 1 — ESI-1 check (any single criterion = ESI-1)
    critical_keywords = ["cardiac arrest", "unresponsive", "respiratory failure",
        "pulseless", "not breathing"]
    if v.gcs <= 8: return 1
    if v.spo2 <= 85: return 1
    if patient.age >= 18 and v.bp_systolic < 80: return 1
    if v.heart_rate > 150 and category in ("CARDIAC", "RESPIRATORY"): return 1
    if any(kw in complaint_lower for kw in critical_keywords): return 1

    # STEP 2 — ESI-2 check (any single criterion = ESI-2)
    if 9 <= v.gcs <= 13: return 2
    if 86 <= v.spo2 <= 92: return 2
    if patient.age >= 18 and 80 <= v.bp_systolic <= 100: return 2
    if v.heart_rate >= 120: return 2
    if v.pain_score >= 8 and category in ("CARDIAC","NEUROLOGICAL","ABDOMINAL"): return 2
    if v.temperature > 39.5 and v.heart_rate > 118: return 2
    if patient.age >= 65 and "chest" in complaint_lower and v.pain_score >= 6: return 2

    # STEP 3 — Resource count for ESI 3/4/5
    resources = 0
    if v.spo2 < 96: resources += 1
    if v.heart_rate > 100 or v.heart_rate < 55: resources += 1
    if v.temperature > 38.5: resources += 1
    if v.pain_score >= 5: resources += 1
    if category in ("CARDIAC","RESPIRATORY","NEUROLOGICAL"): resources += 1
    if v.bp_systolic < 105 and patient.age >= 18: resources += 1

    if resources >= 2: return 3
    if resources == 1: return 4
    return 5


def grade_single_action(patient: Patient, action: TriageAction) -> dict:
    """Pure function for grading a single triage action.
    
    Compares agent's triage decision against the ground truth ESI level
    and evaluates correctness of priority, bed assignment, and escalation.
    
    Args:
        patient: Patient with ground truth ESI
        action: Agent's triage action
        
    Returns:
        Dictionary with grading metrics and boolean flags
    """
    esi_gt = patient.esi_ground_truth
    delta = abs(action.priority - esi_gt)
    correct_bed = CORRECT_BED_MAP.get(esi_gt, "minor")
    return {
        "esi_ground_truth": esi_gt,
        "agent_priority": action.priority,
        "priority_delta": delta,
        "priority_correct": delta == 0,
        "bed_correct": action.bed_type == correct_bed,
        "correct_bed": correct_bed,
        "escalation_correct": action.escalate == (esi_gt <= 2),
        "is_p1_miss": esi_gt == 1 and action.priority >= 3,
        "is_fatal_miss": esi_gt == 1 and action.priority == 4,
        "is_undertriage": action.priority > esi_gt,
        "undertriage_severity": max(0, action.priority - esi_gt),
    }


def grade_task_easy(patient: Patient, action: TriageAction) -> float:
    """Single-patient grader for easy task. Returns float 0.0-1.0.
    
    Scoring system:
    - Priority accuracy: exact=+0.50, off-by-1=+0.25, off-by-2=+0.10
    - Bed assignment: correct=+0.30, near-correct=+0.10
    - Escalation: correct=+0.20
    - P1 as P3 penalty: -0.30
    - P1 as P4 penalty: -0.50
    
    Args:
        patient: Single patient with ground truth ESI
        action: Agent's triage action
        
    Returns:
        Score between 0.0 and 1.0
    """
    grade = grade_single_action(patient, action)
    PRIORITY_SCORES = {0: 0.50, 1: 0.25, 2: 0.10, 3: 0.0}
    score = PRIORITY_SCORES.get(min(grade["priority_delta"], 3), 0.0)
    if grade["bed_correct"]:
        score += 0.30
    elif grade["priority_delta"] <= 1:
        score += 0.10
    if grade["escalation_correct"]:
        score += 0.20
    if grade["is_fatal_miss"]:
        score -= 0.50
    elif grade["is_p1_miss"]:
        score -= 0.30
    return round(max(0.0, min(1.0, score)), 4)


def grade_task_medium(patients: list, actions: list, sim_time: float = 0.0) -> float:
    """15-patient shift grader. Returns float 0.0-1.0.
    
    Scoring system:
    - Base score: mean of per-patient easy task scores
    - P1 miss penalty: -0.15 per P1 triaged as P3 or P4
    - Fast P1 bonus: +0.05 if ALL P1 patients triaged in first 3 steps
    
    Args:
        patients: List of patients in the shift
        actions: List of agent actions in order
        sim_time: Total simulation time (unused in current implementation)
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not actions: return 0.0
    paired = list(zip(patients, actions))
    scores = [grade_task_easy(pt, ac) for pt, ac in paired]
    base = sum(scores) / len(scores)
    p1_misses = sum(1 for pt, ac in paired if pt.esi_ground_truth==1 and ac.priority>=3)
    penalty = p1_misses * 0.15
    p1_ids = {pt.patient_id for pt in patients if pt.esi_ground_truth == 1}
    p1_positions = [i for i,(pt,_) in enumerate(paired) if pt.patient_id in p1_ids]
    bonus = 0.05 if (p1_positions and max(p1_positions) < 3) else 0.0
    return round(max(0.0, min(1.0, base - penalty + bonus)), 4)


def grade_task_hard(patients: list, actions: list, resus_used: int = 0) -> float:
    """40-patient MCI grader. Returns float 0.0-1.0.
    
    Scoring system for mass casualty incidents:
    - Base score: mean of per-patient easy task scores
    - Resuscitation overuse penalty: -0.05 per resus assignment beyond 5
    - Death penalty: -0.10 per P1 triaged as P3 or P4
    
    Args:
        patients: List of patients in the MCI
        actions: List of agent actions in order
        resus_used: Number of resuscitation beds assigned
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not actions: return 0.0
    paired = list(zip(patients, actions))
    scores = [grade_task_easy(pt, ac) for pt, ac in paired]
    base = sum(scores) / len(scores)
    overuse = max(0, resus_used - 5) * 0.05
    deaths = sum(1 for pt,ac in paired if pt.esi_ground_truth==1 and ac.priority>=3)
    return round(max(0.0, min(1.0, base - overuse - deaths*0.10)), 4)
