import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from app.models import Patient, VitalSigns, TriageAction
from app.grader import calculate_esi, grade_single_action, grade_task_easy
from app.patient_gen import generate_patient_batch
from app.reward import calculate_reward, PENALTY_TABLE

def make_vitals(**kwargs):
    defaults = dict(heart_rate=80, bp_systolic=120, bp_diastolic=78,
        spo2=98, respiratory_rate=14, temperature=37.0, pain_score=3, gcs=15)
    defaults.update(kwargs)
    return VitalSigns(**defaults)

def make_patient(age=40, sex='M', category='MINOR', complaint='ankle sprain',
                 esi_gt=4, **vital_kwargs):
    v = make_vitals(**vital_kwargs)
    return Patient(patient_id='TEST001', name='Test Patient', age=age, sex=sex,
        chief_complaint=complaint, complaint_category=category,
        arrival_time=0.0, arrival_mode='walk_in', vitals=v,
        esi_ground_truth=esi_gt, waiting_minutes=0.0, status='waiting',
        deterioration_risk=0.0)

def make_action(priority=3, bed='minor', escalate=False):
    return TriageAction(patient_id='TEST001', priority=priority,
        bed_type=bed, escalate=escalate)

def test_esi_p1_spo2_84():
    assert calculate_esi(make_patient(spo2=84)) == 1

def test_esi_p1_spo2_85():
    assert calculate_esi(make_patient(spo2=85)) == 1

def test_esi_p1_gcs_7():
    assert calculate_esi(make_patient(gcs=7)) == 1

def test_esi_p1_gcs_8():
    assert calculate_esi(make_patient(gcs=8)) == 1

def test_esi_p1_sbp_79_adult():
    assert calculate_esi(make_patient(age=40, bp_systolic=79)) == 1

def test_esi_p1_cardiac_high_hr():
    p = make_patient(category='CARDIAC', heart_rate=155, complaint='chest pain', age=50)
    assert calculate_esi(p) == 1

def test_esi_p2_gcs_11():
    assert calculate_esi(make_patient(gcs=11)) == 2

def test_esi_p2_spo2_89():
    assert calculate_esi(make_patient(spo2=89)) == 2

def test_esi_p2_sbp_92_adult():
    assert calculate_esi(make_patient(age=45, bp_systolic=92)) == 2

def test_esi_p2_sepsis_screen():
    p = make_patient(temperature=40.0, heart_rate=120, category='INFECTION',
        complaint='fever with rigors')
    assert calculate_esi(p) == 2

def test_esi_p4_all_normal():
    p = make_patient(spo2=99, heart_rate=76, bp_systolic=122, gcs=15, pain_score=3)
    assert calculate_esi(p) in (4, 5)

def test_esi_returns_valid_range():
    batch = generate_patient_batch(50, seed=42)
    for pat in batch:
        esi = calculate_esi(pat)
        assert 1 <= esi <= 5

def test_grade_perfect_p1():
    patient = make_patient(spo2=80, gcs=7, esi_gt=1)
    patient.esi_ground_truth = 1
    score = grade_task_easy(patient, make_action(priority=1, bed='resus', escalate=True))
    assert score >= 0.90, f"Perfect P1 action scored too low: {score}"

def test_grade_p1_as_p3_penalised():
    patient = make_patient(spo2=82, esi_gt=1)
    patient.esi_ground_truth = 1
    score = grade_task_easy(patient, make_action(priority=3, bed='minor', escalate=False))
    assert score <= 0.30

def test_grade_p1_as_p4_severe_penalty():
    patient = make_patient(spo2=80, esi_gt=1)
    patient.esi_ground_truth = 1
    score = grade_task_easy(patient, make_action(priority=4, bed='waiting_area', escalate=False))
    assert score <= 0.10

def test_grade_all_scores_in_range():
    batch = generate_patient_batch(20, seed=99)
    for pat in batch:
        pat.esi_ground_truth = calculate_esi(pat)
        for pr in [1, 2, 3, 4]:
            a = TriageAction(patient_id=pat.patient_id, priority=pr,
                bed_type='minor', escalate=(pr<=2))
            score = grade_task_easy(pat, a)
            assert 0.0 <= score <= 1.0

def test_reward_loop_guard():
    patient = make_patient(esi_gt=3)
    patient.esi_ground_truth = 3
    reward = calculate_reward(patient, make_action(), {"TEST001"})
    assert reward.penalty == PENALTY_TABLE["loop_guard"]
    assert reward.total == -0.10

def test_reward_total_in_range():
    batch = generate_patient_batch(30, seed=42)
    for pat in batch:
        pat.esi_ground_truth = calculate_esi(pat)
        for pr in [1, 2, 3, 4]:
            a = TriageAction(patient_id=pat.patient_id, priority=pr,
                bed_type='minor', escalate=False)
            r = calculate_reward(pat, a, set())
            assert -1.0 <= r.total <= 1.0

def test_reproducibility():
    b1 = generate_patient_batch(15, seed=42)
    b2 = generate_patient_batch(15, seed=42)
    for p1, p2 in zip(b1, b2):
        assert p1.patient_id == p2.patient_id
        assert p1.chief_complaint == p2.chief_complaint
        assert p1.vitals.spo2 == p2.vitals.spo2

@given(st.integers(min_value=60, max_value=100))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_no_crash_any_spo2(spo2):
    p = make_patient(spo2=spo2)
    esi = calculate_esi(p)
    assert 1 <= esi <= 5

@given(st.integers(min_value=1, max_value=4))
def test_no_crash_any_priority(priority):
    patient = make_patient(spo2=98, esi_gt=priority)
    patient.esi_ground_truth = priority
    a = TriageAction(patient_id='TEST001', priority=priority,
        bed_type='minor', escalate=False)
    assert isinstance(grade_task_easy(patient, a), float)
