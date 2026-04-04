"""Synthetic patient generator. All output is fully synthetic — no real patient data."""
from faker import Faker
import numpy as np
import random
from copy import deepcopy
from app.models import Patient, VitalSigns, PatientObservation


CHIEF_COMPLAINTS = [
    # CARDIAC (10 entries, esi_bias 1-2)
    {"complaint": "Crushing chest pain radiating to left arm, diaphoresis", "category": "CARDIAC", "esi_bias": 1, "description": "Classic MI symptoms", "trap": False},
    {"complaint": "Chest tightness sudden onset, age >60", "category": "CARDIAC", "esi_bias": 1, "description": "Acute coronary syndrome", "trap": False},
    {"complaint": "Palpitations with near-syncope, dizzy", "category": "CARDIAC", "esi_bias": 2, "description": "Arrhythmia concern", "trap": False},
    {"complaint": "Syncope, witnessed collapse", "category": "CARDIAC", "esi_bias": 1, "description": "Cardiac syncope", "trap": False},
    {"complaint": "Mild chest discomfort, denies pain — silent MI", "category": "CARDIAC", "esi_bias": 1, "description": "Silent myocardial infarction", "trap": True},
    {"complaint": "Atrial fibrillation with fast ventricular rate", "category": "CARDIAC", "esi_bias": 2, "description": "A-fib with RVR", "trap": False},
    {"complaint": "Chest pain 3 out of 10, patient calm — possible dissection", "category": "CARDIAC", "esi_bias": 1, "description": "Aortic dissection", "trap": True},
    {"complaint": "Exertional chest pain resolved on rest", "category": "CARDIAC", "esi_bias": 2, "description": "Stable angina", "trap": False},
    {"complaint": "Pacemaker malfunction, feeling dizzy", "category": "CARDIAC", "esi_bias": 2, "description": "Device malfunction", "trap": False},
    {"complaint": "Hypertensive urgency, BP 210 over 130", "category": "CARDIAC", "esi_bias": 2, "description": "Severe hypertension", "trap": False},
    
    # RESPIRATORY (8 entries, esi_bias 1-2)
    {"complaint": "Acute severe asthma, cannot complete sentences", "category": "RESPIRATORY", "esi_bias": 1, "description": "Status asthmaticus", "trap": False},
    {"complaint": "Respiratory failure, accessory muscle use", "category": "RESPIRATORY", "esi_bias": 1, "description": "Acute respiratory failure", "trap": False},
    {"complaint": "Mild cough, slightly unwell — SpO2 84 percent", "category": "RESPIRATORY", "esi_bias": 1, "description": "Silent hypoxia", "trap": True},
    {"complaint": "Hemoptysis, coughing up blood", "category": "RESPIRATORY", "esi_bias": 2, "description": "Pulmonary hemorrhage", "trap": False},
    {"complaint": "COPD exacerbation, increased dyspnea", "category": "RESPIRATORY", "esi_bias": 2, "description": "COPD exacerbation", "trap": False},
    {"complaint": "Pulmonary embolism symptoms, pleuritic chest pain", "category": "RESPIRATORY", "esi_bias": 2, "description": "PE suspicion", "trap": False},
    {"complaint": "Pneumothorax suspected, sudden onset SOB", "category": "RESPIRATORY", "esi_bias": 1, "description": "Tension pneumothorax", "trap": False},
    {"complaint": "Post-op day 2, increasing respiratory rate", "category": "RESPIRATORY", "esi_bias": 2, "description": "Post-op respiratory decline", "trap": True},
    
    # NEUROLOGICAL (8 entries, esi_bias 1-2)
    {"complaint": "Stroke symptoms, facial droop, arm weakness, slurred speech", "category": "NEUROLOGICAL", "esi_bias": 1, "description": "Acute stroke", "trap": False},
    {"complaint": "Active tonic-clonic seizure", "category": "NEUROLOGICAL", "esi_bias": 1, "description": "Active seizure", "trap": False},
    {"complaint": "Altered consciousness, GCS 9", "category": "NEUROLOGICAL", "esi_bias": 1, "description": "Decreased consciousness", "trap": False},
    {"complaint": "Worst headache of life, sudden onset", "category": "NEUROLOGICAL", "esi_bias": 2, "description": "Subarachnoid hemorrhage", "trap": False},
    {"complaint": "Confusion in elderly, new onset, looks tired", "category": "NEUROLOGICAL", "esi_bias": 2, "description": "Delirium", "trap": True},
    {"complaint": "Neck stiffness, photophobia, meningism signs", "category": "NEUROLOGICAL", "esi_bias": 2, "description": "Meningitis", "trap": False},
    {"complaint": "Overdose, GCS 10, brought by friend, appears sleepy", "category": "NEUROLOGICAL", "esi_bias": 1, "description": "Drug overdose", "trap": True},
    {"complaint": "TIA symptoms resolved 30 minutes ago", "category": "NEUROLOGICAL", "esi_bias": 2, "description": "Transient ischemic attack", "trap": False},
    
    # TRAUMA (8 entries, esi_bias 2-3)
    {"complaint": "MVA high speed, multiple injuries", "category": "TRAUMA", "esi_bias": 2, "description": "Major trauma", "trap": False},
    {"complaint": "Fall from height greater than 3m", "category": "TRAUMA", "esi_bias": 2, "description": "Significant fall", "trap": False},
    {"complaint": "Stab wound to abdomen", "category": "TRAUMA", "esi_bias": 2, "description": "Penetrating trauma", "trap": False},
    {"complaint": "Head injury with loss of consciousness", "category": "TRAUMA", "esi_bias": 2, "description": "Head trauma", "trap": False},
    {"complaint": "Minor fall, elderly patient on warfarin", "category": "TRAUMA", "esi_bias": 2, "description": "Anticoagulated fall", "trap": True},
    {"complaint": "Laceration forearm actively bleeding", "category": "TRAUMA", "esi_bias": 3, "description": "Active bleeding", "trap": False},
    {"complaint": "Fractured ankle low energy mechanism", "category": "TRAUMA", "esi_bias": 3, "description": "Simple fracture", "trap": False},
    {"complaint": "Blunt chest trauma, rib pain", "category": "TRAUMA", "esi_bias": 3, "description": "Chest wall trauma", "trap": False},
    
    # ABDOMINAL (6 entries, esi_bias 2-3)
    {"complaint": "Severe acute abdomen, rigid belly", "category": "ABDOMINAL", "esi_bias": 2, "description": "Peritonitis", "trap": False},
    {"complaint": "GI bleed, hematemesis", "category": "ABDOMINAL", "esi_bias": 2, "description": "Upper GI bleed", "trap": False},
    {"complaint": "Appendicitis symptoms, RLQ pain 8 out of 10", "category": "ABDOMINAL", "esi_bias": 3, "description": "Appendicitis", "trap": False},
    {"complaint": "Renal colic, severe flank pain", "category": "ABDOMINAL", "esi_bias": 3, "description": "Kidney stone", "trap": False},
    {"complaint": "Abdominal pain elderly patient, silent presentation", "category": "ABDOMINAL", "esi_bias": 2, "description": "Silent abdomen", "trap": True},
    {"complaint": "Bowel obstruction symptoms, abdominal distension", "category": "ABDOMINAL", "esi_bias": 2, "description": "Bowel obstruction", "trap": False},
    
    # INFECTION (6 entries, esi_bias 1-3)
    {"complaint": "Sepsis signs, fever 39.8 HR 128 looks well — normotensive", "category": "INFECTION", "esi_bias": 2, "description": "Early sepsis", "trap": True},
    {"complaint": "Non-blanching petechiae rash, meningococcal suspected", "category": "INFECTION", "esi_bias": 1, "description": "Meningococcemia", "trap": False},
    {"complaint": "Severe cellulitis with systemic features", "category": "INFECTION", "esi_bias": 3, "description": "Cellulitis", "trap": False},
    {"complaint": "Urinary sepsis with rigors", "category": "INFECTION", "esi_bias": 2, "description": "Urosepsis", "trap": False},
    {"complaint": "Diabetic foot infection, tracking cellulitis", "category": "INFECTION", "esi_bias": 3, "description": "Diabetic foot", "trap": False},
    {"complaint": "High fever SpO2 dropping, severe COVID", "category": "INFECTION", "esi_bias": 2, "description": "Severe COVID", "trap": False},
    
    # MINOR (14 entries, esi_bias 3-5)
    {"complaint": "Twisted ankle walking", "category": "MINOR", "esi_bias": 4, "description": "Ankle sprain", "trap": False},
    {"complaint": "Minor finger laceration controlled", "category": "MINOR", "esi_bias": 4, "description": "Minor laceration", "trap": False},
    {"complaint": "UTI symptoms, no fever, frequency and dysuria", "category": "MINOR", "esi_bias": 4, "description": "Uncomplicated UTI", "trap": False},
    {"complaint": "Earache, no fever", "category": "MINOR", "esi_bias": 5, "description": "Ear pain", "trap": False},
    {"complaint": "Mild rash, no systemic features", "category": "MINOR", "esi_bias": 5, "description": "Minor rash", "trap": False},
    {"complaint": "Sore throat, able to swallow", "category": "MINOR", "esi_bias": 5, "description": "Pharyngitis", "trap": False},
    {"complaint": "Chronic back pain, no red flags", "category": "MINOR", "esi_bias": 4, "description": "Chronic pain", "trap": False},
    {"complaint": "Toothache, paracetamol not working", "category": "MINOR", "esi_bias": 5, "description": "Dental pain", "trap": False},
    {"complaint": "Anxiety attack, heart racing, vitals all normal", "category": "MINOR", "esi_bias": 4, "description": "Anxiety", "trap": True},
    {"complaint": "Tension headache, no red flags", "category": "MINOR", "esi_bias": 4, "description": "Tension headache", "trap": False},
    {"complaint": "Viral URTI, runny nose and cough", "category": "MINOR", "esi_bias": 5, "description": "Common cold", "trap": False},
    {"complaint": "Medication refill, no acute issue", "category": "MINOR", "esi_bias": 5, "description": "Medication refill", "trap": False},
    {"complaint": "Minor burn less than 5 percent BSA", "category": "MINOR", "esi_bias": 4, "description": "Minor burn", "trap": False},
    {"complaint": "Insect bite, no anaphylaxis signs", "category": "MINOR", "esi_bias": 5, "description": "Insect bite", "trap": False},
]


def generate_vitals(age: int, category: str, esi_bias: int, rng: np.random.Generator) -> VitalSigns:
    """Generate realistic vital signs based on ESI bias and category."""
    
    # Base distributions by esi_bias
    if esi_bias == 1:
        spo2 = max(60, min(87, round(rng.normal(81, 4))))
        heart_rate = max(130, min(200, round(rng.normal(148, 18))))
        bp_systolic = max(55, min(88, round(rng.normal(78, 10))))
        respiratory_rate = max(28, min(50, round(rng.normal(34, 5))))
        gcs = max(3, min(9, round(rng.normal(7, 2))))
        pain_score = max(0, min(10, round(rng.normal(8.5, 1.5))))
    elif esi_bias == 2:
        spo2 = max(85, min(93, round(rng.normal(89, 3))))
        heart_rate = max(100, min(135, round(rng.normal(118, 12))))
        bp_systolic = max(80, min(105, round(rng.normal(93, 8))))
        respiratory_rate = max(20, min(30, round(rng.normal(24, 4))))
        gcs = max(9, min(14, round(rng.normal(12, 1.5))))
        pain_score = max(0, min(10, round(rng.normal(6.5, 1.5))))
    elif esi_bias == 3:
        spo2 = max(90, min(97, round(rng.normal(94, 2))))
        heart_rate = max(88, min(118, round(rng.normal(102, 10))))
        bp_systolic = max(98, min(138, round(rng.normal(118, 12))))
        respiratory_rate = max(16, min(26, round(rng.normal(20, 3))))
        gcs = 15
        pain_score = max(0, min(10, round(rng.normal(5, 2))))
    else:  # esi_bias >= 4: handles both 4 and 5
        spo2 = max(95, min(100, round(rng.normal(98, 1.5))))
        heart_rate = max(55, min(95, round(rng.normal(76, 12))))
        bp_systolic = max(105, min(148, round(rng.normal(126, 14))))
        respiratory_rate = max(10, min(20, round(rng.normal(15, 2))))
        gcs = 15
        pain_score = max(0, min(10, round(rng.normal(3, 2))))
    
    # Temperature by category
    if category == "INFECTION":
        temperature = round(max(38.0, min(41.0, rng.normal(39.2, 0.6))), 1)
    elif category == "TRAUMA":
        temperature = round(max(36.5, min(37.8, rng.normal(37.1, 0.3))), 1)
    elif category in ["CARDIAC", "RESPIRATORY"]:
        temperature = round(max(36.0, min(38.5, rng.normal(37.0, 0.4))), 1)
    elif category == "MINOR":
        temperature = round(max(36.2, min(37.6, rng.normal(36.8, 0.3))), 1)
    else:
        temperature = round(rng.normal(37.0, 0.5), 1)
    
    # Diastolic blood pressure
    bp_diastolic = max(30, min(110, round(bp_systolic * rng.normal(0.62, 0.05))))
    
    return VitalSigns(
        heart_rate=heart_rate,
        bp_systolic=bp_systolic,
        bp_diastolic=bp_diastolic,
        spo2=spo2,
        respiratory_rate=respiratory_rate,
        temperature=temperature,
        pain_score=pain_score,
        gcs=gcs
    )


def generate_patient(patient_id: str, arrival_time: float, rng: np.random.Generator, fake: Faker) -> Patient:
    """Generate a single synthetic patient with realistic demographics and vitals."""
    
    # Pick random complaint
    complaint = rng.choice(CHIEF_COMPLAINTS)
    
    # Age distribution: 12% children, 38% young adults, 32% middle-aged, 18% elderly
    age_roll = rng.random()
    if age_roll < 0.12:
        age = rng.integers(2, 18)
    elif age_roll < 0.50:
        age = rng.integers(18, 45)
    elif age_roll < 0.82:
        age = rng.integers(45, 65)
    else:
        age = rng.integers(65, 100)
    
    # Elderly with esi_bias=2: 30% chance upgrade to esi_bias=1
    esi_bias = complaint["esi_bias"]
    if age >= 65 and esi_bias == 2 and rng.random() < 0.30:
        esi_bias = 1
    
    # Sex
    sex = rng.choice(['M', 'F'])
    
    # Arrival mode based on acuity
    if esi_bias == 1:
        arrival_mode = "ambulance"
    elif esi_bias == 2:
        mode_roll = rng.random()
        if mode_roll < 0.65:
            arrival_mode = "ambulance"
        elif mode_roll < 0.90:
            arrival_mode = "walk_in"
        else:
            arrival_mode = "transfer"
    else:  # esi_bias >= 3: handles 3, 4, and 5
        mode_roll = rng.random()
        if mode_roll < 0.20:
            arrival_mode = "ambulance"
        elif mode_roll < 0.90:
            arrival_mode = "walk_in"
        else:
            arrival_mode = "transfer"
    
    # Deterioration risk
    if esi_bias == 1:
        deterioration_risk = 0.0
    elif esi_bias == 2:
        deterioration_risk = float(rng.uniform(0.25, 0.45))
    elif esi_bias == 3:
        deterioration_risk = float(rng.uniform(0.08, 0.20))
    else:  # esi_bias >= 4: handles 4 and 5
        deterioration_risk = 0.0
    
    # Generate vitals
    vitals = generate_vitals(age, complaint["category"], esi_bias, rng)
    
    return Patient(
        patient_id=patient_id,
        name=fake.name(),
        age=age,
        sex=sex,
        chief_complaint=complaint["complaint"],
        complaint_category=complaint["category"],
        arrival_time=arrival_time,
        arrival_mode=arrival_mode,
        vitals=vitals,
        esi_ground_truth=1,  # Will be set by environment.py
        waiting_minutes=0.0,
        status="waiting",
        deterioration_risk=deterioration_risk
    )


def generate_patient_batch(n_patients: int, seed: int = 42) -> list[Patient]:
    """Generate a batch of synthetic patients with deterministic output."""
    
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)
    
    patients = []
    
    # Generate patients
    for i in range(n_patients):
        patient_id = f"P{i+1:03d}"
        if i == 0:
            arrival_time = 0.0
        else:
            arrival_time = patients[-1].arrival_time + float(rng.exponential(3.0))
        
        patient = generate_patient(patient_id, arrival_time, rng, fake)
        patients.append(patient)
    
    # For n_patients >= 10: ensure minimum ESI distribution
    if n_patients >= 10:
        # Count current ESI bias distribution
        esi_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for patient in patients:
            complaint = next(c for c in CHIEF_COMPLAINTS if c["complaint"] == patient.chief_complaint)
            esi_counts[complaint["esi_bias"]] += 1
        
        # Resample last slots if needed to meet minimums
        needs_resample = []
        if esi_counts[1] < 1:
            needs_resample.extend([1] * (1 - esi_counts[1]))
        if esi_counts[2] < 2:
            needs_resample.extend([2] * (2 - esi_counts[2]))
        if esi_counts[3] < 3:
            needs_resample.extend([3] * (3 - esi_counts[3]))
        if esi_counts[4] + esi_counts[5] < 2:  # combined low acuity
            needs_resample.extend([4, 5] * 2)
        
        # Replace last patients with required ESI levels
        for i, target_esi in enumerate(needs_resample):
            if i < len(patients):
                # Find complaint with target ESI
                target_complaints = [c for c in CHIEF_COMPLAINTS if c["esi_bias"] == target_esi]
                complaint = rng.choice(target_complaints)
                
                # Regenerate patient with specific complaint
                patient_id = patients[-(i+1)].patient_id
                arrival_time = patients[-(i+1)].arrival_time
                
                # Temporarily set the complaint for generation
                original_complaints = CHIEF_COMPLAINTS.copy()
                CHIEF_COMPLAINTS.clear()
                CHIEF_COMPLAINTS.append(complaint)
                
                new_patient = generate_patient(patient_id, arrival_time, rng, fake)
                
                # Restore original complaints
                CHIEF_COMPLAINTS.clear()
                CHIEF_COMPLAINTS.extend(original_complaints)
                
                patients[-(i+1)] = new_patient
    
    # Sort by arrival time
    patients.sort(key=lambda p: p.arrival_time)
    
    return patients


def generate_observation(patient: Patient) -> PatientObservation:
    """Generate agent-visible observation from internal patient record."""
    return PatientObservation(
        patient_id=patient.patient_id,
        age=patient.age,
        sex=patient.sex,
        chief_complaint=patient.chief_complaint,
        arrival_mode=patient.arrival_mode,
        vitals=patient.vitals,
        waiting_minutes=patient.waiting_minutes,
        active_alerts=[]
    )


def apply_deterioration(patient: Patient, minutes_waited: float) -> Patient:
    """Apply physiological deterioration based on wait time and risk."""
    if minutes_waited <= 20 or patient.deterioration_risk <= 0.1:
        return patient
    
    prob = patient.deterioration_risk * min(minutes_waited / 60.0, 1.0)
    if random.random() < prob:
        p = deepcopy(patient)
        p.vitals.spo2 = max(60, p.vitals.spo2 - random.randint(2, 6))
        p.vitals.heart_rate = min(220, p.vitals.heart_rate + random.randint(5, 15))
        p.vitals.bp_systolic = max(50, p.vitals.bp_systolic - random.randint(4, 12))
        return p
    
    return patient


def get_deterioration_alert(patient: Patient, old_spo2: int) -> str:
    """Generate deterioration alert message based on vital changes."""
    alerts = []
    
    # Check SpO2 changes
    if patient.vitals.spo2 < 85:
        alerts.append(f"CRITICAL SpO2: {patient.vitals.spo2}%")
    elif patient.vitals.spo2 < old_spo2:
        alerts.append(f"DETERIORATING: SpO2 now {patient.vitals.spo2}% (was {old_spo2}%)")
    
    # Check heart rate
    if patient.vitals.heart_rate > 140:
        alerts.append(f"TACHYCARDIA: HR {patient.vitals.heart_rate}bpm")
    
    # Check blood pressure
    if patient.vitals.bp_systolic < 90:
        alerts.append(f"HYPOTENSION: BP {patient.vitals.bp_systolic}mmHg")
    
    return "; ".join(alerts) if alerts else ""
