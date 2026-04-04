"""ERTriageEnv environment simulation for emergency room triage.

Implements OpenEnv-compatible environment with:
- Deterministic patient generation and ESI calculation
- Real-time deterioration simulation
- Bed assignment and resource management
- Structured reward calculation
- Episode management and state tracking
"""
from app.models import (Patient, PatientObservation, ERState, BedStatus,
    TriageAction, StepResult, TaskConfig, TriageReward)
from app.patient_gen import generate_patient_batch, generate_observation, apply_deterioration, get_deterioration_alert
from app.grader import calculate_esi, grade_task_easy, grade_task_medium, grade_task_hard
from app.reward import calculate_reward, calculate_episode_summary
import uuid
import random
import numpy as np
from copy import deepcopy


TASK_CONFIGS = {
    "task_easy": TaskConfig(task_id="task_easy", name="Single Patient Triage",
        description="Triage one patient correctly",
        difficulty="easy", max_steps=1, passing_score=0.70, n_patients=1),
    "task_medium": TaskConfig(task_id="task_medium", name="ER Shift Management",
        description="15-patient shift, 2 simulated hours",
        difficulty="medium", max_steps=30, passing_score=0.60, n_patients=15),
    "task_hard": TaskConfig(task_id="task_hard", name="Mass Casualty Event",
        description="40-patient MCI, limited resus beds",
        difficulty="hard", max_steps=80, passing_score=0.50, n_patients=40),
}

BED_INVENTORY = {
    "task_easy":   {"resus": 2, "acute": 4, "minor": 6},
    "task_medium": {"resus": 3, "acute": 6, "minor": 10},
    "task_hard":   {"resus": 5, "acute": 10, "minor": 20},
}


class EREnvironment:
    """Emergency Room triage environment implementing OpenEnv interface.
    
    Manages patient flow, bed assignments, deterioration simulation,
    and reward calculation for triage training scenarios.
    """
    
    def __init__(self):
        self.patients: dict[str, Patient] = {}
        self.beds: list[BedStatus] = []
        self.triaged_ids: set[str] = set()
        self.triaged_patients: list[Patient] = []
        self.triaged_actions: list[TriageAction] = []
        self.step_count: int = 0
        self.shift_time: float = 0.0
        self.task_config: TaskConfig = None
        self.episode_id: str = ""
        self.rewards_history: list[TriageReward] = []
        self.cumulative_reward: float = 0.0
        self.seed: int = 42
        self.resus_used: int = 0

    def reset(self, task_id: str = "task_easy", seed: int = 42) -> ERState:
        """Reset environment to start new episode.
        
        Args:
            task_id: Task configuration identifier
            seed: Random seed for reproducible episodes
            
        Returns:
            Initial ERState observation
            
        Raises:
            ValueError: If task_id is not recognized
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}")
        self.seed = seed
        self.task_config = TASK_CONFIGS[task_id]
        self.episode_id = str(uuid.uuid4())[:8]
        self.patients = {}
        self.triaged_ids = set()
        self.triaged_patients = []
        self.triaged_actions = []
        self.step_count = 0
        self.shift_time = 0.0
        self.rewards_history = []
        self.cumulative_reward = 0.0
        self.resus_used = 0
        self.beds = []
        
        # Initialize bed inventory
        for bed_type, count in BED_INVENTORY[task_id].items():
            for i in range(1, count + 1):
                self.beds.append(BedStatus(
                    bed_id=f"{bed_type}_{i}", bed_type=bed_type,
                    occupied=False, patient_id=None))
        
        # Generate patients and calculate ground truth ESI
        batch = generate_patient_batch(self.task_config.n_patients, seed=seed)
        for patient in batch:
            try:
                patient.esi_ground_truth = calculate_esi(patient)
            except Exception as e:
                print(f"ESI calc error for {patient.patient_id}: {e}")
                patient.esi_ground_truth = 3  # safe default
            self.patients[patient.patient_id] = patient
        
        return self._build_state()

    def step(self, action: TriageAction) -> StepResult:
        """Execute one triage action and return result.
        
        Args:
            action: TriageAction to execute
            
        Returns:
            StepResult with observation, reward, done flag, and info
            
        Raises:
            ValueError: If patient_id is not found
        """
        if action.patient_id not in self.patients:
            raise ValueError(f"Unknown patient_id '{action.patient_id}'. "
                f"Available: {list(self.patients.keys())}")
        
        patient = self.patients[action.patient_id]
        patient.waiting_minutes = max(0.0, self.shift_time - patient.arrival_time)
        
        # Calculate reward
        reward = calculate_reward(patient, action, self.triaged_ids)
        self.rewards_history.append(reward)
        self.cumulative_reward = round(self.cumulative_reward + reward.total, 4)
        
        # Process triage action if not already done
        if action.patient_id not in self.triaged_ids:
            self.triaged_ids.add(action.patient_id)
            patient.status = "triaged"
            self.triaged_patients.append(patient)
            self.triaged_actions.append(action)
            
            # Assign bed
            bed = self._assign_bed(action.bed_type)
            if bed:
                bed.occupied = True
                bed.patient_id = action.patient_id
                if action.bed_type == "resus":
                    self.resus_used += 1
        
        # Advance time and update state
        self.step_count += 1
        self.shift_time += 5.0
        self._deterioration_tick()
        
        # Check episode completion
        done = (len(self.triaged_ids) >= len(self.patients) or
                self.step_count >= self.task_config.max_steps)
        
        info = {
            "step": self.step_count,
            "triaged": len(self.triaged_ids),
            "total": len(self.patients),
            "shift_time": self.shift_time,
            "episode_id": self.episode_id,
            "task_id": self.task_config.task_id,
            "reward_breakdown": reward.explanation,
        }
        
        if done:
            info["episode_summary"] = calculate_episode_summary(self.rewards_history)
        
        return StepResult(observation=self._build_state(), reward=reward, done=done, info=info)

    def state(self) -> ERState:
        """Get current environment state without advancing time."""
        return self._build_state()

    def _assign_bed(self, bed_type: str):
        """Find and assign available bed of specified type."""
        for bed in self.beds:
            if bed.bed_type == bed_type and not bed.occupied:
                return bed
        return None

    def _deterioration_tick(self):
        """Apply physiological deterioration to waiting patients."""
        for pid, patient in self.patients.items():
            if patient.status == "waiting" and patient.deterioration_risk > 0.1:
                patient.waiting_minutes = max(0.0, self.shift_time - patient.arrival_time)
                old_spo2 = patient.vitals.spo2
                updated = apply_deterioration(patient, patient.waiting_minutes)
                new_esi = calculate_esi(updated)
                
                # Update ESI if deteriorated
                if new_esi < updated.esi_ground_truth:
                    updated.esi_ground_truth = new_esi
                
                # Generate alerts
                alert = get_deterioration_alert(updated, old_spo2)
                if alert:
                    if not hasattr(updated, '_alerts'):
                        updated._alerts = []
                    updated._alerts = [alert]
                
                self.patients[pid] = updated

    def _build_state(self) -> ERState:
        """Build current ERState observation."""
        waiting = []
        for p in self.patients.values():
            if p.status == "waiting":
                obs = generate_observation(p)
                if hasattr(p, '_alerts') and p._alerts:
                    obs.active_alerts = p._alerts
                waiting.append(obs)
        
        # Sort by waiting time (longest first)
        waiting.sort(key=lambda x: x.waiting_minutes, reverse=True)
        
        return ERState(
            task_id=self.task_config.task_id if self.task_config else "none",
            episode_id=self.episode_id,
            shift_time=self.shift_time,
            patients_waiting=waiting,
            beds=self.beds,
            patients_triaged=len(self.triaged_ids),
            total_patients=len(self.patients),
            cumulative_reward=self.cumulative_reward,
            episode_done=(self.step_count >= self.task_config.max_steps
                if self.task_config else False)
        )

    def get_task_configs(self) -> dict:
        """Get available task configurations."""
        return TASK_CONFIGS


# Global environment instance
env = EREnvironment()
