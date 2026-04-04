"""ERTriageEnv typed models — OpenEnv spec interface."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    """Patient vital signs measurements used for triage assessment."""
    heart_rate: int = Field(ge=20, le=250, description="Heart rate in BPM")
    bp_systolic: int = Field(ge=50, le=250, description="Systolic BP mmHg")
    bp_diastolic: int = Field(ge=30, le=150, description="Diastolic BP mmHg")
    spo2: int = Field(ge=60, le=100, description="O2 saturation percent")
    respiratory_rate: int = Field(ge=4, le=60, description="Breaths per minute")
    temperature: float = Field(ge=32.0, le=43.0, description="Body temp Celsius")
    pain_score: int = Field(ge=0, le=10, description="Pain scale 0-10")
    gcs: int = Field(ge=3, le=15, description="Glasgow Coma Scale")


class Patient(BaseModel):
    """Complete internal patient record - NOT sent to agent (contains hidden fields)."""
    patient_id: str
    name: str
    age: int = Field(ge=0, le=120)
    sex: str
    chief_complaint: str
    complaint_category: str
    arrival_time: float = Field(description="Sim minutes since shift start")
    arrival_mode: str = Field(description="ambulance or walk_in or transfer")
    vitals: VitalSigns
    esi_ground_truth: int = Field(ge=1, le=5, description="True ESI — HIDDEN from agent")
    waiting_minutes: float = Field(default=0.0)
    status: str = Field(default="waiting")
    deterioration_risk: float = Field(default=0.0, ge=0.0, le=1.0)


class PatientObservation(BaseModel):
    """What the agent sees about a patient (no hidden fields)."""
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    arrival_mode: str
    vitals: VitalSigns
    waiting_minutes: float
    active_alerts: list[str] = Field(default_factory=list,
        description="Deterioration or escalation alerts visible to agent")


class BedStatus(BaseModel):
    """Current status of an ER bed."""
    bed_id: str
    bed_type: str = Field(description="resus or acute or minor")
    occupied: bool
    patient_id: Optional[str] = None


class ERState(BaseModel):
    """Complete state of the ER environment for agent observation."""
    task_id: str
    episode_id: str
    shift_time: float = Field(description="Simulation minutes elapsed")
    patients_waiting: list[PatientObservation]
    beds: list[BedStatus]
    patients_triaged: int
    total_patients: int
    cumulative_reward: float
    active_alerts: list[str] = Field(default_factory=list)
    episode_done: bool = False


class TriageAction(BaseModel):
    """Action submitted by agent to triage a patient."""
    patient_id: str = Field(description="ID of patient being triaged")
    priority: int = Field(ge=1, le=4,
        description="ESI priority: 1=immediate, 2=emergent, 3=urgent, 4=less-urgent")
    bed_type: str = Field(
        description="Bed assignment: resus or acute or minor or waiting_area")
    escalate: bool = Field(default=False,
        description="True = request immediate physician attention")
    reasoning: Optional[str] = Field(default=None,
        description="Agent explanation — logged but NOT used by grader")


class TriageReward(BaseModel):
    """Reward breakdown for triage decision."""
    total: float = Field(description="Sum of all signals. Range: -1.0 to 1.0")
    priority_accuracy: float = Field(ge=0.0, le=0.5,
        description="Signal for correct ESI level")
    bed_assignment: float = Field(ge=0.0, le=0.3,
        description="Signal for correct bed type")
    escalation_timing: float = Field(ge=0.0, le=0.2,
        description="Signal for escalation decision")
    penalty: float = Field(le=0.0,
        description="Negative value for dangerous errors")
    explanation: str = Field(description="Human-readable reward breakdown")


class StepResult(BaseModel):
    """Result of an agent step in the environment."""
    observation: ERState
    reward: TriageReward
    done: bool
    info: dict


class ResetRequest(BaseModel):
    """Request to reset the environment to a new episode."""
    task_id: str = Field(default="task_easy")
    seed: Optional[int] = Field(default=42)


class TaskConfig(BaseModel):
    """Configuration for a specific triage task."""
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    passing_score: float
    n_patients: int
