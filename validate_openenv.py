#!/usr/bin/env python3
"""Run OpenEnv validation for ERTriageEnv"""
import sys
import requests
import json

def validate_openenv(base_url="http://localhost:7860"):
    """Validate OpenEnv endpoints"""
    print("🔍 Running OpenEnv Validation...")
    print(f"📍 Target URL: {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health endpoint: {health_data}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test reset endpoint
    try:
        response = requests.get(f"{base_url}/reset?task_id=task_easy&seed=42")
        if response.status_code == 200:
            reset_data = response.json()
            print(f"✅ Reset endpoint: Episode {reset_data.get('episode_id', 'unknown')}")
        else:
            print(f"❌ Reset endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Reset endpoint error: {e}")
        return False
    
    # Test state endpoint
    try:
        response = requests.get(f"{base_url}/state")
        if response.status_code == 200:
            state_data = response.json()
            patients_waiting = len(state_data.get('patients_waiting', []))
            print(f"✅ State endpoint: {patients_waiting} patients waiting")
        else:
            print(f"❌ State endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ State endpoint error: {e}")
        return False
    
    # Test step endpoint
    try:
        # First get a patient from state
        state_response = requests.get(f"{base_url}/state")
        if state_response.status_code == 200:
            state_data = state_response.json()
            patients = state_data.get('patients_waiting', [])
            if patients:
                patient_id = patients[0].get('patient_id')
                action = {
                    "patient_id": patient_id,
                    "priority": 2,
                    "bed_type": "acute",
                    "escalate": False,
                    "reasoning": "Test validation"
                }
                
                step_response = requests.post(
                    f"{base_url}/step",
                    json=action,
                    headers={"Content-Type": "application/json"}
                )
                if step_response.status_code == 200:
                    step_data = step_response.json()
                    print(f"✅ Step endpoint: Action processed, reward={step_data.get('reward', {}).get('total', 0)}")
                else:
                    print(f"❌ Step endpoint failed: {step_response.status_code}")
                    return False
            else:
                print("⚠️  No patients available for step test")
        else:
            print(f"❌ Could not get state for step test")
            return False
    except Exception as e:
        print(f"❌ Step endpoint error: {e}")
        return False
    
    print("\n🎉 OpenEnv Validation PASSED!")
    print("✅ All endpoints working correctly")
    print("✅ Environment is ready for AI agent training")
    return True

if __name__ == "__main__":
    success = validate_openenv()
    sys.exit(0 if success else 1)
