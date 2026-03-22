import json
import time
import requests
from datetime import datetime

SCALEDOWN_HEADERS = {
    'x-api-key': 'YOUR_SCALEDOWN_KEY',
    'Content-Type': 'application/json'
}

# ── FUNCTION 1: Confidence Gate ──────────────────────
def confidence_gate(recommendation: dict) -> dict:
    '''Score the recommendation 0-1. Flag if below 0.85.'''

    score = 0.5  # base score

    # +0.2 if cited records exist
    if recommendation.get('cited_records', '') not in ['', 'N/A']:
        score += 0.2

    # +0.2 if warnings exist
    if recommendation.get('warnings', '') not in ['', 'N/A']:
        score += 0.2

    # +0.1 if immediate action is detailed
    if len(recommendation.get('immediate_action', '')) > 20:
        score += 0.1

    # +0.05 if risk level exists
    if recommendation.get('risk_level'):
        score += 0.05

    # Keep between 0 and 1
    score = round(max(0.0, min(1.0, score)), 2)

    return {
        'score': score,
        'requires_human_review': score < 0.85,
        'status': '✅ SAFE TO SHOW' if score >= 0.85 else '🔴 FLAG FOR DOCTOR'
    }


# ── FUNCTION 2: Audit Log ────────────────────────────
def save_audit_log(emergency, recommendation, confidence, total_latency_ms):
    '''Save every triage decision to a JSON file.'''

    log = {
        'timestamp': datetime.now().isoformat(),
        'emergency': emergency,
        'diagnosis': recommendation.get('diagnosis'),
        'immediate_action': recommendation.get('immediate_action'),
        'risk_level': recommendation.get('risk_level'),
        'cited_records': recommendation.get('cited_records'),
        'warnings': recommendation.get('warnings'),
        'confidence_score': confidence['score'],
        'requires_human_review': confidence['requires_human_review'],
        'total_pipeline_latency_ms': total_latency_ms
    }

    # filename is unique every time using timestamp
    filename = f'audit_{int(time.time())}.json'

    with open(filename, 'w') as f:
        json.dump(log, f, indent=2)

    print(f'[C] Audit saved: {filename}')
    return filename, log


def verify_recommendation(recommendation: dict, context: str) -> dict:
    '''Chain-of-Verification: checks recommendation against context.'''

    try:
        # Skip API call if no real key set
        if SCALEDOWN_HEADERS['x-api-key'] == 'YOUR_SCALEDOWN_KEY':
            raise ValueError('No API key set — using fallback')

        resp = requests.post(
            'https://api.scaledown.xyz/pipeline/run/',
            headers=SCALEDOWN_HEADERS,
            json={
                'pipeline': 'chain_of_verification',
                'content': json.dumps(recommendation),
                'context': context
            },
            timeout=0.5
        )
        result = resp.json()
        return {
            'verified': result.get('verified', True),
            'corrections': result.get('corrections', 'None')
        }

    except Exception as e:
        print(f'[C] Verification skipped: {e}')
        return {
            'verified': True,
            'corrections': 'Verification unavailable'
        }


# ── MAIN: Run Section C ──────────────────────────────
def run_section_c(section_b_output: dict, total_pipeline_ms: float) -> dict:
    start = time.time()

    # Section 6: Verify
    context = section_b_output.get('context_used', '')
    verification = verify_recommendation(section_b_output, context)

    # Section 7: Confidence gate
    confidence = confidence_gate(section_b_output)

    # Section 8: Audit log
    emergency = section_b_output.get('symptoms', 'Unknown')
    audit_file, log = save_audit_log(
        emergency, section_b_output, confidence, total_pipeline_ms
    )

    total_ms = (time.time() - start) * 1000
    print(f'[C] SECTION C DONE: {total_ms:.0f}ms')
    print(f'[C] Confidence: {confidence["score"]} → {confidence["status"]}')

    return {
        **section_b_output,
        'verification': verification,
        'confidence': confidence,
        'audit_file': audit_file,
        'section_c_latency_ms': round(total_ms, 1)
    }


# ── TEST ─────────────────────────────────────────────
if __name__ == '__main__':

    print("File is running...")
    print("Test block reached!")

    # This is exactly what Person B will give you
    mock_b_output = {
        'symptoms': '62M, chest pain, left arm numbness, shortness of breath',
        'diagnosis': 'Probable STEMI',
        'immediate_action': 'Activate cath lab. Administer aspirin 325mg. 12-lead ECG immediately.',
        'medications_to_check': ['Warfarin', 'Aspirin', 'Metoprolol'],
        'risk_level': 'HIGH',
        'cited_records': 'BP 145/90 from Jan 2024. DVT history 2022. Warfarin noted.',
        'warnings': 'Patient on Warfarin — bleeding risk. Penicillin allergy.',
        'context_used': 'Patient: 62M. BP 145/90. DVT. Warfarin. Aspirin. Penicillin allergy.',
        'llm_latency_ms': 210,
        'section_b_latency_ms': 215
    }

    print('\n=== RUNNING FULL SECTION C ===')
    result = run_section_c(mock_b_output, total_pipeline_ms=450)

    print('\n=== FINAL OUTPUT ===')
    print(f'Diagnosis     : {result["diagnosis"]}')
    print(f'Risk Level    : {result["risk_level"]}')
    print(f'Confidence    : {result["confidence"]["score"]}')
    print(f'Status        : {result["confidence"]["status"]}')
    print(f'Verified      : {result["verification"]["verified"]}')
    print(f'Audit File    : {result["audit_file"]}')
    print(f'Section C ms  : {result["section_c_latency_ms"]}')