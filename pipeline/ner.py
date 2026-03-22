"""
pipeline/ner.py
DarkSyntax — Medical Triage NER Module
Person A owns this file.

Model   : en_ner_bc5cdr_md
Versions: spacy==3.7.5, scispacy==0.5.5, numpy==1.26.4
Python  : 3.12

BC5CDR model labels:
  DISEASE  -> symptoms, diagnoses, conditions
  CHEMICAL -> drugs, medications, substances

Anatomy is extracted via rule-based keyword matching
because BC5CDR does not tag body parts.

Install model:
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_ner_bc5cdr_md-0.5.5.tar.gz
"""

import re
import spacy
from dataclasses import dataclass, field
from typing import Optional


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class NERResult:
    symptoms:         list = field(default_factory=list)  # DISEASE entities
    chemicals:        list = field(default_factory=list)  # CHEMICAL entities
    anatomy:          list = field(default_factory=list)  # rule-based body parts
    conditions:       list = field(default_factory=list)  # other entities
    severity_flags:   list = field(default_factory=list)  # matched severity words
    vitals:           dict = field(default_factory=dict)  # extracted vital signs
    negations:        list = field(default_factory=list)  # denied symptoms
    duration_hints:   list = field(default_factory=list)  # time expressions
    query:            str  = ""       # cleaned string for ScaleDown retrieval
    risk_level:       str  = "unknown"  # critical | high | moderate | low
    raw_text:         str  = ""

    def to_dict(self) -> dict:
        return {
            "symptoms":       self.symptoms,
            "chemicals":      self.chemicals,
            "anatomy":        self.anatomy,
            "conditions":     self.conditions,
            "severity_flags": self.severity_flags,
            "vitals":         self.vitals,
            "negations":      self.negations,
            "duration_hints": self.duration_hints,
            "query":          self.query,
            "risk_level":     self.risk_level,
        }


# ─── Anatomy keywords ─────────────────────────────────────────────────────────
# BC5CDR does not tag anatomy — we extract these with keyword matching.

ANATOMY_KEYWORDS = [
    # Limbs
    "left arm", "right arm", "left leg", "right leg",
    "left hand", "right hand", "left foot", "right foot",
    "arm", "leg", "hand", "foot", "finger", "toe",
    "wrist", "elbow", "shoulder", "ankle", "knee", "hip",
    # Trunk
    "chest", "abdomen", "stomach", "back", "lower back",
    "upper back", "lower abdomen", "upper abdomen", "groin",
    # Head / neck
    "head", "neck", "jaw", "face", "scalp", "forehead",
    "eye", "ear", "nose", "throat", "mouth", "tongue",
    # Organs
    "heart", "lung", "liver", "kidney", "bladder",
    "bowel", "colon", "rectum", "uterus", "ovary",
    "prostate", "pancreas", "spleen", "brain",
    # Tissue types
    "muscle", "bone", "joint", "nerve", "vein",
    "artery", "skin", "tissue", "spine",
]


# ─── Severity scoring ─────────────────────────────────────────────────────────

SEVERITY_WORDS = {
    "critical": [
        "severe", "crushing", "excruciating", "unbearable",
        "worst", "sudden", "acute", "extreme", "intense",
        "profuse", "massive", "uncontrolled", "unconscious",
        "not breathing", "no pulse", "seizure", "stroke",
    ],
    "high": [
        "sharp", "strong", "significant", "worsening",
        "persistent", "radiating", "shooting", "stabbing",
        "heavy", "pressure", "tightness", "high fever",
        "difficulty breathing", "can't breathe",
    ],
    "moderate": [
        "moderate", "dull", "mild to moderate", "intermittent",
        "throbbing", "aching", "discomfort", "nausea",
        "dizziness", "light-headed",
    ],
    "low": [
        "mild", "slight", "minor", "little", "occasional",
        "on and off", "manageable",
    ],
}

# These phrases force risk_level = critical immediately
IMMEDIATE_RED_FLAGS = [
    "chest pain", "can't breathe", "difficulty breathing",
    "not breathing", "no pulse", "unconscious", "unresponsive",
    "stroke", "seizure", "overdose", "suicidal",
    "severe bleeding", "anaphylaxis", "allergic reaction",
    "sudden vision loss", "sudden numbness",
    "worst headache", "stiff neck with fever",
]


# ─── Vital sign patterns ──────────────────────────────────────────────────────

VITAL_PATTERNS = [
    # Temperature: "fever 103F" / "temp 38.5C" / "temperature 101"
    (r"(?:temp(?:erature)?|fever)[^\d]*(\d{2,3}(?:\.\d)?)\s*([FC]?)",
     "temperature"),
    # Blood pressure: "BP 140/90" / "blood pressure 120/80"
    (r"(?:bp|blood\s*pressure)[^\d]*(\d{2,3}/\d{2,3})",
     "blood_pressure"),
    # Heart rate: "heart rate 110" / "pulse 95 bpm" / "HR 88"
    (r"(?:heart\s*rate|pulse|hr)[^\d]*(\d{2,3})\s*(?:bpm)?",
     "heart_rate"),
    # O2 sat: "SpO2 94%" / "oxygen 88" / "o2 sat 91"
    (r"(?:o2\s*sat|spo2|oxygen(?:\s*sat(?:uration)?)?)[^\d]*(\d{2,3})\s*%?",
     "oxygen_saturation"),
    # Respiratory rate: "respiratory rate 22" / "RR 18"
    (r"(?:respiratory\s*rate|rr)[^\d]*(\d{1,2})",
     "respiratory_rate"),
]

NEGATION_CUES = [
    "no ", "not ", "denies ", "without ", "absence of ",
    "negative for ", "never had ", "hasn't had ",
]

DURATION_PATTERNS = [
    r"\d+\s*(?:hour|hr|minute|min|day|week|month)s?\b",
    r"since\s+(?:yesterday|this morning|last night|this week)",
    r"for\s+the\s+(?:past|last)\s+\d+",
    r"started\s+\d+\s*(?:hour|day|week)s?\s+ago",
    r"onset\s+\d+",
]


# ─── Main NER class ───────────────────────────────────────────────────────────

class MedicalNER:
    """
    Wraps en_ner_bc5cdr_md and adds:
    - proper DISEASE / CHEMICAL label handling
    - rule-based anatomy extraction
    - severity scoring with risk level
    - vital sign regex extraction
    - negation detection via look-behind
    - duration hint extraction
    - ScaleDown query string builder
    """

    def __init__(self, model: str = "en_ner_bc5cdr_md"):
        print(f"[NER] Loading model: {model} ...")
        self.nlp = spacy.load(model)
        print("[NER] Ready.")

    def extract(self, text: str) -> NERResult:
        result     = NERResult(raw_text=text)
        text_lower = text.lower()
        doc        = self.nlp(text)

        # ── 1. BC5CDR entity pass ─────────────────────────────────────────────
        for ent in doc.ents:
            span = ent.text.strip()
            if not span:
                continue

            # Negation: check 40 chars before this entity
            context = text_lower[max(0, ent.start_char - 40): ent.start_char]
            negated = any(cue in context for cue in NEGATION_CUES)

            if negated:
                result.negations.append(span)
                continue

            if ent.label_ == "DISEASE":
                result.symptoms.append(span)
            elif ent.label_ == "CHEMICAL":
                result.chemicals.append(span)
            else:
                # Catch-all for any other labels
                result.conditions.append(span)

        # ── 2. Rule-based anatomy extraction ─────────────────────────────────
        # Sort longer phrases first so "left arm" matches before "arm"
        for kw in sorted(ANATOMY_KEYWORDS, key=len, reverse=True):
            if kw in text_lower:
                result.anatomy.append(kw)

        # ── 3. Severity scoring ───────────────────────────────────────────────
        result.severity_flags, result.risk_level = self._score_severity(
            text_lower
        )

        # ── 4. Vital signs ────────────────────────────────────────────────────
        result.vitals = self._extract_vitals(text)

        # ── 5. Duration hints ─────────────────────────────────────────────────
        result.duration_hints = self._extract_durations(text_lower)

        # ── 6. Dedup all lists ────────────────────────────────────────────────
        result.symptoms       = _dedup(result.symptoms)
        result.chemicals      = _dedup(result.chemicals)
        result.anatomy        = _dedup(result.anatomy)
        result.conditions     = _dedup(result.conditions)
        result.severity_flags = _dedup(result.severity_flags)
        result.negations      = _dedup(result.negations)

        # ── 7. Build ScaleDown query string ───────────────────────────────────
        # DISEASE entities + anatomy give strongest retrieval signal
        query_parts  = (
            result.symptoms       # DISEASE — highest priority
            + result.anatomy      # body parts
            + result.severity_flags  # severity context
            + result.chemicals    # drugs / substances
        )
        result.query = " ".join(query_parts[:20])

        # Fallback if NER extracted nothing
        if not result.query.strip():
            result.query = text[:200]

        return result

    # ── private helpers ───────────────────────────────────────────────────────

    def _score_severity(self, text_lower: str):
        """
        Returns (matched_flags, risk_level).
        Level priority: critical > high > moderate > low > unknown
        """
        matched_flags = []
        level         = "low"
        level_order   = ["critical", "high", "moderate", "low"]

        # Immediate red flags → instant critical
        for flag in IMMEDIATE_RED_FLAGS:
            if flag in text_lower:
                matched_flags.append(flag)
                level = "critical"

        if level == "critical":
            return _dedup(matched_flags), level

        # Tiered scoring — only upgrade, never downgrade
        for tier in level_order:
            for word in SEVERITY_WORDS[tier]:
                if word in text_lower:
                    matched_flags.append(word)
                    if level_order.index(tier) < level_order.index(level):
                        level = tier

        return _dedup(matched_flags), level if matched_flags else "unknown"

    def _extract_vitals(self, text: str) -> dict:
        vitals = {}
        for pattern, key in VITAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vitals[key] = match.group(1)
        return vitals

    def _extract_durations(self, text_lower: str) -> list:
        durations = []
        for pattern in DURATION_PATTERNS:
            durations.extend(re.findall(pattern, text_lower))
        return durations


# ─── Singleton + public API ───────────────────────────────────────────────────

_ner_instance: Optional[MedicalNER] = None


def get_ner() -> MedicalNER:
    """Model loads once on first call, reused for all subsequent requests."""
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = MedicalNER()
    return _ner_instance


def extract_entities(text: str) -> dict:
    """
    Main entry point called by triage.py and compressor.py.

    Returns:
    {
        "symptoms":       [...],   DISEASE entities from BC5CDR
        "chemicals":      [...],   CHEMICAL entities (drugs)
        "anatomy":        [...],   body parts via keyword matching
        "conditions":     [...],   other NER entities
        "severity_flags": [...],   matched severity words
        "vitals":         {...},   temp / bp / hr / spo2 / rr
        "negations":      [...],   denied symptoms
        "duration_hints": [...],   time expressions
        "query":          "...",   ready for ScaleDown retrieval
        "risk_level":     "..."    critical|high|moderate|low|unknown
    }
    """
    return get_ner().extract(text).to_dict()


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TEST_CASES = [
        (
            "CRITICAL — MI presentation",
            "Patient presents with severe crushing chest pain radiating to "
            "the left arm and jaw, profuse sweating, and difficulty breathing "
            "for the past 30 minutes. BP 90/60, heart rate 110 bpm.",
        ),
        (
            "HIGH — meningitis signs",
            "Sudden onset worst headache of my life, stiff neck, "
            "fever 102F, sensitivity to light since this morning.",
        ),
        (
            "MODERATE — tension headache",
            "Throbbing headache for 2 days, mild nausea, "
            "no fever, no vomiting. Pain is 5/10.",
        ),
        (
            "LOW + NEGATION — sore throat",
            "Patient denies chest pain. Slight sore throat for 3 days. "
            "No fever, no difficulty breathing.",
        ),
        (
            "VITALS + CHEMICAL — appendicitis + meds",
            "BP 140/90, temp 38.9C, SpO2 94%, respiratory rate 22. "
            "Patient on metformin 500mg and aspirin. "
            "Sharp stabbing pain in lower right abdomen for 6 hours.",
        ),
    ]

    print("=" * 65)
    print("DarkSyntax — NER Self-Test")
    print("Model : en_ner_bc5cdr_md")
    print("spaCy : 3.7.5  |  scispaCy : 0.5.5  |  Python : 3.12")
    print("=" * 65)

    all_ok = True
    for label, text in TEST_CASES:
        print(f"\n[{label}]")
        print(f"  Input    : {text[:70]}...")

        result = extract_entities(text)

        print(f"  Risk     : {result['risk_level'].upper()}")
        print(f"  Symptoms : {result['symptoms']}")
        print(f"  Chemicals: {result['chemicals']}")
        print(f"  Anatomy  : {result['anatomy']}")
        print(f"  Severity : {result['severity_flags']}")
        print(f"  Vitals   : {result['vitals']}")
        print(f"  Negated  : {result['negations']}")
        print(f"  Duration : {result['duration_hints']}")
        print(f"  Query    : {result['query'][:70]}")

        # Basic assertions
        if "CRITICAL" in label and result["risk_level"] != "critical":
            print(f"  [WARN] Expected critical, got {result['risk_level']}")
            all_ok = False

    print("\n" + "=" * 65)
    if all_ok:
        print("All cases passed. NER module ready.")
        print("Next step: python pipeline/extractor.py")
    else:
        print("Some cases did not match expected risk level — check above.")
    print("=" * 65)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _dedup(lst: list) -> list:
    """Preserve insertion order, remove case-insensitive duplicates."""
    seen = set()
    out  = []
    for item in lst:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
