"""
Phase 1 – Part 2
Generate synthetic hospital policy documents, chunk them,
embed them (via sentence-transformers or OpenAI), and store in FAISS.
"""

import os
import json
import pickle
import hashlib
import numpy as np

POLICIES_DIR = os.path.join(os.path.dirname(__file__), "policies")
INDEX_PATH   = os.path.join(os.path.dirname(__file__), "faiss_index")

# ── 1. Synthetic Policy Documents ────────────────────────────────────────────
POLICY_DOCUMENTS = {
    "admission_policy.txt": """
HOSPITAL ADMISSION POLICY — Version 3.2 | Effective: January 2024

1. GENERAL ADMISSION PROCEDURES
All patients must present a valid government-issued photo ID and insurance card at
the time of admission. Patients without insurance must complete a financial assessment
before elective procedures. Emergency admissions are processed immediately regardless
of insurance status, and financial documentation may be completed post-stabilisation.

2. ADMISSION TYPES
  2.1 Emergency Admission: Patients arriving via emergency services or walk-in with
      life-threatening conditions are admitted immediately. No pre-authorisation
      required. The attending physician determines admission necessity.
  2.2 Elective Admission: Scheduled procedures require pre-authorisation from the
      insurance provider at least 72 hours in advance. Patients must complete
      pre-admission testing (PAT) within 30 days of the scheduled procedure.
  2.3 Urgent Admission: Conditions requiring treatment within 24–48 hours. Verbal
      pre-authorisation from insurer must be obtained within 8 hours of admission.

3. REQUIRED DOCUMENTATION
  - Government-issued photo ID
  - Insurance card and policy number
  - Referral letter (for specialist admissions)
  - Previous medical records (if available)
  - Advance directive or healthcare proxy (if applicable)

4. PAEDIATRIC ADMISSIONS
  Children under 18 must be accompanied by a parent or legal guardian who must
  provide written consent for treatment. In cases of emergency, treatment proceeds
  without consent if the guardian is unreachable.

5. PATIENT RIGHTS ON ADMISSION
  Every patient has the right to be informed of their diagnosis and proposed
  treatment plan, to accept or refuse treatment, to privacy and confidentiality,
  and to access their medical records within 30 days of request.
""",

    "billing_policy.txt": """
HOSPITAL BILLING AND FINANCIAL POLICY — Version 2.8 | Effective: March 2024

1. BILLING OVERVIEW
The hospital provides an itemised bill for all services rendered. Bills are generated
within 7 business days of discharge. Patients may request an itemised statement at
any time during or after their stay.

2. INSURANCE BILLING
  2.1 In-Network Claims: Submitted automatically within 5 business days of discharge.
      Patients are responsible only for co-pay, co-insurance, and deductible amounts.
  2.2 Out-of-Network Claims: Patient is responsible for the full amount upfront;
      the hospital will provide a superbill for independent reimbursement claims.
  2.3 Medicare and Medicaid: Billed according to CMS guidelines. Any balance billing
      is prohibited for Medicare Advantage and Medicaid patients.

3. PAYMENT OPTIONS
  - Full payment at discharge (5% prompt-pay discount applies)
  - Payment plans: 0% interest for balances under $5,000 paid within 12 months
  - Financial hardship: Charity care available for patients below 250% federal
    poverty line. Applications must be submitted within 90 days of discharge.

4. PRIOR AUTHORISATION REQUIREMENTS
  Prior insurance authorisation is REQUIRED for:
  - All elective surgical procedures
  - MRI, CT scans, and PET scans (non-emergency)
  - Extended inpatient stays beyond 5 days
  - Specialist referrals outside the primary network
  - Durable medical equipment (DME) over $500
  Prior auth is NOT required for emergency services, lab tests, or routine X-rays.

5. DISPUTE RESOLUTION
  Billing disputes must be submitted in writing within 60 days of receiving the bill.
  The hospital will respond within 30 days. Unresolved disputes may be escalated to
  the Patient Financial Advocate.
""",

    "discharge_policy.txt": """
PATIENT DISCHARGE POLICY — Version 4.1 | Effective: January 2024

1. DISCHARGE CRITERIA
  A patient is eligible for discharge when:
  - The attending physician certifies medical stability
  - Vital signs are within acceptable parameters for 24 hours
  - The patient or caregiver demonstrates understanding of home care instructions
  - Appropriate follow-up appointments are scheduled
  - Prescriptions and discharge medications are prepared

2. DISCHARGE PLANNING PROCESS
  Discharge planning begins at admission. The multidisciplinary team (physician,
  nurse, case manager, social worker) collaborates to develop a discharge plan.
  Plans are reviewed daily and updated based on clinical progress.

3. DISCHARGE AGAINST MEDICAL ADVICE (AMA)
  Patients have the right to leave at any time. If leaving AMA:
  - Patient must sign the AMA form acknowledging associated risks
  - Physician must document the discussion and patient's decision
  - Insurance may deny coverage for complications arising from AMA discharge
  - Hospital is not liable for adverse outcomes following AMA discharge

4. DISCHARGE INSTRUCTIONS
  All patients receive written discharge instructions covering:
  - Medication schedule and dosage
  - Activity restrictions and diet
  - Wound care instructions (if applicable)
  - Warning signs requiring immediate return to ER
  - Follow-up appointment schedule

5. APPEAL PROCESS
  If a patient disagrees with the discharge decision, they may:
  - Request a physician review within 24 hours
  - Contact the Patient Advocate for assistance
  - File an appeal with their insurance company
  - Request a Peer Review from a second attending physician
""",

    "insurance_policy.txt": """
INSURANCE AND PRE-AUTHORISATION POLICY — Version 3.5 | Effective: February 2024

1. ACCEPTED INSURANCE PROVIDERS
  The hospital is in-network with: BlueCross BlueShield, Aetna, UnitedHealth Group,
  Cigna, Humana, Medicare (Parts A & B), and Medicaid. Patients with other insurers
  are treated as out-of-network; pre-payment or financial guarantee may be required.

2. PRE-AUTHORISATION PROCESS
  Step 1: Physician submits clinical justification to the insurance provider.
  Step 2: Insurance reviews and responds within 3 business days (urgent: 24 hours).
  Step 3: Authorisation number is recorded in the patient's file.
  Step 4: Procedure is scheduled only after written authorisation is received.

3. SURGERY PRE-AUTHORISATION
  Prior authorisation is MANDATORY for all surgical procedures, including:
  - Cardiac surgery and catheterisation
  - Orthopaedic surgery (joint replacement, spinal)
  - General surgery (appendectomy, cholecystectomy, hernia repair)
  - Bariatric surgery (requires additional documentation of BMI and prior treatment)
  - Neurosurgery (pre-auth + second opinion required)
  Emergency surgery is exempt; retrospective authorisation is obtained within 24 hours.

4. INSURANCE VERIFICATION
  Insurance eligibility is verified electronically at admission and again 24 hours
  before elective procedures. Patients must notify the hospital of any insurance
  changes at least 48 hours before scheduled procedures.

5. DENIAL MANAGEMENT
  If insurance denies a claim:
  - Hospital files a first-level appeal within 15 days
  - Clinical staff provide additional supporting documentation
  - If denied again, an external independent review is requested
  - Patients are notified of all decisions within 5 business days

6. CO-PAY AND DEDUCTIBLE COLLECTION
  Co-pays are collected at registration. Annual deductible balances are estimated
  and collected upfront for elective procedures. Estimates may vary from final bill
  based on actual services rendered.
""",

    "emergency_policy.txt": """
EMERGENCY SERVICES POLICY — Version 5.0 | Effective: January 2024

1. EMERGENCY TRIAGE PROTOCOL
  All patients presenting to the Emergency Department (ED) are triaged using the
  Emergency Severity Index (ESI) 5-level triage system:
  - ESI 1 (Immediate): Life-threatening; immediate physician assessment
  - ESI 2 (Emergent): High risk; assessed within 15 minutes
  - ESI 3 (Urgent): Stable but requires multiple resources; wait up to 30 minutes
  - ESI 4 (Less Urgent): Single resource needed; assessed within 60 minutes
  - ESI 5 (Non-Urgent): Minimal resources; may be redirected to urgent care

2. NO-REFUSAL POLICY
  Under EMTALA (Emergency Medical Treatment and Labor Act), the hospital must
  provide a medical screening exam to any individual requesting emergency care,
  regardless of insurance status, ability to pay, national origin, or citizenship.

3. EMERGENCY ADMISSION PROCESS
  - Registration and triage occur simultaneously for ESI 1 and 2 patients
  - Consent is implied for unconscious or incapacitated patients
  - Family notification is attempted within 2 hours of admission
  - No prior authorisation is required for emergency treatment
  - Retrospective authorisation from insurance is obtained within 24 hours

4. CRITICAL CARE PROTOCOLS
  - Code Blue (Cardiac Arrest): Rapid Response Team dispatched immediately
  - Stroke Protocol: CT scan within 25 minutes; thrombolytics within 60 minutes
  - Trauma Protocol: Trauma team activation for major injuries
  - Sepsis Bundle: Initiated for patients meeting sepsis criteria within 1 hour

5. TRANSFER POLICY
  Patients requiring specialised care not available at this facility are transferred
  after stabilisation. Receiving facility must confirm acceptance before transfer.
  All transfers are documented with a Transfer Summary and signed consent.
""",

    "privacy_policy.txt": """
PATIENT PRIVACY AND DATA PROTECTION POLICY — Version 2.3 | Effective: January 2024

1. HIPAA COMPLIANCE
  This hospital complies fully with the Health Insurance Portability and
  Accountability Act (HIPAA). All Protected Health Information (PHI) is safeguarded
  through administrative, physical, and technical controls.

2. PERMITTED DISCLOSURES
  Patient information may be shared without explicit consent for:
  - Treatment by authorised healthcare providers
  - Payment processing with insurance providers
  - Healthcare operations (quality reviews, audits)
  - Public health reporting as required by law
  - Compliance with court orders or law enforcement requests

3. PATIENT RIGHTS UNDER HIPAA
  - Right to access and receive copies of medical records
  - Right to request amendments to records
  - Right to receive an accounting of disclosures
  - Right to restrict certain disclosures
  - Right to receive confidential communications

4. DATA BREACH PROTOCOL
  In the event of a data breach involving PHI:
  - Affected patients are notified within 60 days of discovery
  - HHS Office for Civil Rights is notified within 60 days
  - If more than 500 individuals affected, media notification is required
  - Incident is logged and root cause analysis conducted within 30 days
"""
}

# ── 2. Chunking ───────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

# ── 3. Simple TF-IDF-style embeddings (no heavy deps) ────────────────────────
# Replaced by sentence-transformers / OpenAI in production (see comments below)

class SimpleVectorStore:
    """
    Lightweight vector store using TF-IDF vectors.
    In production replace embed() with:
      from sentence_transformers import SentenceTransformer
      model = SentenceTransformer('all-MiniLM-L6-v2')
      vectors = model.encode(chunks)
    Or with OpenAI:
      from openai import OpenAI
      client = OpenAI()
      response = client.embeddings.create(input=chunks, model="text-embedding-3-small")
      vectors = [r.embedding for r in response.data]
    """

    def __init__(self):
        self.chunks   = []
        self.metadata = []
        self.vocab    = {}
        self.vectors  = None

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r"[a-z]+", text.lower())

    def _build_vocab(self, corpus: list[str]):
        for doc in corpus:
            for tok in self._tokenize(doc):
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    def _tfidf_vector(self, text: str) -> np.ndarray:
        from math import log
        tokens = self._tokenize(text)
        tf     = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        N   = len(self.chunks) + 1
        for t, count in tf.items():
            if t in self.vocab:
                df   = sum(1 for c in self.chunks if t in c.lower()) + 1
                idf  = log(N / df)
                vec[self.vocab[t]] = (count / len(tokens)) * idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add_documents(self, chunks: list[str], meta: list[dict]):
        self.chunks.extend(chunks)
        self.metadata.extend(meta)

    def build_index(self):
        self._build_vocab(self.chunks)
        self.vectors = np.array([self._tfidf_vector(c) for c in self.chunks])
        print(f"  ✓ Vector index built — {len(self.chunks)} chunks, vocab size {len(self.vocab)}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_vec = self._tfidf_vector(query)
        if len(q_vec) < self.vectors.shape[1]:
            q_vec = np.pad(q_vec, (0, self.vectors.shape[1] - len(q_vec)))
        elif len(q_vec) > self.vectors.shape[1]:
            q_vec = q_vec[:self.vectors.shape[1]]
        scores = self.vectors @ q_vec
        top    = np.argsort(scores)[::-1][:top_k]
        return [{"chunk": self.chunks[i], "meta": self.metadata[i], "score": float(scores[i])}
                for i in top]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump(self, f)
        print(f"  ✓ Vector store saved → {path}")

    @staticmethod
    def load(path: str) -> "SimpleVectorStore":
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            return pickle.load(f)

# ── 4. Build knowledge base ───────────────────────────────────────────────────
def build_knowledge_base():
    os.makedirs(POLICIES_DIR, exist_ok=True)
    store  = SimpleVectorStore()
    chunks = []
    metas  = []

    for filename, content in POLICY_DOCUMENTS.items():
        # Save raw policy file
        fpath = os.path.join(POLICIES_DIR, filename)
        with open(fpath, "w") as f:
            f.write(content.strip())

        doc_chunks = chunk_text(content)
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metas.append({
                "source"    : filename,
                "chunk_id"  : i,
                "doc_type"  : filename.replace("_policy.txt", "").replace("_", " ").title(),
            })
        print(f"  ✓ Processed {filename} → {len(doc_chunks)} chunks")

    store.add_documents(chunks, metas)
    store.build_index()
    store.save(INDEX_PATH)
    print(f"\n  Total chunks indexed: {len(chunks)}")
    return store

if __name__ == "__main__":
    print("\n[Phase 1 – Part 2] Building hospital policy knowledge base...")
    store = build_knowledge_base()
    print("\n  Testing retrieval:")
    results = store.search("prior authorisation for surgery", top_k=3)
    for r in results:
        print(f"    [{r['meta']['source']}] score={r['score']:.3f}")
        print(f"    {r['chunk'][:120]}...\n")