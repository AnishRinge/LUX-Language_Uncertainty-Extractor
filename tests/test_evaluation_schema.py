"""Schema-level tests for the CP4.4 evaluator.

Covers: valid structured output, invalid enum values, missing required fields,
UNVERIFIABLE != UNSUPPORTED, materiality escalation rules, the FAILED invariant
(never a generation label), and serialization/deserialization round-trips.
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.schemas import (
    SCHEMA_VERSION,
    ClaimEvaluation,
    ClaimStatus,
    EvaluationRecord,
    EvaluationStatus,
    FailureType,
    GenerationLabel,
    Materiality,
    StructuredEvaluation,
    infer_generation_label_from_claims,
    is_failure_claim,
)


def _valid_claim(status=ClaimStatus.SUPPORTED, failure=FailureType.NONE,
                  materiality=Materiality.NON_MATERIAL):
    return {
        "claim_text": "The context states X.",
        "claim_status": status.value,
        "evidence_text": "passage sentence",
        "failure_type": failure.value,
        "materiality": materiality.value,
    }


def _valid_structured(**overrides):
    base = {
        "evaluation_status": "SUCCESS",
        "claims": [_valid_claim()],
        "generation_label": "RELIABLE",
        "generation_failure_types": [],
        "rationale": "no failure",
        "needs_human_review": False,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# 1. valid structured evaluator output
# --------------------------------------------------------------------------
class TestValidStructuredOutput(unittest.TestCase):
    def test_parses_valid_structured_evaluation(self):
        structured = StructuredEvaluation.from_dict(_valid_structured())
        self.assertEqual(structured.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(structured.generation_label, GenerationLabel.RELIABLE)
        self.assertEqual(len(structured.claims), 1)
        self.assertEqual(structured.claims[0].claim_status, ClaimStatus.SUPPORTED)

    def test_claim_enums_coerced_and_strict(self):
        structured = StructuredEvaluation.from_dict(_valid_structured())
        self.assertIsInstance(structured.claims[0].claim_status, ClaimStatus)
        self.assertIsInstance(structured.claims[0].failure_type, FailureType)
        self.assertIsInstance(structured.claims[0].materiality, Materiality)

    def test_to_dict_is_schema_shaped(self):
        structured = StructuredEvaluation.from_dict(_valid_structured())
        d = structured.to_dict()
        self.assertEqual(
            set(d.keys()),
            {
                "evaluation_status", "claims", "generation_label",
                "generation_failure_types", "rationale", "needs_human_review",
            },
        )
        self.assertEqual(d["generation_label"], GenerationLabel.RELIABLE.value)


# --------------------------------------------------------------------------
# 2. invalid enum values
# --------------------------------------------------------------------------
class TestInvalidEnumValues(unittest.TestCase):
    def test_invalid_generation_label_rejected(self):
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(_valid_structured(generation_label="MAYBE"))

    def test_invalid_claim_status_rejected(self):
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(
                _valid_structured(claims=[_valid_claim(status=type("X", (), {"value": "GIBBERISH"})())])
            )

    def test_invalid_failure_type_in_claim_rejected(self):
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(_valid_structured(claims=[_valid_claim(failure=type("X",(),{"value":"PHANTOM"})())]))

    def test_invalid_failure_type_in_generation_failure_types(self):
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(_valid_structured(generation_failure_types=["NOT_A_FAILURE"]))

    def test_invalid_materiality_rejected(self):
        bad = {"claim_text": "x", "claim_status": "SUPPORTED",
               "evidence_text": "y", "failure_type": "NONE", "materiality": "KIND_OF"}
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(_valid_structured(claims=[bad]))


# --------------------------------------------------------------------------
# 3. missing required fields
# --------------------------------------------------------------------------
class TestMissingRequiredFields(unittest.TestCase):
    def test_missing_claims_rejected(self):
        data = _valid_structured()
        del data["claims"]
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(data)

    def test_missing_generation_label_rejected(self):
        data = _valid_structured()
        del data["generation_label"]
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(data)

    def test_missing_required_claim_field_rejected(self):
        bad_claim = _valid_claim()
        del bad_claim["claim_text"]
        with self.assertRaises(ValueError):
            StructuredEvaluation.from_dict(_valid_structured(claims=[bad_claim]))

    def test_missing_record_metadata_rejected(self):
        base = {
            "schema_version": SCHEMA_VERSION,
            "evaluator_model": "gemini-3.8-flash",
            "prompt_version": "CP4.4.0",
            "timestamp": "2026-10-06T00:00:00+00:00",
            "source_generation_id": "s1",
            "dataset_id": "squad_v2",
            "evaluation_status": "SUCCESS",
            "retry_count": 0,
        }
        with self.assertRaises(ValueError):
            EvaluationRecord.from_dict({k: v for k, v in base.items()})


# --------------------------------------------------------------------------
# 9. FAILED never becomes a generation label (record-level invariant)
# --------------------------------------------------------------------------
class TestFailedInvariant(unittest.TestCase):
    def _base_failed(self):
        return EvaluationRecord(
            schema_version=SCHEMA_VERSION,
            evaluator_model="gemini-3.8-flash",
            prompt_version="CP4.4.0",
            timestamp="2026-10-06T00:00:00+00:00",
            source_generation_id="s1",
            dataset_id="squad_v2",
            evaluation_status=EvaluationStatus.FAILED,
            retry_count=2,
            claims=[],
            generation_label=None,
            generation_failure_types=[],
            rationale="",
            needs_human_review=False,
            error_type="TransientEvaluationError",
            error_message="rate limited",
            attempts=[{"attempt": 1, "status": "RETRY"}],
        )

    def test_failed_record_has_no_generation_label(self):
        rec = self._base_failed().validate()
        self.assertIsNone(rec.generation_label)
        self.assertEqual(rec.evaluation_status, EvaluationStatus.FAILED)

    def test_failed_record_carries_error_info(self):
        rec = self._base_failed().validate()
        self.assertEqual(rec.error_type, "TransientEvaluationError")
        self.assertIn("rate limited", rec.error_message)

    def test_failed_with_label_is_rejected(self):
        bad = self._base_failed()
        bad.generation_label = GenerationLabel.RELIABLE
        with self.assertRaises(ValueError):
            bad.validate()

    def test_failed_with_claims_is_rejected(self):
        bad = self._base_failed()
        bad.claims = [ClaimEvaluation.from_dict(_valid_claim()).validate()]
        with self.assertRaises(ValueError):
            bad.validate()

    def test_failed_record_round_trips(self):
        rec = self._base_failed().validate()
        d = rec.to_dict()
        back = EvaluationRecord.from_dict(d)
        self.assertIsNone(back.generation_label)
        self.assertEqual(back.evaluation_status, EvaluationStatus.FAILED)


# --------------------------------------------------------------------------
# 10. UNVERIFIABLE != UNSUPPORTED
# --------------------------------------------------------------------------
class TestUnverifiableVsUnsupported(unittest.TestCase):
    def test_distinct_enum_members(self):
        self.assertNotEqual(ClaimStatus.UNVERIFIABLE.value, ClaimStatus.UNSUPPORTED.value)
        self.assertIsInstance(ClaimStatus("UNVERIFIABLE"), ClaimStatus)
        self.assertIsInstance(ClaimStatus("UNSUPPORTED"), ClaimStatus)

    def test_unverifiable_claim_is_not_a_failure_claim(self):
        c = ClaimEvaluation.from_dict(
            _valid_claim(
                status=ClaimStatus.UNVERIFIABLE, failure=FailureType.AMBIGUOUS_INTERPRETATION
            )
        ).validate()
        self.assertFalse(is_failure_claim(c))

    def test_unverifiable_claim_with_concrete_failure_rejected(self):
        # A UNVERIFIABLE claim must not carry a concrete failure_type.
        with self.assertRaises(ValueError):
            ClaimEvaluation.from_dict(
                _valid_claim(status=ClaimStatus.UNVERIFIABLE, failure=FailureType.FABRICATED_ENTITY)
            ).validate()

    def test_supported_and_contradicted_are_distinct(self):
        self.assertNotEqual(ClaimStatus.SUPPORTED.value, ClaimStatus.CONTRADICTED.value)


# --------------------------------------------------------------------------
# 11. one non-material claim does not automatically produce UNRELIABLE
# 12. material contradiction can produce UNRELIABLE
# --------------------------------------------------------------------------
class TestMaterialityEscalation(unittest.TestCase):
    def _claim(self, status, failure, materiality):
        return ClaimEvaluation.from_dict(
            {
                "claim_text": "claim",
                "claim_status": status.value,
                "evidence_text": "ev",
                "failure_type": failure.value,
                "materiality": materiality.value,
            }
        ).validate()

    def test_non_material_unsupported_claim_stays_reliable(self):
        claims = [
            self._claim(ClaimStatus.SUPPORTED, FailureType.NONE, Materiality.NON_MATERIAL),
            # A peripheral, non-material issue must not escalate.
            self._claim(ClaimStatus.UNSUPPORTED, FailureType.UNSUPPORTED_INFERENCE, Materiality.NON_MATERIAL),
        ]
        label = infer_generation_label_from_claims(claims)
        self.assertEqual(label, GenerationLabel.RELIABLE)

    def test_non_material_claim_is_not_a_failure(self):
        c = self._claim(ClaimStatus.CONTRADICTED, FailureType.CONTEXT_CONTRADICTION, Materiality.NON_MATERIAL)
        self.assertFalse(is_failure_claim(c))

    def test_material_contradiction_produces_unreliable(self):
        claims = [
            self._claim(ClaimStatus.SUPPORTED, FailureType.NONE, Materiality.NON_MATERIAL),
            self._claim(ClaimStatus.CONTRADICTED, FailureType.CONTEXT_CONTRADICTION, Materiality.MATERIAL),
        ]
        label = infer_generation_label_from_claims(claims)
        self.assertEqual(label, GenerationLabel.UNRELIABLE)

    def test_material_fabricated_entity_produces_unreliable(self):
        claims = [
            self._claim(ClaimStatus.UNSUPPORTED, FailureType.FABRICATED_ENTITY, Materiality.MATERIAL),
        ]
        label = infer_generation_label_from_claims(claims)
        self.assertEqual(label, GenerationLabel.UNRELIABLE)

    def test_unverifiable_claim_does_not_escalate(self):
        claims = [
            self._claim(ClaimStatus.UNVERIFIABLE, FailureType.AMBIGUOUS_INTERPRETATION, Materiality.UNRESOLVED),
        ]
        label = infer_generation_label_from_claims(claims)
        self.assertEqual(label, GenerationLabel.INADEQUATE)

    def test_empty_claims_reliable(self):
        self.assertEqual(infer_generation_label_from_claims([]), GenerationLabel.RELIABLE)

    def test_needs_human_review_makes_inadequate(self):
        claims = [self._claim(ClaimStatus.SUPPORTED, FailureType.NONE, Materiality.NON_MATERIAL)]
        label = infer_generation_label_from_claims(claims, needs_human_review=True)
        self.assertEqual(label, GenerationLabel.INADEQUATE)


# --------------------------------------------------------------------------
# 15. schema serialization / deserialization
# --------------------------------------------------------------------------
class TestSerialization(unittest.TestCase):
    def test_structured_round_trip(self):
        structured = StructuredEvaluation.from_dict(_valid_structured())
        d = structured.to_dict()
        back = StructuredEvaluation.from_dict(d)
        self.assertEqual(back.evaluation_status, structured.evaluation_status)
        self.assertEqual(back.generation_label, structured.generation_label)
        self.assertEqual(len(back.claims), len(structured.claims))

    def test_record_round_trip_preserves_status_and_label(self):
        rec = EvaluationRecord(
            schema_version=SCHEMA_VERSION,
            evaluator_model="gemini-3.8-flash",
            prompt_version="CP4.4.0",
            timestamp="2026-10-06T00:00:00+00:00",
            source_generation_id="s1",
            dataset_id="squad_v2",
            evaluation_status=EvaluationStatus.SUCCESS,
            retry_count=1,
            claims=[ClaimEvaluation.from_dict(_valid_claim()).validate()],
            generation_label=GenerationLabel.RELIABLE,
            generation_failure_types=[],
            rationale="ok",
            needs_human_review=False,
            attempts=[{"attempt": 1, "status": "RETRY"}, {"attempt": 2, "status": "SUCCESS"}],
        ).validate()
        d = rec.to_dict()
        back = EvaluationRecord.from_dict(d)
        self.assertEqual(back.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(back.generation_label, GenerationLabel.RELIABLE)
        self.assertEqual(back.retry_count, 1)
        self.assertEqual(len(back.attempts), 2)

    def test_record_to_dict_is_json_serializable(self):
        import json
        rec = EvaluationRecord(
            schema_version=SCHEMA_VERSION,
            evaluator_model="gemini-3.8-flash",
            prompt_version="CP4.4.0",
            timestamp="2026-10-06T00:00:00+00:00",
            source_generation_id="s1",
            dataset_id="squad_v2",
            evaluation_status=EvaluationStatus.SUCCESS,
            retry_count=0,
            generation_label=GenerationLabel.RELIABLE,
        ).validate()
        json.dumps(rec.to_dict())


if __name__ == "__main__":
    unittest.main()
