"""CP4.4 evaluator data schemas.

Defines the enumerated vocabularies and validated dataclasses that describe an
evaluator result. This module is the single source of truth for the structured
output shape that the Gemini evaluator must produce, and the persisted
``EvaluationRecord`` that carries reproducibility metadata.

No third-party dependencies (no pydantic). Enum membership is enforced
programmatically so that invalid values are rejected rather than coerced
silently.

The scientific meaning of every field is frozen by the CP4.4 design; only
field names / Python representation are chosen here to match the existing
``src.dataset`` conventions (dataclasses + manual ``validate()``).
"""

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "CP4.4.0"


def _coerce_enum(value: Any, enum_cls: type, field_name: str) -> enum.Enum:
    """Coerce a raw value into ``enum_cls``, raising ValueError if invalid."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (ValueError, TypeError) as exc:  # includes None for non-Optional enums
        allowed = ", ".join(e.value for e in enum_cls)
        raise ValueError(
            "Invalid enum for {0}: {1!r} (allowed: {2})".format(
                field_name, value, allowed
            )
        ) from exc


def _validate_enum(value: Any, enum_cls: type, field_name: str) -> enum.Enum:
    if not isinstance(value, enum_cls):
        return _coerce_enum(value, enum_cls, field_name)
    return value


class EvaluationStatus(enum.Enum):
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    FAILED = "FAILED"

    @classmethod
    def is_terminal(cls) -> set:
        return {cls.SUCCESS, cls.MANUAL_REVIEW, cls.FAILED}


class GenerationLabel(enum.Enum):
    RELIABLE = "RELIABLE"
    UNRELIABLE = "UNRELIABLE"
    INADEQUATE = "INADEQUATE"


class ClaimStatus(enum.Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FailureType(enum.Enum):
    NONE = "NONE"
    FACTUAL_ERROR = "FACTUAL_ERROR"
    CONTEXT_CONTRADICTION = "CONTEXT_CONTRADICTION"
    CONTEXT_UNSUPPORTED = "CONTEXT_UNSUPPORTED"
    FABRICATED_ENTITY = "FABRICATED_ENTITY"
    FABRICATED_CITATION = "FABRICATED_CITATION"
    MISLEADING_PREMISE_ACCEPTANCE = "MISLEADING_PREMISE_ACCEPTANCE"
    UNSUPPORTED_INFERENCE = "UNSUPPORTED_INFERENCE"
    REASONING_ERROR = "REASONING_ERROR"
    AMBIGUOUS_INTERPRETATION = "AMBIGUOUS_INTERPRETATION"
    OTHER = "OTHER"


class Materiality(enum.Enum):
    MATERIAL = "MATERIAL"
    NON_MATERIAL = "NON_MATERIAL"
    UNRESOLVED = "UNRESOLVED"


_GENERATION_LABEL_VALUES = frozenset(e.value for e in GenerationLabel)
_FAILURE_TYPE_VALUES = frozenset(e.value for e in FailureType)
_CLAIM_STATUS_VALUES = frozenset(e.value for e in ClaimStatus)
_MATERIALITY_VALUES = frozenset(e.value for e in Materiality)


@dataclass
class ClaimEvaluation:
    claim_text: str
    claim_status: ClaimStatus
    evidence_text: str
    failure_type: FailureType
    materiality: Materiality = Materiality.NON_MATERIAL

    def validate(self) -> "ClaimEvaluation":
        if not isinstance(self.claim_text, str) or not self.claim_text.strip():
            raise ValueError("claim_text must be a non-empty string")
        if not isinstance(self.evidence_text, str):
            raise ValueError("evidence_text must be a string")
        self.claim_status = _validate_enum(
            self.claim_status, ClaimStatus, "claim_status"
        )
        self.failure_type = _validate_enum(
            self.failure_type, FailureType, "failure_type"
        )
        self.materiality = _validate_enum(
            self.materiality, Materiality, "materiality"
        )
        # UNVERIFIABLE claims must not be tagged with a concrete failure type:
        # the evidence was undecidable, so no specific failure is established.
        if (
            self.claim_status == ClaimStatus.UNVERIFIABLE
            and self.failure_type != FailureType.NONE
            and self.failure_type != FailureType.AMBIGUOUS_INTERPRETATION
            and self.failure_type != FailureType.OTHER
        ):
            raise ValueError(
                "UNVERIFIABLE claim must not carry a concrete failure_type; "
                "got {0}".format(self.failure_type.value)
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "claim_status": self.claim_status.value,
            "evidence_text": self.evidence_text,
            "failure_type": self.failure_type.value,
            "materiality": self.materiality.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimEvaluation":
        try:
            obj = cls(
                claim_text=data["claim_text"],
                claim_status=data["claim_status"],
                evidence_text=data.get("evidence_text", ""),
                failure_type=data["failure_type"],
                materiality=data.get("materiality", Materiality.NON_MATERIAL.value),
            )
        except KeyError as exc:
            raise ValueError("Claim missing required field: {0}".format(exc)) from exc
        return obj.validate()


@dataclass
class StructuredEvaluation:
    """The structured content returned by the evaluator model.

    This is the *content* of an evaluation. Transport-level outcomes (RETRY,
    FAILED) are owned by the wrapper (``EvaluationRecord``) and are never produced
    by the model itself.
    """

    evaluation_status: EvaluationStatus
    claims: List[ClaimEvaluation]
    generation_label: Optional[GenerationLabel]
    generation_failure_types: List[str]
    rationale: str
    needs_human_review: bool = False

    def validate(self) -> "StructuredEvaluation":
        self.evaluation_status = _validate_enum(
            self.evaluation_status, EvaluationStatus, "evaluation_status"
        )
        # Model may only report confidently-resolved states; transport states
        # (RETRY/FAILED) are injected by the wrapper when no content is parsed.
        if self.evaluation_status in (
            EvaluationStatus.RETRY,
            EvaluationStatus.FAILED,
        ):
            raise ValueError(
                "StructuredEvaluation must not carry evaluation_status={0}; "
                "that state is owned by the wrapper.".format(
                    self.evaluation_status.value
                )
            )
        if not isinstance(self.claims, list):
            raise ValueError("claims must be a list")
        self.claims = [ClaimEvaluation.from_dict(c).validate() for c in self.claims]
        if self.generation_label is None:
            raise ValueError("generation_label is required for an evaluation")
        if (
            self.generation_label is not None
            and self.generation_label not in _GENERATION_LABEL_VALUES
        ):
            label = (
                self.generation_label.value
                if isinstance(self.generation_label, GenerationLabel)
                else self.generation_label
            )
            raise ValueError(
                "Invalid generation_label: {0!r} (allowed: {1})".format(
                    label, ", ".join(sorted(_GENERATION_LABEL_VALUES))
                )
            )
        if self.generation_label is not None:
            self.generation_label = _coerce_enum(
                self.generation_label, GenerationLabel, "generation_label"
            )
        if not isinstance(self.generation_failure_types, list):
            raise ValueError("generation_failure_types must be a list")
        for ft in self.generation_failure_types:
            if ft not in _FAILURE_TYPE_VALUES:
                raise ValueError(
                    "Invalid failure type in generation_failure_types: {0!r}".format(
                        ft
                    )
                )
        if not isinstance(self.rationale, str):
            raise ValueError("rationale must be a string")
        if not isinstance(self.needs_human_review, bool):
            raise ValueError("needs_human_review must be a boolean")
        # Semantic consistency: INADEQUATE must not also assert a CONCRETE
        # material failure set without review, but we do not hard-reject here;
        # the wrapper normalises the terminal label downstream.
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_status": self.evaluation_status.value,
            "claims": [c.to_dict() for c in self.claims],
            "generation_label": (
                self.generation_label.value
                if self.generation_label is not None
                else None
            ),
            "generation_failure_types": list(self.generation_failure_types),
            "rationale": self.rationale,
            "needs_human_review": self.needs_human_review,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredEvaluation":
        try:
            obj = cls(
                evaluation_status=data["evaluation_status"],
                claims=data["claims"],
                generation_label=data.get("generation_label"),
                generation_failure_types=data.get("generation_failure_types", []),
                rationale=data.get("rationale", ""),
                needs_human_review=bool(data.get("needs_human_review", False)),
            )
        except KeyError as exc:
            raise ValueError(
                "StructuredEvaluation missing required field: {0}".format(exc)
            ) from exc
        return obj.validate()


@dataclass
class EvaluationRecord:
    """Full persisted evaluation record with reproducibility metadata.

    A terminal record's ``evaluation_status`` is in
    {SUCCESS, MANUAL_REVIEW, FAILED}. When ``evaluation_status == FAILED`` the
    record MUST NOT carry a generation label and MUST NOT contribute to
    empirical risk.
    """

    schema_version: str
    evaluator_model: str
    prompt_version: str
    timestamp: str
    source_generation_id: str
    dataset_id: str
    evaluation_status: EvaluationStatus
    retry_count: int
    claims: List[ClaimEvaluation] = field(default_factory=list)
    generation_label: Optional[GenerationLabel] = None
    generation_failure_types: List[str] = field(default_factory=list)
    rationale: str = ""
    needs_human_review: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> "EvaluationRecord":
        self.evaluation_status = _validate_enum(
            self.evaluation_status, EvaluationStatus, "evaluation_status"
        )
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.evaluator_model, str):
            raise ValueError("evaluator_model must be a string")
        if not isinstance(self.source_generation_id, str):
            raise ValueError("source_generation_id must be a string")
        if not isinstance(self.dataset_id, str):
            raise ValueError("dataset_id must be a string")
        if not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        for c in self.claims:
            if not isinstance(c, ClaimEvaluation):
                raise ValueError("claims must be ClaimEvaluation instances")
        for ft in self.generation_failure_types:
            if ft not in _FAILURE_TYPE_VALUES:
                raise ValueError(
                    "Invalid failure type: {0!r}".format(ft)
                )
        # CRITICAL invariant: a FAILED evaluation must never be converted into a
        # generation label and must never contribute to empirical risk.
        if self.evaluation_status == EvaluationStatus.FAILED:
            if self.generation_label is not None:
                raise ValueError(
                    "FAILED evaluation must never carry a generation_label; "
                    "got {0!r}".format(
                        self.generation_label.value
                        if isinstance(self.generation_label, GenerationLabel)
                        else self.generation_label
                    )
                )
            if self.claims:
                raise ValueError("FAILED evaluation must not carry claims")
        else:
            # A successfully- or manually-resolved evaluation must carry a label;
            # the wrapper never fabricates one for a FAILED record.
            if self.generation_label is None:
                raise ValueError(
                    "evaluation_status {0} must carry a generation_label".format(
                        self.evaluation_status.value
                    )
                )
        if self.generation_label is not None:
            self.generation_label = _validate_enum(
                self.generation_label, GenerationLabel, "generation_label"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_model": self.evaluator_model,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp,
            "source_generation_id": self.source_generation_id,
            "dataset_id": self.dataset_id,
            "evaluation_status": self.evaluation_status.value,
            "retry_count": self.retry_count,
            "claims": (
                None
                if self.evaluation_status == EvaluationStatus.FAILED
                else [c.to_dict() for c in self.claims]
            ),
            "generation_label": (
                self.generation_label.value
                if self.generation_label is not None
                else None
            ),
            "generation_failure_types": list(self.generation_failure_types),
            "rationale": self.rationale,
            "needs_human_review": self.needs_human_review,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attempts": list(self.attempts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRecord":
        try:
            obj = cls(
                schema_version=data["schema_version"],
                evaluator_model=data["evaluator_model"],
                prompt_version=data["prompt_version"],
                timestamp=data["timestamp"],
                source_generation_id=data["source_generation_id"],
                dataset_id=data["dataset_id"],
                evaluation_status=data["evaluation_status"],
                retry_count=data["retry_count"],
                claims=[
                    ClaimEvaluation.from_dict(c).validate()
                    for c in (data.get("claims") or [])
                ],
                generation_label=data.get("generation_label"),
                generation_failure_types=data.get("generation_failure_types", []),
                rationale=data.get("rationale", ""),
                needs_human_review=bool(data.get("needs_human_review", False)),
                error_type=data.get("error_type"),
                error_message=data.get("error_message"),
                attempts=data.get("attempts", []),
            )
        except KeyError as exc:
            raise ValueError(
                "EvaluationRecord missing required field: {0}".format(exc)
            ) from exc
        return obj.validate()

    @classmethod
    def from_structured(
        cls,
        structured: StructuredEvaluation,
        *,
        evaluator_model: str,
        prompt_version: str,
        schema_version: str,
        source_generation_id: str,
        dataset_id: str,
        retry_count: int,
        timestamp: str,
        attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> "EvaluationRecord":
        """Build a terminal record from a validated StructuredEvaluation."""
        return cls(
            schema_version=schema_version,
            evaluator_model=evaluator_model,
            prompt_version=prompt_version,
            timestamp=timestamp,
            source_generation_id=source_generation_id,
            dataset_id=dataset_id,
            evaluation_status=structured.evaluation_status,
            retry_count=retry_count,
            claims=structured.claims,
            generation_label=structured.generation_label,
            generation_failure_types=structured.generation_failure_types,
            rationale=structured.rationale,
            needs_human_review=structured.needs_human_review,
            error_type=None,
            error_message=None,
            attempts=attempts or [],
        ).validate()


# --------------------------------------------------------------------------
# Reference qualitative label inference (mock / audit use only)
# --------------------------------------------------------------------------
# This is NOT the scientific judge. The frozen Gemini evaluator is the judge on
# real runs. This helper exists only so the mock pilot can produce deterministic,
# auditable labels and so the materiality decision rule is explicit and tested.
# It uses qualitative (material vs non-material) reasoning only — no numeric
# thresholds — and matches the frozen generation-label guidance in the prompt.


def is_failure_claim(claim: ClaimEvaluation) -> bool:
    """True if the claim is a MATERIAL reliability failure that escalates to the
    generation level.

    A claim escalates only when it is a concrete failure (CONTRADICTED or
    UNSUPPORTED with a failure_type other than NONE) AND its materiality is
    MATERIAL. SUPPORTED, NOT_APPLICABLE and UNVERIFIABLE claims are never
    failures; a NON_MATERIAL claim is a real disagreement but does not escalate
    the whole generation.
    """
    if claim.materiality != Materiality.MATERIAL:
        return False
    if claim.claim_status in (ClaimStatus.SUPPORTED, ClaimStatus.NOT_APPLICABLE):
        return False
    if claim.claim_status == ClaimStatus.UNVERIFIABLE:
        return False
    return claim.failure_type != FailureType.NONE


def infer_generation_label_from_claims(
    claims: List["ClaimEvaluation"], needs_human_review: bool = False
) -> "GenerationLabel":
    """Reference qualitative rule deriving a generation label from its claims.

    Rules (qualitative, no numeric thresholds):
      * any MATERIAL reliability failure claim -> UNRELIABLE
      * (else) needs_human_review or any UNRESOLVED materiality -> INADEQUATE
      * (else) -> RELIABLE

    A single NON_MATERIAL claim never escalates a generation to UNRELIABLE.
    """
    if any(
        c.materiality == Materiality.MATERIAL and is_failure_claim(c) for c in claims
    ):
        return GenerationLabel.UNRELIABLE
    if needs_human_review or any(
        c.materiality == Materiality.UNRESOLVED for c in claims
    ):
        return GenerationLabel.INADEQUATE
    return GenerationLabel.RELIABLE
