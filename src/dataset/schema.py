import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class SourceProvenance:
    dataset: str
    split: str
    original_id: str
    article_id: str
    paragraph_id: str
    article_title: str

@dataclass
class PromptContent:
    question: str
    context: str

@dataclass
class ReferenceContent:
    answerability: str  # "ANSWERABLE" or "UNANSWERABLE_FROM_CONTEXT"
    answers: List[Dict[str, Any]]
    supporting_information: Optional[str] = None

@dataclass
class ScenarioMetadata:
    phenomenon: str
    expected_behavior: str
    source_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CanonicalPromptScenario:
    sample_id: str
    source: SourceProvenance
    prompt: PromptContent
    reference: ReferenceContent
    metadata: ScenarioMetadata

    def validate(self) -> None:
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise ValueError("sample_id must be a non-empty string.")
        
        if not self.source.dataset or not self.source.original_id:
            raise ValueError("source dataset and original_id must be present.")
        
        if not self.prompt.question or not isinstance(self.prompt.question, str) or not self.prompt.question.strip():
            raise ValueError("question must be a non-empty string.")
        
        if not self.prompt.context or not isinstance(self.prompt.context, str) or not self.prompt.context.strip():
            raise ValueError("context must be a non-empty string.")
        
        valid_answerabilities = {"ANSWERABLE", "UNANSWERABLE_FROM_CONTEXT"}
        if self.reference.answerability not in valid_answerabilities:
            raise ValueError(f"Invalid answerability: {self.reference.answerability}")
        
        if self.reference.answerability == "ANSWERABLE":
            if not self.reference.answers or not isinstance(self.reference.answers, list):
                raise ValueError("ANSWERABLE samples must contain reference answers.")

        # Check for forbidden future-stage fields
        forbidden_fields = [
            "qwen_answer", "generated_answer", "hallucination_label",
            "empirical_risk", "evaluator_output", "hidden_states", "predictor_features"
        ]
        for field_name in forbidden_fields:
            if hasattr(self, field_name):
                raise ValueError(f"Forbidden future-stage field present: {field_name}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalPromptScenario":
        source = SourceProvenance(**data["source"])
        prompt = PromptContent(**data["prompt"])
        reference = ReferenceContent(**data["reference"])
        metadata = ScenarioMetadata(**data["metadata"])
        scenario = cls(
            sample_id=data["sample_id"],
            source=source,
            prompt=prompt,
            reference=reference,
            metadata=metadata
        )
        scenario.validate()
        return scenario
