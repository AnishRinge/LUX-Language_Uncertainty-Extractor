import json
from pathlib import Path
from typing import List, Dict, Any, Union
from .schema import CanonicalPromptScenario, SourceProvenance, PromptContent, ReferenceContent, ScenarioMetadata

class SQuADAdapter:
    def __init__(self, raw_file_path: Union[str, Path], split: str = "train"):
        self.raw_file_path = Path(raw_file_path)
        self.split = split
        self._raw_data: Optional[Dict[str, Any]] = None

    def load_raw(self) -> Dict[str, Any]:
        if self._raw_data is None:
            if not self.raw_file_path.exists():
                raise FileNotFoundError(f"SQuAD raw file not found at {self.raw_file_path}")
            with open(self.raw_file_path, "r", encoding="utf-8") as f:
                self._raw_data = json.load(f)
        return self._raw_data

    def validate_raw(self) -> bool:
        data = self.load_raw()
        if not isinstance(data, dict):
            raise ValueError("SQuAD raw data top-level must be a dictionary.")
        if "data" not in data:
            raise ValueError("SQuAD raw data missing 'data' key.")
        
        articles = data["data"]
        if not isinstance(articles, list):
            raise ValueError("SQuAD 'data' field must be a list of articles.")
        
        for art_idx, article in enumerate(articles):
            if not isinstance(article, dict):
                raise ValueError(f"Article at index {art_idx} is not a dictionary.")
            if "title" not in article or "paragraphs" not in article:
                raise ValueError(f"Article at index {art_idx} missing 'title' or 'paragraphs'.")
            
            paragraphs = article["paragraphs"]
            if not isinstance(paragraphs, list):
                raise ValueError(f"Paragraphs in article {art_idx} must be a list.")
            
            for p_idx, para in enumerate(paragraphs):
                if not isinstance(para, dict):
                    raise ValueError(f"Paragraph at index {p_idx} in article {art_idx} is not a dictionary.")
                if "context" not in para or "qas" not in para:
                    raise ValueError(f"Paragraph at index {p_idx} in article {art_idx} missing 'context' or 'qas'.")
                
                qas = para["qas"]
                if not isinstance(qas, list):
                    raise ValueError(f"QAs in paragraph {p_idx} of article {art_idx} must be a list.")
                
                for qa_idx, qa in enumerate(qas):
                    if not isinstance(qa, dict):
                        raise ValueError(f"QA record at index {qa_idx} is not a dictionary.")
                    if "id" not in qa or "question" not in qa or "is_impossible" not in qa:
                        raise ValueError(f"QA record at index {qa_idx} missing required fields ('id', 'question', 'is_impossible').")
        return True

    def normalize(self) -> List[CanonicalPromptScenario]:
        self.validate_raw()
        data = self.load_raw()
        articles = data["data"]
        
        scenarios: List[CanonicalPromptScenario] = []
        seen_ids = set()

        for art_idx, article in enumerate(articles):
            title = article.get("title", f"article_{art_idx}")
            paragraphs = article.get("paragraphs", [])
            
            for p_idx, para in enumerate(paragraphs):
                context = para.get("context", "")
                qas = para.get("qas", [])
                
                for qa_idx, qa in enumerate(qas):
                    orig_id = qa["id"]
                    sample_id = f"squad_v2_{orig_id}"
                    
                    if sample_id in seen_ids:
                        raise ValueError(f"Duplicate sample_id detected: {sample_id}")
                    seen_ids.add(sample_id)
                    
                    question = qa["question"]
                    is_impossible = qa["is_impossible"]
                    
                    if is_impossible:
                        answerability = "UNANSWERABLE_FROM_CONTEXT"
                        expected_behavior = "ABSTAIN"
                        answers = []
                    else:
                        answerability = "ANSWERABLE"
                        expected_behavior = "ANSWER"
                        answers = qa.get("answers", [])

                    source = SourceProvenance(
                        dataset="squad_v2",
                        split=self.split,
                        original_id=orig_id,
                        article_id=str(art_idx),
                        paragraph_id=str(p_idx),
                        article_title=title
                    )
                    
                    prompt = PromptContent(
                        question=question,
                        context=context
                    )
                    
                    reference = ReferenceContent(
                        answerability=answerability,
                        answers=answers,
                        supporting_information=None
                    )
                    
                    metadata = ScenarioMetadata(
                        phenomenon="grounded_contextual_reliability",
                        expected_behavior=expected_behavior,
                        source_metadata={
                            "is_impossible": is_impossible
                        }
                    )
                    
                    scenario = CanonicalPromptScenario(
                        sample_id=sample_id,
                        source=source,
                        prompt=prompt,
                        reference=reference,
                        metadata=metadata
                    )
                    scenarios.append(scenario)
                    
        return scenarios

    def validate_normalized(self, scenarios: List[CanonicalPromptScenario]) -> bool:
        seen_ids = set()
        answerable_count = 0
        unanswerable_count = 0

        for scenario in scenarios:
            scenario.validate()
            if scenario.sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id found in normalized scenarios: {scenario.sample_id}")
            seen_ids.add(scenario.sample_id)

            if scenario.reference.answerability == "ANSWERABLE":
                answerable_count += 1
            elif scenario.reference.answerability == "UNANSWERABLE_FROM_CONTEXT":
                unanswerable_count += 1

        total = len(scenarios)
        if total != 130319:
            raise ValueError(f"Expected 130319 normalized records, got {total}.")
        if answerable_count != 86821:
            raise ValueError(f"Expected 86821 answerable records, got {answerable_count}.")
        if unanswerable_count != 43498:
            raise ValueError(f"Expected 43498 unanswerable records, got {unanswerable_count}.")

        return True

    def export_prompt_scenarios(self, output_path: Union[str, Path]) -> None:
        scenarios = self.normalize()
        self.validate_normalized(scenarios)
        
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_to_export = [s.to_dict() for s in scenarios]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data_to_export, f, indent=2)
