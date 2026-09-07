import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

VALID_STATUSES = {
    "NOT_ACQUIRED",
    "ACQUIRED",
    "VALIDATED",
    "NORMALIZED",
    "APPROVED_FOR_PILOT",
    "APPROVED_FOR_TARGET_GENERATION",
    "REJECTED"
}

SHA256_REGEX = re.compile(r"^[a-fA-F0-9]{64}$")

@dataclass
class DatasetManifest:
    dataset_id: str
    dataset_name: str
    source_reference: str  # source_url or source_reference
    source_version: str
    acquisition_date: str
    license_type: str  # represented as license in requirements, using license_type or license (let's support both or name it license)
    raw_file_paths: List[str] = field(default_factory=list)
    file_size: Optional[Union[int, List[int], Dict[str, int]]] = None
    sha256: Optional[Union[str, List[str], Dict[str, str]]] = None
    processing_version: str = "0.1.0"
    status: str = "NOT_ACQUIRED"
    notes: Optional[str] = None

    def validate(self) -> None:
        """
        Validates the dataset manifest.
        Raises ValueError if any required field is missing, malformed, invalid status, or invalid hash format.
        """
        # Required string fields check
        required_fields = {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "source_reference": self.source_reference,
            "source_version": self.source_version,
            "acquisition_date": self.acquisition_date,
            "license_type": self.license_type,
            "processing_version": self.processing_version,
            "status": self.status
        }
        
        for field_name, value in required_fields.items():
            if not value or not isinstance(value, str) or not value.strip():
                raise ValueError(f"Manifest validation error: '{field_name}' is required and must be a non-empty string.")

        # Validate status
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Manifest validation error: Invalid status '{self.status}'. Must be one of {VALID_STATUSES}.")

        # Validate raw_file_paths
        if not isinstance(self.raw_file_paths, list):
            raise ValueError("Manifest validation error: 'raw_file_paths' must be a list of strings.")
        
        for path in self.raw_file_paths:
            if not isinstance(path, str) or not path.strip():
                raise ValueError("Manifest validation error: Each entry in 'raw_file_paths' must be a non-empty string.")

        # Validate sha256 if provided
        if self.sha256 is not None:
            if isinstance(self.sha256, str):
                if not SHA256_REGEX.match(self.sha256):
                    raise ValueError(f"Manifest validation error: Invalid SHA-256 hash format '{self.sha256}'.")
            elif isinstance(self.sha256, list):
                for h in self.sha256:
                    if not isinstance(h, str) or not SHA256_REGEX.match(h):
                        raise ValueError(f"Manifest validation error: Invalid SHA-256 hash format in list '{h}'.")
            elif isinstance(self.sha256, dict):
                for k, h in self.sha256.items():
                    if not isinstance(h, str) or not SHA256_REGEX.match(h):
                        raise ValueError(f"Manifest validation error: Invalid SHA-256 hash format in dict for key '{k}': '{h}'.")
            else:
                raise ValueError("Manifest validation error: 'sha256' must be a string, list of strings, or dictionary of strings.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        # Handle field mapping if 'license' is passed instead of 'license_type'
        if "license" in data and "license_type" not in data:
            data = data.copy()
            data["license_type"] = data.pop("license")
        
        manifest = cls(**data)
        manifest.validate()
        return manifest

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetManifest":
        data = json.loads(json_str)
        return cls.from_dict(data)
