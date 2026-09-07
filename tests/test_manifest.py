import unittest
from src.dataset.manifest import DatasetManifest, VALID_STATUSES

class TestDatasetManifest(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            "dataset_id": "test_ds_01",
            "dataset_name": "Test Dataset",
            "source_reference": "https://example.com/data.json",
            "source_version": "1.0",
            "acquisition_date": "2026-08-30",
            "license_type": "MIT",
            "raw_file_paths": ["data/raw/test.json"],
            "file_size": 12345,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "processing_version": "0.1.0",
            "status": "ACQUIRED",
            "notes": "Test notes"
        }

    def test_valid_manifest(self):
        manifest = DatasetManifest.from_dict(self.valid_data)
        try:
            manifest.validate()
        except Exception as e:
            self.fail(f"Valid manifest raised validation error: {e}")

    def test_missing_required_field(self):
        invalid_data = self.valid_data.copy()
        invalid_data["dataset_id"] = ""
        manifest = DatasetManifest(**invalid_data)
        with self.assertRaises(ValueError):
            manifest.validate()

    def test_invalid_status(self):
        invalid_data = self.valid_data.copy()
        invalid_data["status"] = "INVALID_STATUS_XYZ"
        manifest = DatasetManifest(**invalid_data)
        with self.assertRaises(ValueError):
            manifest.validate()

    def test_invalid_sha256(self):
        invalid_data = self.valid_data.copy()
        invalid_data["sha256"] = "not_a_valid_sha256_hash"
        manifest = DatasetManifest(**invalid_data)
        with self.assertRaises(ValueError):
            manifest.validate()

    def test_serialization_deserialization(self):
        manifest = DatasetManifest.from_dict(self.valid_data)
        json_str = manifest.to_json()
        
        reconstructed = DatasetManifest.from_json(json_str)
        self.assertEqual(manifest.dataset_id, reconstructed.dataset_id)
        self.assertEqual(manifest.status, reconstructed.status)
        self.assertEqual(manifest.sha256, reconstructed.sha256)

if __name__ == "__main__":
    unittest.main()
