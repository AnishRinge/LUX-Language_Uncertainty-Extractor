import unittest
import tempfile
from pathlib import Path
from src.utils.hashing import compute_sha256

class TestHashing(unittest.TestCase):
    def test_compute_sha256_known_content(self):
        # Known bytes: b\"hello world\\n\" -> sha256: a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            with open(test_file, "wb") as f:
                f.write(b"hello world\n")
            
            expected_hash = "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
            computed_hash = compute_sha256(test_file)
            self.assertEqual(computed_hash, expected_hash)

    def test_compute_sha256_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_file = Path(tmpdir) / "nonexistent.txt"
            with self.assertRaises(FileNotFoundError):
                compute_sha256(missing_file)

    def test_compute_sha256_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(IsADirectoryError):
                compute_sha256(tmpdir)

if __name__ == "__main__":
    unittest.main()
