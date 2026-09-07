import hashlib
from pathlib import Path
from typing import Union

def compute_sha256(file_path: Union[str, Path]) -> str:
    """
    Calculate the SHA-256 hash of a file safely in chunks.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Hex digest string of the SHA-256 hash.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path points to a directory.
        PermissionError: If the file cannot be read due to permissions.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        
    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        if isinstance(e, (FileNotFoundError, IsADirectoryError, PermissionError)):
            raise
        raise IOError(f"Error reading file {path}: {e}")
