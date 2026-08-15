import os
import shutil
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

CHROMA_DIR = "chroma_data"
REGISTRY_PATH = "file_registry.json"
DOCS_FOLDER = "data/documents"


def _calculate_folder_hash(folder_path: str) -> str:
    hasher = hashlib.md5()
    if not os.path.exists(folder_path):
        return ""

    filenames = sorted(os.listdir(folder_path))
    for filename in filenames:
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            hasher.update(filename.encode("utf-8"))
            with open(filepath, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)

    return hasher.hexdigest()


def verify_and_clean_chroma():
    current_hash = _calculate_folder_hash(DOCS_FOLDER)

    stored_hash = ""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            try:
                data = json.load(f)
                stored_hash = data.get("folder_hash", "")
            except json.JSONDecodeError:
                stored_hash = ""

    if current_hash != stored_hash or not os.path.exists(CHROMA_DIR):
        print(
            "🔄 Document changes detected (or missing DB). Cleaning Chroma database..."
        )
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
            print(f"Deleted old {CHROMA_DIR}/")

        with open(REGISTRY_PATH, "w") as f:
            json.dump({"folder_hash": current_hash}, f)

        print("✅ Chroma cleanup complete!")
    else:
        print("⚡ No document changes detected.")


# --- ALLOW IT TO RUN STANDALONE ---
if __name__ == "__main__":
    verify_and_clean_chroma()
