"""
setup_data.py
Run this ONCE before starting the app.
Downloads the O*NET 28.0 database and extracts the two files we need.

Usage:
    python setup_data.py
"""

import os
import zipfile
import urllib.request
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ONET_URL = "https://www.onetcenter.org/dl_files/database/db_28_0_text.zip"
ZIP_PATH = os.path.join(DATA_DIR, "onet.zip")

# Exact filenames we want — must match precisely, not as a suffix
REQUIRED_FILES = [
    "Occupation Data.txt",
    "Skills.txt",
]


def download_onet():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading O*NET database (~30 MB)...")
    urllib.request.urlretrieve(ONET_URL, ZIP_PATH)
    print("Download complete.")

    print("Extracting required files...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = z.namelist()

        # Debug: print all available txt files so we can see what's in the zip
        txt_files = [n for n in all_names if n.endswith(".txt")]
        print(f"\nFound {len(txt_files)} .txt files in zip.")

        for required in REQUIRED_FILES:
            # Match on exact filename (basename), not suffix match
            match = next(
                (n for n in all_names if os.path.basename(n) == required),
                None
            )
            if match:
                z.extract(match, DATA_DIR)
                extracted_path = os.path.join(DATA_DIR, match)
                final_path = os.path.join(DATA_DIR, required)
                if extracted_path != final_path:
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    shutil.move(extracted_path, final_path)
                print(f"  ✅ {required}")
            else:
                print(f"  ❌ Could not find '{required}' in zip.")
                print(f"     Available files containing 'Skills': {[n for n in all_names if 'Skills' in n]}")

    # Clean up zip and any leftover subdirectories
    os.remove(ZIP_PATH)
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print("\n✅ Setup complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    download_onet()
