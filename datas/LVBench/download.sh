#!/bin/bash

DATASET_ID="AIWinter/LVBench"
SAVE_DIR="./videos"
TMP_DIR="./tmp_downloads"

mkdir -p "$SAVE_DIR"
mkdir -p "$TMP_DIR"

echo "Fetching file list from Hugging Face..."

FILES=$(python3 - <<EOF
from huggingface_hub import list_repo_files
repo_id = "$DATASET_ID"
file_list = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith("all_videos_split.zip")]
print(" ".join(file_list))
EOF
)

if [ -z "$FILES" ]; then
    echo "No files found in the dataset!"
    exit 1
fi

echo "Downloading video dataset..."

python3 - <<EOF
from huggingface_hub import hf_hub_download

dataset_id = "$DATASET_ID"
save_dir = "$TMP_DIR"

files = """$FILES""".split()
for file in files:
    print(f"Downloading: {file}")
    hf_hub_download(repo_id=dataset_id, filename=file, repo_type="dataset", local_dir=save_dir)
EOF

echo "Extracting files..."
cat "$TMP_DIR"/all_videos_split.zip.* > "$TMP_DIR"/all_videos_split.zip
unzip -o "$TMP_DIR/all_videos_split.zip" -d "$SAVE_DIR"

if [ -d "$SAVE_DIR/all_videos" ]; then
    echo "Moving files from nested folder to $SAVE_DIR..."
    mv "$SAVE_DIR/all_videos"/* "$SAVE_DIR"/
    rm -rf "$SAVE_DIR/all_videos"
fi

echo "Cleaning up..."
rm -rf "$TMP_DIR"

echo "All videos downloaded and extracted to: $SAVE_DIR"