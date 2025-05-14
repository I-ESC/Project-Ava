#!/bin/bash

BASE_URL="https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/"

# Directory to save downloaded files
DOWNLOAD_DIR="./"
mkdir -p "$DOWNLOAD_DIR"

# List of files to download (from 01 to 20)
for i in $(seq -w 01 20); do
    FILE_NAME="videos_chunked_${i}.zip"
    FILE_URL="${BASE_URL}${FILE_NAME}?download=true"
    FILE_PATH="${DOWNLOAD_DIR}${FILE_NAME}"

    # Download the file using wget
    echo "Downloading ${FILE_NAME}..."
    wget -O "$FILE_PATH" "$FILE_URL"

    # Check if the file was downloaded successfully
    if [ -f "$FILE_PATH" ]; then
        echo "Extracting ${FILE_NAME}..."
        unzip -q "$FILE_PATH" -d "$DOWNLOAD_DIR"
        echo "Deleting ${FILE_NAME}..."
        rm "$FILE_PATH"
    else
        echo "Failed to download ${FILE_NAME}. Skipping."
    fi
done

mv "./data" "./videos"

echo "All files downloaded, extracted, and original zip files deleted."
