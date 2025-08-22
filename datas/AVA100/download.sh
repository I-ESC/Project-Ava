#!/usr/bin/env bash
set -euo pipefail

REPO_ID="iesc/Ava-100"
VIDEOS_DIR="videos"

ZIPS=("citytour1.zip" "citytour2.zip" "ego1.zip" "ego2.zip" "traffic1.zip" "traffic2.zip" "wildlife1.zip" "wildlife2.zip")
JSONS=("citytour.json" "ego.json" "traffic.json" "wildlife.json")
MP4S=("citytour1.mp4" "citytour2.mp4" "ego1.mp4" "ego2.mp4" "traffic1.mp4" "traffic2.mp4" "wildlife1.mp4" "wildlife2.mp4")

command -v huggingface-cli >/dev/null || { echo "huggingface-cli not found. Install: pip install -U huggingface_hub"; exit 1; }
command -v unzip >/dev/null || { echo "unzip not found."; exit 1; }

mkdir -p "$VIDEOS_DIR"

download_one () {
  local f="$1"
  echo "Downloading $f"
  huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir . --include "$f" >/dev/null
}

for z in "${ZIPS[@]}"; do
  download_one "$z"
  echo "Extracting $z"
  unzip -j -o "$z" "*.mp4" -d "$VIDEOS_DIR" >/dev/null
done

for j in "${JSONS[@]}"; do
  download_one "$j"
done

echo "Normalizing video filenames"
for i in "${!ZIPS[@]}"; do
  base="${ZIPS[$i]%.zip}"
  want="${MP4S[$i]}"
  if [ ! -f "$VIDEOS_DIR/$want" ]; then
    cand=$(find "$VIDEOS_DIR" -maxdepth 1 -type f -name "${base}*.mp4" -o -name "*${base}*.mp4" | head -n 1 || true)
    [ -n "${cand:-}" ] && mv -f "$cand" "$VIDEOS_DIR/$want"
  fi
done

echo "Removing non-mp4 files in videos"
find "$VIDEOS_DIR" -type f ! -name "*.mp4" -delete
find "$VIDEOS_DIR" -type d -empty -not -path "$VIDEOS_DIR" -delete

echo "Keeping only the 8 expected mp4s"
for f in "$VIDEOS_DIR"/*.mp4; do
  [ -e "$f" ] || break
  name="$(basename "$f")"
  keep=0
  for w in "${MP4S[@]}"; do
    [ "$name" = "$w" ] && keep=1 && break
  done
  [ $keep -eq 0 ] && rm -f "$f"
done

echo "Cleaning downloaded zips"
rm -f "${ZIPS[@]}"

echo "Cleaning stray files in current folder"
videos_base="$(basename "$VIDEOS_DIR")"
for f in * .*; do
  [ "$f" = "." ] && continue
  [ "$f" = ".." ] && continue
  [ "$f" = "download.sh" ] && continue
  [ "$f" = "$videos_base" ] && continue
  keep=0
  for j in "${JSONS[@]}"; do
    [ "$f" = "$j" ] && keep=1 && break
  done
  [ $keep -eq 0 ] && rm -rf "$f"
done

echo "Verifying final layout"
missing=0
for w in "${MP4S[@]}"; do
  [ -f "$VIDEOS_DIR/$w" ] || { echo "Missing: $VIDEOS_DIR/$w"; missing=1; }
done
for j in "${JSONS[@]}"; do
  [ -f "$j" ] || { echo "Missing: $j"; missing=1; }
done
[ $missing -eq 1 ] && echo "Warning: Some expected files are missing." || echo "Success: videos/ has 8 mp4s; root has 4 jsons + download.sh."