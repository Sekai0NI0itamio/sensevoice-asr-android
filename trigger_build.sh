#!/bin/bash
# Trigger and monitor the GitHub Actions build, then download the APK.
# Usage: ./trigger_build.sh [debug|release]
#
# Prerequisites: gh CLI authenticated with repo + workflow scopes

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-debug}"
REPO="Sekai0NI0itamio/sensevoice-asr-android"
OUTPUT_DIR="$SCRIPT_DIR/build-output"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "SenseVoice ASR - Android Build Trigger"
echo "============================================"
echo "Repository:  $REPO"
echo "Build type:  $BUILD_TYPE"
echo "Output dir:  $OUTPUT_DIR"
echo "============================================"

# Step 1: Push any uncommitted changes
echo ""
echo "[1/4] Pushing latest changes..."
cd "$SCRIPT_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "  Uncommitted changes detected. Committing..."
    git add -A
    git commit -m "Auto: trigger build"
fi
git push origin main
echo "  Push complete."

# Step 2: Trigger the workflow
echo ""
echo "[2/4] Triggering GitHub Actions workflow..."
if [ "$BUILD_TYPE" = "release" ]; then
    gh workflow run build.yml -R "$REPO" -f build_type=release
else
    gh workflow run build.yml -R "$REPO" -f build_type=debug
fi
echo "  Workflow triggered."

# Step 3: Wait for the workflow run to complete
echo ""
echo "[3/4] Waiting for workflow to start..."
sleep 5

# Get the latest run ID
RUN_ID=""
for i in $(seq 1 20); do
    RUN_ID=$(gh run list -R "$REPO" --workflow=build.yml --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || echo "")
    if [ -n "$RUN_ID" ]; then
        break
    fi
    sleep 3
done

if [ -z "$RUN_ID" ]; then
    echo "  ERROR: Could not find workflow run."
    exit 1
fi

echo "  Run ID: $RUN_ID"
echo "  View at: https://github.com/$REPO/actions/runs/$RUN_ID"
echo ""

# Wait for completion
echo "  Waiting for build to complete..."
gh run watch "$RUN_ID" -R "$REPO" --exit-status 2>&1 || true

# Check final status
CONCLUSION=$(gh run view "$RUN_ID" -R "$REPO" --json conclusion --jq '.conclusion' 2>/dev/null || echo "unknown")
echo ""
echo "  Run conclusion: $CONCLUSION"

if [ "$CONCLUSION" != "success" ]; then
    echo "  ERROR: Build failed with status: $CONCLUSION"
    echo "  Check logs at: https://github.com/$REPO/actions/runs/$RUN_ID"
    exit 1
fi

# Step 4: Download the APK artifact
echo ""
echo "[4/4] Downloading APK artifact..."
ARTIFACT_NAME="sensevoice-asr-$BUILD_TYPE"
gh run download "$RUN_ID" -R "$REPO" -n "$ARTIFACT_NAME" -D "$OUTPUT_DIR" 2>&1

echo ""
echo "============================================"
if [ -f "$OUTPUT_DIR/app-$BUILD_TYPE.apk" ] || [ -f "$OUTPUT_DIR/app-debug.apk" ]; then
    APK_FILE=$(ls "$OUTPUT_DIR"/*.apk 2>/dev/null | head -1)
    APK_SIZE=$(ls -lh "$APK_FILE" | awk '{print $5}')
    echo "BUILD SUCCESSFUL"
    echo "APK: $APK_FILE"
    echo "Size: $APK_SIZE"
    echo ""
    echo "To install on your phone:"
    echo "  adb install $APK_FILE"
    echo ""
    echo "Or transfer via:"
    echo "  - AirDrop / Nearby Share"
    echo "  - Upload to Google Drive"
    echo "  - USB cable + Android File Transfer"
else
    echo "BUILD SUCCESSFUL but APK not found at expected path."
    echo "Contents of $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR"
fi
echo "============================================"