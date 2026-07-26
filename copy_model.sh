#!/bin/bash
# Copy model files to Android assets directory
# Run from the android-app/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../onnx_model"
ASSETS_DIR="$SCRIPT_DIR/app/src/main/assets"

echo "Copying model files to Android assets..."
mkdir -p "$ASSETS_DIR"

# Copy ONNX model (use quantized version for smaller size)
if [ -f "$MODEL_DIR/model_quant.onnx" ]; then
    cp "$MODEL_DIR/model_quant.onnx" "$ASSETS_DIR/"
    echo "  Copied model_quant.onnx"
elif [ -f "$MODEL_DIR/model.onnx" ]; then
    cp "$MODEL_DIR/model.onnx" "$ASSETS_DIR/"
    echo "  Copied model.onnx"
else
    echo "  ERROR: No ONNX model found in $MODEL_DIR"
    exit 1
fi

# Copy tokenizer files
if [ -f "$MODEL_DIR/tokens.json" ]; then
    cp "$MODEL_DIR/tokens.json" "$ASSETS_DIR/"
    echo "  Copied tokens.json"
else
    echo "  WARNING: tokens.json not found"
fi

# Copy config
if [ -f "$MODEL_DIR/config.yaml" ]; then
    cp "$MODEL_DIR/config.yaml" "$ASSETS_DIR/"
    echo "  Copied config.yaml"
fi

# Copy CMVN if exists
if [ -f "$MODEL_DIR/am.mvn" ]; then
    cp "$MODEL_DIR/am.mvn" "$ASSETS_DIR/"
    echo "  Copied am.mvn"
fi

echo ""
echo "Model files copied to $ASSETS_DIR"
echo "Ready to build with: ./gradlew assembleDebug"