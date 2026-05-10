#!/bin/bash

# Configuration
REPO="ibm-granite/granite-speech-4.1-2b"
BASE_URL="https://huggingface.co/$REPO/resolve/main"
TARGET_DIR="./models/granite-2b"
TOKEN="${HF_TOKEN}"

# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# List of files to download
FILES=(
    ".gitattributes"
    "README.md"
    "added_tokens.json"
    "chat_template.jinja"
    "config.json"
    "merges.txt"
    "model-00001-of-00003.safetensors"
    "model-00002-of-00003.safetensors"
    "model-00003-of-00003.safetensors"
    "model.safetensors.index.json"
    "model.sig"
    "multilingual_sample.wav"
    "out_llm.safetensors"
    "preprocessor_config.json"
    "processor_config.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    ".eval_results/open_asr_leaderboard.yaml"
)

echo "🚀 Starting High-Speed Turbo Download for $REPO"
echo "📂 Target Directory: $TARGET_DIR"

if [ -z "$TOKEN" ]; then
    echo "⚠️  HF_TOKEN not detected in environment. Gated files may fail."
else
    echo "🔑 Using HF_TOKEN for authentication."
fi

# Check if aria2c is installed for maximum speed, otherwise fallback to parallel wget
if command -v aria2c >/dev/null 2>&1; then
    echo "⚡ Using aria2c (Multi-connection enabled)"
    
    # Create temporary input file for aria2c
    INPUT_FILE=$(mktemp)
    for FILE in "${FILES[@]}"; do
        echo "$BASE_URL/$FILE" >> "$INPUT_FILE"
        echo "  out=$FILE" >> "$INPUT_FILE"
    done

    HEADER_ARG=""
    if [ -n "$TOKEN" ]; then
        HEADER_ARG="--header=Authorization: Bearer $TOKEN"
    fi

    # -x 16: 16 connections per server
    # -s 16: 16 connections per file
    # -j 5: 5 concurrent files
    # -c: Continue partial downloads
    aria2c -i "$INPUT_FILE" \
        -d "$TARGET_DIR" \
        -x 16 \
        -s 16 \
        -j 5 \
        -c \
        $HEADER_ARG \
        --summary-interval=10 \
        --console-log-level=warn \
        --download-result=hide

    rm "$INPUT_FILE"
else
    echo "⚙️  aria2c not found. Falling back to parallel wget..."
    
    download_file() {
        local file="$1"
        local target_dir="$2"
        local token="$3"
        local base_url="$4"
        
        local dir_name=$(dirname "$file")
        mkdir -p "$target_dir/$dir_name"
        
        local auth_header=""
        if [ -n "$token" ]; then
            auth_header="--header=Authorization: Bearer $token"
        fi
        
        wget -c $auth_header "$base_url/$file" -O "$target_dir/$file" --quiet --show-progress --tries=5
    }
    export -f download_file
    
    printf "%s\n" "${FILES[@]}" | xargs -I {} -P 4 bash -c 'download_file "{}" "'"$TARGET_DIR"'" "'"$TOKEN"'" "'"$BASE_URL"'"'
fi

echo "✅ Download finished! Model files are in $TARGET_DIR"
