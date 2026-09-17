#!/bin/bash
#SBATCH --job-name=rag-ollama
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --nodelist=mocgpu01

set -euo pipefail

cd /storage/home/lmeroi/chris-rag

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate chris-rag

export OLLAMA_MODELS="$HOME/tools/ollama/models"
OLLAMA_DIR="$HOME/tools/ollama"
OLLAMA_SIF="$OLLAMA_DIR/ollama.sif"

echo "Node:"
hostname

echo "Checking GPU:"
nvidia-smi || true

echo "Starting Ollama..."
singularity exec --nv \
  --bind "$OLLAMA_MODELS:$OLLAMA_MODELS" \
  "$OLLAMA_SIF" \
  ollama serve > "$OLLAMA_DIR/ollama_serve_rag.log" 2>&1 &

SERVE_PID=$!

echo "Waiting for ollama server..."
for i in {1..60}; do
  sleep 2
  if curl -s http://127.0.0.1:11434 >/dev/null 2>&1; then
    echo "ollama server is up"
    break
  fi
done

echo "Ollama GPU log:"
grep -iE "cuda|gpu|nvidia|vram|compute" "$OLLAMA_DIR/ollama_serve_rag.log" || true

echo "Running RAG..."
python -u src/rag/generate_answer_ollama.py \
  "What is the CHRIS data access policy?" \
  --top-k 3 \
  --model llama3.1:8b \
  --json-output src/evaluation/ollama_policy_output.json

echo "Stopping Ollama..."
kill $SERVE_PID || true
wait $SERVE_PID || true

echo "Done."