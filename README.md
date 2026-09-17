1. Create your PuTTY session (first use only)
PuTTY saved sessions are local to your Windows computer. Start from your normal MOC connection and add: (Connection ? SSH ?Tunnels)
•	Local tunnel 8000 ? Source port: 8000; Destination: mocgpu01:8000
•	Local tunnel 8501 ? Source port: 8501; Destination: mocgpu01:8501
Add these two tunnels and save the session locally with any name, for example mocshell-chris-rag, then connect.

2. Start the prototype runtime
For the current prototype, each authorized user needs access to the prepared private project/runtime. Python requirements do not need to be installed manually when using the provided Singularity image.
srun -w mocgpu01 -p gpu --pty --mem=32G --cpus-per-task=4 --time=04:00:00 bash
cd ~/chris-rag
export CHRIS_DATA_ROOT=/storage/data/chris-rag
export CHRIS_GWAS_ROOT=$HOME/chris-rag/data/processed/gwas
export CHRIS_API_URL=http://127.0.0.1:8000/query
export CHRIS_LLM_MODEL=llama3.1:8b
export OLLAMA_HOST=http://127.0.0.1:11434
nohup ollama serve > logs/ollama.log 2>&1 &
singularity exec --bind "$CHRIS_DATA_ROOT:$CHRIS_DATA_ROOT:ro" --bind "$CHRIS_GWAS_ROOT:$CHRIS_GWAS_ROOT:ro" --bind "$HOME/chris-rag/logs:/app/logs" "$HOME/containers/chris-rag.sif" uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > logs/uvicorn.log 2>&1 &
singularity exec "$HOME/containers/chris-rag.sif" streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 8501 > logs/streamlit.log 2>&1 &

3. Open and use CHRIS RAG
Open http://127.0.0.1:8501 in your Windows browser.
Document example: How many measurements are used for the mean systolic blood pressure variable?
Citations [Tn] identify text evidence and [Bn] table/codebook evidence; expand Sources to inspect them.
GWAS example: Show genome-wide significant variants for peppermint in females using GRCh38

4. Basic checks
•	Ollama: curl http://127.0.0.1:11434/api/tags
•	FastAPI: curl http://127.0.0.1:8000/health
•	If the Slurm job expires, start a new GPU allocation and restart the three services.
•	Do not upload CHRIS-sensitive data to external services or public repositories.
