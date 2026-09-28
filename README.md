# CHRIS-RAG README FILE


## 1. Configure PuTTY once
Open your normal MOC PuTTY session and go to Connection --> SSH --> Tunnels. Add both local tunnels:
* Source port 8000 --> Destination mocgpu01:8000 --> Local --> Add
* Source port 8501 --> Destination mocgpu01:8501 --> Local --> Add

Return to Session, save locally with any name, for example mocshell-chris-rag, close the current connection if necessary, and reconnect. The tunnels only become active in the newly connected session.


## 2. Open CHRIS RAG
With PuTTY connected, open this address in your Windows browser:
http://127.0.0.1:8501


## 3. OPTIONAL connection check
In Windows PowerShell, you can verify that the local tunnel is listening:

netstat -ano | findstr :8501

netstat -ano | findstr :8000

If 8501 is listening locally but the page does not open, contact the CHRIS RAG administrator.


## 4. Use the application
Document example: How many measurements are used for the mean systolic blood pressure variable?

**GWAS example**: Show genome-wide significant variants for peppermint in females using GRCh38

**Document answers** include evidence labels such as [T1] for text evidence and [B1] for table/codebook evidence. Detailed source information is shown in the Sources section of the UI.


## 5. If the service is unavailable
Do not start your own Ollama, Uvicorn, Streamlit, or Slurm job. The administrator manages the single shared service. Report the problem to the administrator, who can check the Slurm job and shared logs.