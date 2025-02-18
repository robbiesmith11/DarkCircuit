# DarkCircuit
## 🛠️ AI Hacking Lab with Streamlit, Ollama, and Kali Linux

This project provides a **fully integrated AI-powered hacking lab** that includes:
- **A Streamlit UI** for chatting with an LLM and executing Kali Linux commands.
- **An Ollama container** that serves an LLM, allowing users to select and chat with different models.
- **A Kali Linux container** where users can run hacking tools directly from the UI.

---

## **📂 Repository Structure**
```bash
/project-root
│── docker-compose.yml # Orchestrates all containers
│── README.md # This documentation file
│── streamlit/ # Streamlit UI for LLM chat & Kali terminal
│ ├── Dockerfile # Builds the Streamlit UI container
│ ├── app.py # Main Streamlit application
│ ├── requirements.txt # Python dependencies
│── ollama/ # Ollama LLM container
│ ├── Dockerfile # Builds Ollama with model auto-pulling
│ ├── entrypoint.sh # Ensures models are pulled and server starts
│ ├── models.txt # List of models to pull from Ollama
│ ├── .gitignore # Ignores large model files
│ ├── models/ # Shared folder for storing LLM models
│── kali/ # Kali Linux container
│ ├── Dockerfile # Builds the Kali container with SSH enabled
│ ├── entrypoint.sh # Configures SSH and sets up the root password
```

---

## 🚀 Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/robbiesmith11/DarkCircuit-
cd DarkCircuit-
```

### 2. Modify `models.txt` (Optional)
Edit `ollama/models.txt` to specify the LLMs you want to pull, e.g.:
```bash
mistral
llama3
phi3:3.8b
```
> ⚠️ Must be available on [Ollama](https://ollama.com/)

### 3. Build and Run the Containers
```bash
docker-compose up --build
```
- This will build and start the Streamlit UI, Ollama, and Kali Linux.
- Ollama will automatically pull the models listed in `models.txt`.

### 4. Access the UI
Once running, open http://localhost:8501 in your browser.

## 🖥️ Features
### Streamlit UI
- Chat with an LLM – Select different models and have conversations.
- Real-time Kali Terminal – Execute commands and see live output.

### Ollama LLM
- Automatically pulls models from `models.txt` on startup.
- Allows dynamic model selection in the UI.

### Kali Linux
- Fully operational Kali environment inside Docker.
- Root access with SSH support (`root/kali`).

## 🛠️ Development & Customization
### 🔹 Modify Available LLM Models

To add or remove models, edit `ollama/models.txt` and restart the Ollama container:
```bash
docker-compose restart ollama
```

### 🔹 Debugging
Check logs for any issues:
```bash
docker-compose logs -f
```

If SSH isn’t working, manually reset the root password inside the Kali container:
```bash
docker exec -it kali bash
echo "root:kali" | chpasswd
service ssh restart
```