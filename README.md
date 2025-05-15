# DarkCircuit
## 🛠️ AI Hacking Lab with Modal Cloud Deployment and Local Dev / EXE Build

This project provides a **fully integrated AI-powered hacking lab** that includes:
- **A React frontend UI** for chatting with an LLM and executing commands in a Pwnbox instance in HackTheBox.
- **LangGraph Agent** that uses OpenAI models for the LLM portion and tools such as Run SSH commands, RAG, and DuckDuckGo search.

---

## **📂 Repository Structure**
```bash
/project-root
│
│── README.md # This documentation file and guide to Repository
│
│── build/ # Folder for completely built app
│  └── DarkCircuit/ # DarkCircuit app
│     ├── _internal/ # Internal files for build
│     └── DarkCircuit.exe # Executable
│ 
│── docs/ # RAG documents
│  
│── local-deployment/ # Implementation to run locally and build exe
│  ├── frontend/ # React frontend code and style scripts
│  │  ├── dist/ # NPM build of Frontend
│  │  ├── public/ # Files to add to frontend build
│  │  └── src/ # React frontend code
│  │
│  ├── icons/ # Exe icons
│  ├── agent_utils.py # Utilities for loading system prompts and optimising commands
│  ├── darkcircuit_agent_modular.py # LangGraph agent script
│  ├── local_app.py # Main app script
│  ├── Rag_tool.py # RAG Tool
│  ├── DarkCircuit_Linux.spec # Pyinstaller Linux build script
│  ├── DarkCircuit_Windows.spec # Pyinstaller Windows build script
│  ├── README.md # Guide for developing locally and building exe
│  └── requirements.txt # Python dependencies for local and build
│  
│── full-modal-deployment/ # Implementation to deploy in Modal cloud
│  ├── frontend/ # React frontend code and style scripts
│  │  ├── dist/ # NPM build of Frontend
│  │  ├── public/ # Files to add to frontend build
│  │  └── src/ # React frontend code
│  │
│  ├── darkcircuit_agent.py # LangGraph agent script
│  ├── darkcircuit_app.py # Main app script
│  ├── README.md # Guide for deploying on Modal
│  └── requirements.txt # Python dependencies for Modal
│  
│── media/ # App's visual style files and documentation
│  ├── Logos/ # Logo images
│  └── README.md # Visual style guide
│  
│── RAG_langchain/ # RAG experimentations
│
└── prompt_example.txt # Examples of some effective system prompts
```

---

## 🚀 Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/robbiesmith11/DarkCircuit
cd DarkCircuit
```

### 2. To Run and Deploy in the Cloud
Navigate to `/project-root/full-modal-deployment` and consult `README.md` file.

### 3. To Develop & Run Locally and/or Build Native EXE
Navigate to `/project-root/local-deployment` and consult `README.md` file.

## 🖥️ Features
### React  UI
- Chat with an LLM – Select different models and have conversations.
- Real-time Terminal – Execute commands and see live output.

### OpenAI Models
- Allows dynamic model selection and system prompt changes in the UI to chat with LLM.

### Pwnbox Instance
- Fully operational Pwnbox environment run from and connected to on HackTheBox.
