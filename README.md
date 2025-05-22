# DarkCircuit 🔒⚡

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/react-19.1.0-blue.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/typescript-5.8.3-blue.svg)](https://typescriptlang.org)
[![LangChain](https://img.shields.io/badge/langchain-0.3.25-green.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🛠️ AI-Powered Hacking Lab with Cloud Deployment and Local Development

**DarkCircuit** is a comprehensive AI-powered cybersecurity laboratory that bridges the gap between artificial intelligence and hands-on penetration testing. It provides an integrated environment for learning and practicing ethical hacking with real-time AI assistance.

### 🎯 Key Features

- **🤖 AI Security Assistant**: LangGraph-powered agent using OpenAI models for intelligent security analysis
- **🖥️ Real-time Terminal Interface**: Live SSH connection to HackTheBox Pwnbox instances
- **📚 RAG-Enhanced Knowledge**: Vector-based document retrieval from security writeups and guides
- **🌐 Dual Deployment Options**: Local executable or Modal cloud deployment
- **⚡ WebSocket Communication**: Real-time bidirectional terminal interaction
- **🎨 Modern React UI**: TypeScript-based frontend with responsive design

---

## 🏗️ Architecture Overview

DarkCircuit employs a modern, scalable architecture:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   React Frontend    │    │   FastAPI Backend    │    │   HackTheBox        │
│                     │    │                      │    │   Pwnbox Instance   │
│ ├─ Chat Interface   │◄──►│ ├─ LangGraph Agent   │◄──►│                     │
│ ├─ XTerminal        │    │ ├─ SSH Manager       │    │ ├─ Target Machines  │
│ ├─ Config Panel     │    │ ├─ RAG System        │    │ ├─ Security Tools   │
│ └─ Debug Panel      │    │ └─ WebSocket Handler │    │ └─ Challenge Env    │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### 🧠 AI Agent Workflow (LangGraph)

```
User Query → Reasoner → Tools → Responder → Streaming Response
              │         │       │
              │         │       └─ Formats final answer
              │         │
              │         ├─ run_command (SSH execution)
              │         ├─ rag_retrieve (Document search)
              │         └─ web_search (DDG search)
              │
              └─ Plans approach and selects tools
```

---

## 🚀 Technology Stack

### Backend Technologies
- **Python 3.8+**: Core runtime environment
- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain/LangGraph**: AI agent orchestration and workflow management
- **OpenAI API**: Large language model integration (GPT-4o-mini, GPT-4o)
- **Paramiko**: SSH client library for secure remote connections
- **FAISS**: Vector database for efficient similarity search
- **FastEmbed**: Lightweight embedding generation
- **WebSockets**: Real-time bidirectional communication
- **Uvicorn**: ASGI server for production deployment

### Frontend Technologies
- **React 19.1.0**: Modern UI library with hooks and concurrent features
- **TypeScript 5.8.3**: Type-safe JavaScript development
- **Vite 6.3.5**: Fast build tool and development server
- **Tailwind CSS 4.1.7**: Utility-first CSS framework
- **XTerm.js**: Terminal emulator for web browsers
- **React Markdown**: Markdown rendering with syntax highlighting
- **Lucide React**: Modern icon library

### AI and ML Stack
- **LangGraph**: State-based agent workflow orchestration
- **OpenAI Models**: GPT-4o-mini (default), GPT-4o (advanced)
- **RAG Pipeline**: PDF document processing and vector retrieval
- **FastEmbed**: Efficient embedding generation for similarity search
- **FAISS**: Facebook AI Similarity Search for vector operations

### Infrastructure and Deployment
- **Modal**: Serverless cloud deployment platform
- **PyInstaller**: Native executable creation for local deployment
- **Docker**: Containerization support (via Modal)
- **WebSocket Protocol**: Real-time terminal communication
- **SSH Protocol**: Secure remote shell access

---

## 📂 Repository Structure

```bash
DarkCircuit/
├── README.md                           # 📋 Main project documentation                        
├── TEST_PLANS_AND_CASES.md           # 🧪 Comprehensive testing documentation
├── LICENSE                           # ⚖️ Project license
│
├── build/                            # 📦 Pre-built executable distribution
│   └── DarkCircuit/                  # Complete application package
│       ├── _internal/                # PyInstaller internal files
│       └── DarkCircuit.exe          # Windows executable
│
├── docs/                             # 📚 RAG knowledge base
│   └── *.pdf                        # HackTheBox writeups and security guides
│
├── local-deployment/                 # 🏠 Local development and executable build
│   ├── frontend/                     # React application
│   │   ├── src/                      # TypeScript source code
│   │   │   ├── components/           # React components
│   │   │   │   ├── ChatInterface.tsx # AI chat interface
│   │   │   │   ├── XTerminal.tsx     # Terminal emulator
│   │   │   │   ├── SSHConnectForm.tsx# SSH connection form
│   │   │   │   └── Sidebar.tsx       # Configuration sidebar
│   │   │   ├── types.ts              # TypeScript type definitions
│   │   │   └── App.tsx               # Main application component
│   │   ├── public/                   # Static assets
│   │   │   └── prompts.json          # Default AI prompts
│   │   ├── dist/                     # Built frontend assets
│   │   └── package.json              # Node.js dependencies
│   │
│   ├── local_app.py                  # 🚀 Main FastAPI application
│   ├── darkcircuit_agent_modular.py  # 🧠 LangGraph AI agent
│   ├── agent_tools.py                # 🔧 Agent tool implementations
│   ├── agent_utils.py                # 🛠️ Utility functions
│   ├── Rag_tool.py                   # 📖 RAG system implementation
│   ├── streaming_handler.py          # 📡 Real-time response streaming
│   ├── utils.py                      # 🔨 General utilities
│   ├── requirements.txt              # 📋 Python dependencies
│   ├── DarkCircuit_Windows.spec      # 🪟 Windows build configuration
│   ├── DarkCircuit_Linux.spec        # 🐧 Linux build configuration
│   └── README.md                     # 📖 Local deployment guide
│
├── full-modal-deployment/            # ☁️ Modal cloud deployment
│   ├── frontend/                     # React application (cloud version)
│   ├── darkcircuit_app.py            # 🌐 Modal application entry point
│   ├── darkcircuit_agent.py          # 🤖 Cloud-optimized AI agent
│   ├── Rag_tool.py                   # 📚 RAG implementation
│   ├── requirements.txt              # 📋 Cloud dependencies
│   └── README.md                     # ☁️ Cloud deployment guide
│
└── media/                            # 🎨 Visual assets and branding
    ├── Logos/                        # Brand logos and icons
    └── README.md                     # Visual style guide
```

---

## 🚀 Quick Start Guide

### Option 1: Use Pre-built Executable (Fastest)

1. **Download the latest release** from the `build/` directory
2. **Extract** the DarkCircuit folder
3. **Create** a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. **Run** `DarkCircuit.exe` (Windows) or the Linux equivalent
5. **Open** your browser to `http://127.0.0.1:8000`

### Option 2: Local Development Setup

#### Prerequisites
- **Python 3.8-3.12** (strict requirement)
- **Node.js 16+** (recommended for compatibility)
- **OpenAI API Key** with sufficient credits
- **Git** for version control

#### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/robbiesmith11/DarkCircuit.git
   cd DarkCircuit/local-deployment
   ```

2. **Set up Python environment**:
   ```bash
   # Create virtual environment (recommended)
   python -m venv darkcircuit-env
   source darkcircuit-env/bin/activate  # Linux/Mac
   # darkcircuit-env\Scripts\activate   # Windows
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Set up frontend**:
   ```bash
   cd frontend
   npm install --force --legacy-peer-deps
   npm run build
   cd ..
   ```

4. **Configure environment**:
   ```bash
   echo 'OPENAI_API_KEY=your_api_key_here' > .env
   ```

5. **Run the application**:
   ```bash
   python local_app.py
   ```

6. **Access the application** at `http://127.0.0.1:8000`

### Option 3: Modal Cloud Deployment

1. **Install Modal CLI**:
   ```bash
   pip install modal
   modal config
   ```

2. **Set up secrets**:
   ```bash
   modal secret create openai-api-key OPENAI_API_KEY=your_api_key_here
   ```

3. **Deploy to cloud**:
   ```bash
   cd full-modal-deployment
   modal deploy darkcircuit_app.py
   ```

---

## 🔧 Usage Guide

### 1. Connect to HackTheBox Pwnbox

1. **Navigate** to [HackTheBox Starting Point](https://app.hackthebox.com/starting-point)
2. **Select** a challenge and click "Connect using Pwnbox"
3. **Start** your Pwnbox instance
4. **Copy** the SSH credentials from "VIEW INSTANCE DETAILS"
5. **Paste** credentials into DarkCircuit's SSH connection form

### 2. Configure Target Machine

1. **Spawn** the target machine from HackTheBox
2. **Copy** the target IP address
3. **Enter** the IP in DarkCircuit's "Target Configuration" sidebar
4. **Select** the appropriate challenge (optional)

### 3. Interact with the AI Agent

- **Ask questions** about the target system
- **Request** specific security assessments
- **Get** step-by-step guidance for challenges
- **Monitor** real-time command execution

Example queries:
```
"Scan the target for open ports"
"Check for common vulnerabilities"
"Help me find the user flag"
"Explain the last command output"
```

---

## 🧪 Development Workflow

### Frontend Development
```bash
cd local-deployment/frontend
npm run dev          # Start development server (http://localhost:5173)
npm run build        # Build for production
```

### Backend Development
```bash
cd local-deployment
python local_app.py  # Start backend server (http://127.0.0.1:8000)
```

### Building Executables
```bash
# Windows
pyinstaller DarkCircuit_Windows.spec

# Linux/Mac
pyinstaller DarkCircuit_Linux.spec
```

### Testing
```bash
# Run test suite (when implemented)
pytest tests/

# Manual testing using TEST_PLANS_AND_CASES.md
```

---

## 🔒 Security Considerations

- **API Keys**: Store OpenAI API keys securely in environment variables
- **SSH Connections**: All connections use secure SSH protocol with proper authentication
- **Command Execution**: Commands are executed in isolated Pwnbox environments
- **Data Privacy**: No sensitive data is stored locally; all processing is ephemeral
- **Access Control**: Application requires explicit SSH credentials for remote access

---

## 🤝 Contributing

### Development Guidelines

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Follow** the existing code style and documentation standards
4. **Add** comprehensive tests for new functionality
5. **Update** documentation for any API changes
6. **Commit** changes: `git commit -m 'Add amazing feature'`
7. **Push** to the branch: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

### Code Style
- **Python**: Follow PEP 8 guidelines
- **TypeScript**: Use ESLint and Prettier configurations
- **Comments**: Comprehensive docstrings and inline comments
- **Testing**: Write tests for all new features

### Git Workflow
- **Main Branch**: Stable, production-ready code
- **Feature Branches**: Individual features and bug fixes
- **Pull Requests**: Required for all changes with code review
- **Semantic Commits**: Use conventional commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **OpenAI API**: Subject to OpenAI's Terms of Service
- **HackTheBox**: Usage subject to HackTheBox Terms and Conditions
- **Dependencies**: Various open-source licenses (see requirements.txt)

---

## 🆘 Support and Troubleshooting

### Common Issues

**SSH Connection Failures**:
- Verify Pwnbox is active and credentials are correct
- Check network connectivity and firewall settings
- Ensure SSH port (22) is accessible

**Frontend Build Errors**:
- Use Node.js v16 for maximum compatibility
- Run `npm install --force --legacy-peer-deps`
- Clear node_modules and package-lock.json if needed

**Agent Not Responding**:
- Verify OpenAI API key is valid and has credits
- Check that target IP is configured in sidebar
- Monitor debug panel for detailed error information

### Getting Help

- **Documentation**: Comprehensive guides in `/docs` and component READMEs
- **Test Cases**: Detailed testing scenarios in `TEST_PLANS_AND_CASES.md`
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Feature requests and general questions via GitHub Discussions

---

## 🎯 Roadmap

### Current Version (v1.0)
- ✅ Local deployment with executable builds
- ✅ Modal cloud deployment
- ✅ LangGraph AI agent with RAG
- ✅ Real-time SSH terminal interface
- ✅ HackTheBox integration

### Upcoming Features (v1.1)
- 🔄 Enhanced vulnerability detection
- 🔄 Multiple target management
- 🔄 Session persistence and replay
- 🔄 Advanced reporting and documentation
- 🔄 Plugin system for custom tools

### Future Considerations (v2.0)
- 🔮 Multi-user collaborative features
- 🔮 Custom lab environment creation
- 🔮 Advanced AI model fine-tuning
- 🔮 Integration with additional platforms
- 🔮 Mobile application support

---

## 📊 Project Statistics

- **Languages**: Python (60%), TypeScript (35%), CSS (5%)
- **Lines of Code**: ~15,000+ lines
- **Components**: 20+ React components
- **Dependencies**: 30+ Python packages, 25+ npm packages
- **Deployment Targets**: Windows, Linux, Modal Cloud
- **Testing Coverage**: Comprehensive test plans with 20+ test cases

---

*Built with ❤️ for the cybersecurity community by security researchers, for security researchers.*

**⚠️ Disclaimer**: DarkCircuit is designed for educational and authorized security testing purposes only. Users are responsible for ensuring compliance with applicable laws and regulations. The developers assume no liability for misuse of this tool.
