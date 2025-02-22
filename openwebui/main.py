from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import paramiko
import subprocess
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

OLLAMA_API = os.getenv("OLLAMA_API", "http://ollama:11434")
KALI_SSH_HOST = os.getenv("KALI_SSH_HOST", "kali")
KALI_SSH_PORT = int(os.getenv("KALI_SSH_PORT", "22"))
KALI_SSH_USER = os.getenv("KALI_SSH_USER", "root")
KALI_SSH_PASS = os.getenv("KALI_SSH_PASS", "kali")
VPN_FILE_PATH = "/tmp/vpn.ovpn"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/connect_vpn")
async def connect_vpn(vpn_file: UploadFile = File(...)):
    try:
        contents = await vpn_file.read()
        with open(VPN_FILE_PATH, "wb") as f:
            f.write(contents)

        # Use docker exec to connect VPN in the 'kali' container
        connect_cmd = f"docker exec kali openvpn --config {VPN_FILE_PATH}"
        process = subprocess.Popen(connect_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(timeout=30)
        if error:
            return JSONResponse(status_code=400, content={"error": error.decode("utf-8")})
        return {"message": "VPN connected successfully!", "output": output.decode("utf-8")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get_models")
def get_models():
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags")
        models = [model["name"] for model in response.json().get("models", [])]
        return {"models": models}
    except Exception as e:
        return {"error": str(e), "models": []}

@app.post("/chat")
def chat(model: str = Form(...), prompt: str = Form(...)):
    try:
        response = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        data = response.json()
        ai_reply = data.get("response", "")
        return {"response": ai_reply}
    except Exception as e:
        return {"error": str(e)}

@app.post("/execute_cmd")
def execute_cmd(cmd: str = Form(...)):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(KALI_SSH_HOST, port=KALI_SSH_PORT, username=KALI_SSH_USER, password=KALI_SSH_PASS)

        stdin, stdout, stderr = ssh.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")
        ssh.close()
        return {"output": output, "error": error, "exit_status": exit_status}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
