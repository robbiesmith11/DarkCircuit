# Deploying DarkCircuit on Modal

This guide explains the steps to deploy and configure the DarkCircuit app on Modal.

## Prerequisites
- Modal account ([https://modal.com](https://modal.com))

> [!TIP]
> Enter card payment details to receive $30/month (USD) of free credits.


## Step-by-Step Deployment & Setup

### Step 1: Navigate to the `full-modal-deployment` directory
In your terminal, navigate to the `/full-modal-deployment` directory of this repository.
```bash
cd full-modal-deployment/
```

### Step 2: Install Requirements & Setup Modal Environment
Make sure your Modal environment is correctly configured. Install Modal CLI if you haven't:
```bash
pip install -r requirements.txt
modal setup
```

### Step 3: Install Frontend Dependencies
If not having js installed, install npm/node.js from https://nodejs.org/en -- > https://github.com/coreybutler/nvm-windows/releases
```bash
nvm install lts
```
Then use the latest version printed from the previous command:
```bash
nvm use <latest-version>
```

Install npm dependencies for frontend:
```bash
cd frontend && npm install && cd ..
```

### Step 4: Deploy Ollama Server (Optional)
> [!CAUTION]
> App currently isn't using the Ollama server so this step isn't necessary at the moment. However, if this step is skipped you will see errors relating to the Ollama server which you can ignore. Eventually we may get the agent using the Ollama server in which case this would be a required step.

Run the following command from your terminal to deploy your Ollama server to Modal:
```bash
modal deploy ollama_server.py
```

> [!NOTE]
> You can set your GPU configuration in `ollama_server.py` by changing `GPU = "T4"`(cheapest GPU) to something faster such as `GPU = "A10G"`(will be a little bit more expensive).

### Step 5: Create Environment File
1. In the top left corner of your Modal dashboard, find your workspace name.

![In this example the workspace name is `brandontoews`.](modal_workspace.png)

2. Insert name where it is indicated on each line in the command below:
```bash
echo 'VITE_BACKEND_API_URL=https://<replace-with-workspace-name>--darkcircuit-app-serve.modal.run' > frontend/.env
```
```bash
echo 'VITE_TERMINAL_WS_URL=wss://<replace-with-workspace-name>--darkcircuit-app-serve.modal.run' >> frontend/.env
```

### Step 6: Create OpenAI API Secret on Modal
> [!CAUTION]
> App can run without performing this step but the user won't be able to use the LLM side.
1. In the top left corner of your Modal dashboard select `Secrets`.
2. Click `Create new secret` button.
3. Under `Choose Type` select `OpenAI`.
![Create Modal OpenAI API Secret](modal_openai_secret.png)
4. Paste OpenAI API key in the `Value` field and click `Done` button.


### Step 7: Build Frontend
Run the following command to build React frontend:
```bash
cd frontend && npm run build && cd ..
```

### Step 8: Deploy DarkCircuit App
Run the following command from your terminal to deploy your DarkCircuit app to Modal:
```bash
modal deploy darkcircuit_app.py
```

### Step 9: Connect to HackTheBox
![Click on connect using Pwnbox.](starting_point.png)

1. Go to [HackTheBox](https://app.hackthebox.com/starting-point) platform, navigate to `Starting Point`, and select a challenge.

![Select `VIEW INSTANCE DETAILS` dropdown menu.](ssh_details.png)

2. Click on `Connect using Pwnbox`, then click `START PWNBOX`, and select dropdown menu `VIEW INSTANCE DETAILS`.
> [!WARNING]
> The HTB Free Plan only provides 2 hrs of Pwnbox usage so be diligent about terminating the instance when you are finished using it.

> [!TIP]
> You can receive **24 hours per month** of Pwnbox usage with a [VIP subscription](https://app.hackthebox.com/vip) or **Unlimited hours** of Pwnbox usage with a [VIP+ subscription](https://app.hackthebox.com/vip). **Make sure to click the billed monthly toggle or you will be charged for annual billing!**

![Confirm `INSTANCE LIFETIME` has started.](pwnbox_started.png)

3. Once you see that the `INSTANCE LIFETIME` has started, copy and paste details from the Pwnbox instance into terminal connection window of the DarkCircuit app and click `Connect`.

![Paste Pwnbox details into SSH form on DarkCircuit.](connect_pwnbox.png)

> [!CAUTION]
> SSH Terminal will not connect until Pwnbox's `INSTANCE LIFETIME` clock has started.


## Future Development
At the moment, the Ollama LLMs do not have RAG or are able to interact directly with the Pwnbox instance. The [Coding Agent example](https://modal.com/docs/examples/agent) on Modal's documentation may be a good starting point for integrating these concepts into the current implementation. Instead of the sandbox container, DarkCircuit has a connection with a Pwnbox instance. As such, the `run_ssh_command()` method in `darkcircuit_app.py` could be used instead of the `run()` method defined in `agent.py` of Modal's [repo](https://github.com/modal-labs/modal-examples/tree/main/13_sandboxes/codelangchain) by the agent to run code. Any changes made to incorporate these things should probably be made to `darkcircuit_app.py`. To re-deploy after changes just delete the `DarkCircuit` app using the Modal dashboard and re-run the command in [Step 7](#step-7-deploy-darkcircuit-app).