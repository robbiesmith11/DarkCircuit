from hackthebox import HTBClient
from dotenv import load_dotenv
import os
import requests
import subprocess

# Load environment variables
load_dotenv()
APP_TOKEN = os.getenv('APP_TOKEN')

# Initialize the HTBClient
client = HTBClient(app_token=APP_TOKEN)

# Get the current VPN server
vpn_server = client.get_current_vpn_server()

# Debugging: Check what we're getting
print(f"DEBUG: VPN Server Name -> {vpn_server.friendly_name}")
print(f"DEBUG: VPN Server ID -> {vpn_server.id}")

# Fetch the actual VPN file content
vpn_file_url = f"https://www.hackthebox.com/api/v4/vpn/download/{vpn_server.id}"

# Use requests to fetch the actual VPN config
headers = {"Authorization": f"Bearer {APP_TOKEN}"}
response = requests.get(vpn_file_url, headers=headers)

if response.status_code == 200:
    vpn_config = response.text  # Get the actual file content
else:
    raise ValueError(f"Error fetching VPN file: {response.status_code} - {response.text}")

# Save VPN configuration to file
vpn_file_path = "/root/htb_vpn.ovpn"
with open(vpn_file_path, "w") as file:
    file.write(vpn_config)

print(f"[*] VPN file successfully saved at {vpn_file_path}")

# Connect using OpenVPN
print("[*] Connecting to HTB VPN...")
subprocess.run(["openvpn", "--config", vpn_file_path])


