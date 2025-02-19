# /root/get_machines.py
from hackthebox import HTBClient
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get credentials from environment variables
APP_TOKEN = os.getenv('APP_TOKEN')

# Initialize the HTBClient
client = HTBClient(app_token=APP_TOKEN)

# Get all machines
machines = client.get_machines()

# Display the machine names and IPs
print("\nAvailable Machines:")
for machine in machines:
    print(f"Name: {machine['name']}, IP: {machine['ip']}")
