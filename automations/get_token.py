from hackthebox import HTBClient
from getpass import getpass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
APP_TOKEN = os.getenv('APP_TOKEN')
HTB_EMAIL = os.getenv('HTB_EMAIL')
HTB_PASSWORD = os.getenv('HTB_PASSWORD')

# Fallback to manual input if not set in .env
if not APP_TOKEN:
    print("APP_TOKEN not found in environment variables.")
    APP_TOKEN = input("Enter your APP_TOKEN: ")

if not HTB_EMAIL:
    print("HTB_EMAIL not found in environment variables.")
    HTB_EMAIL = input("Email: ")

if not HTB_PASSWORD:
    print("HTB_PASSWORD not found in environment variables.")
    HTB_PASSWORD = getpass("Password: ")

# Initialize the client with the loaded credentials
client = HTBClient(app_token=APP_TOKEN)

print("Client initialized successfully!")


