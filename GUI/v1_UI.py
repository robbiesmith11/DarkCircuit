import streamlit as st
import requests

import importlib.util
import sys

# Define the absolute path to the module
module_path = r"C:\Users\smith\Github\DarkCircuit-\models\v1.py"

# Load the module
spec = importlib.util.spec_from_file_location("v1", module_path)
v1 = importlib.util.module_from_spec(spec)
sys.modules["v1"] = v1
spec.loader.exec_module(v1)

# Now you can use the deploy_war function
get_flag = v1.get_flag

st.title("Flag Command - Automated Exploit")

host = st.text_input("Host IP provided", value="94.237.54.190:44038")

# User clicks a button to start the exploit
if st.button("Get Flag"):
    full_host = f"http://{host}"
    flag = get_flag(full_host)  # Calls the hardcoded function from models/v1.py
    if flag:
        st.success("Flag Retrieved Successfully!")
        st.text_area("Flag Output:", flag, height=100)
    else:
        st.error("Failed to retrieve flag.")