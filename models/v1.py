import requests

def deploy_war(deploy_url, username, password, war_content):
    response = requests.put(deploy_url, auth=(username, password), data=war_content)
    return response