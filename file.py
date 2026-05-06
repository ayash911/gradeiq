import os
import subprocess
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line: # skip blank lines or comments
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value) # save env vars


def install_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"]) # update pip first
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"]
    ) # install requirements
    subprocess.check_call(["clear"])
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
load_dotenv() # load the secrets
install_dependencies()

import time
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokHTTPError


NGROK_TOKEN = os.getenv("NGROK_TOKEN")
PORT = os.getenv("PORT")
ngrok.set_auth_token(NGROK_TOKEN) # setup ngrok token

ngrok.kill() # kill old instances
for tunnel in ngrok.get_tunnels(): # loop through tunnels
    ngrok.disconnect(tunnel.public_url) # disconnect them

from app import app # get flask app
try:
    public_url = ngrok.connect(addr=f"http://127.0.0.1:{PORT}") # start tunnel
except PyngrokNgrokHTTPError:
    print("[ngrok] Stale endpoint detected. Waiting 5s for cleanup...")
    ngrok.kill()
    time.sleep(5)
    public_url = ngrok.connect(addr=f"http://127.0.0.1:{PORT}")

print(f"\n SUCCESS! OPEN THIS LINK: {public_url}") # hooray we are live!
app.run(host="0.0.0.0", port=PORT) # run the server!
