"""Install requirements for WD14-tagger."""
import os
import sys
import json
from launch import run  # pylint: disable=import-error

local_dir = os.path.dirname(os.path.realpath(__file__))

NAME = "WD14-tagger"
req_file = os.path.join(local_dir, "requirements.txt")
print(f"loading {NAME} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -q -r "{req_file}"',
    f"Checking {NAME} requirements.",
    f"Couldn't install {NAME} requirements.")
