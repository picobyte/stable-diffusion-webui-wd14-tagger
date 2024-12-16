"""Install requirements for WD14-tagger."""
import os
import sys
import subprocess

from launch import run  # pylint: disable=import-error

NAME = "WD14-tagger"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "requirements.txt")
print(f"loading {NAME} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -q -r "{req_file}"',
    f"Checking {NAME} requirements.",
    f"Couldn't install {NAME} requirements.")

def check_rocm_version():
    try:
        if subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).returncode != 0:
            print("ROCm is not installed.")
            return None
        result = subprocess.run(['hipcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.split('\n'):
            if 'HIP version:' in line:
                version = line.split()[-1]
                return version
    except FileNotFoundError:
        print("ROCm is not installed.")
        return None

def install_tensorflow_rocm(version):
    if version.startswith('6.3.'):
        tf_version = '2.17.0'
        repo_url = 'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.17.0-cp310-cp310-manylinux_2_28_x86_64.whl'
        run(f'"{sys.executable}" -m pip install {repo_url}',
            f"Installing tensorflow-rocm {tf_version} for ROCm {version} from {repo_url}.",
            f"Couldn't install tensorflow-rocm {tf_version}.")
    elif version.startswith('6.2.'):
        tf_version = '2.16.1'
        repo_url = 'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/tensorflow_rocm-2.16.1-cp310-cp310-manylinux_2_28_x86_64.whl'
        run(f'"{sys.executable}" -m pip install {repo_url}',
            f"Installing tensorflow-rocm {tf_version} for ROCm {version} from {repo_url}.",
            f"Couldn't install tensorflow-rocm {tf_version}.")
    elif version.startswith('6.1.'):
        tf_version = '2.15.0'
        repo_url = 'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/tensorflow_rocm-2.15.0-cp310-cp310-manylinux2014_x86_64.whl'
        run(f'"{sys.executable}" -m pip install {repo_url}',
            f"Installing tensorflow-rocm {tf_version} for ROCm {version} from {repo_url}.",
            f"Couldn't install tensorflow-rocm {tf_version}.")
    elif version.startswith('6.0.'):
        tf_version = '2.14.0'
        run(f'"{sys.executable}" -m pip install tensorflow-rocm=={tf_version}',
            f"Installing tensorflow-rocm {tf_version} for ROCm {version}.",
            f"Couldn't install tensorflow-rocm {tf_version}.")
    else:
        print(f"Unsupported ROCm version: {version}. Please upgrade to ROCm 6.0.x, 6.1.x, 6.2.x, or 6.3.x")
        return None

# Check if the system has an AMD GPU and ROCm installed
if 'Radeon' in subprocess.run(['lspci'], stdout=subprocess.PIPE, text=True).stdout:
    rocm_version = check_rocm_version()
    if rocm_version:
        result = install_tensorflow_rocm(rocm_version)
        if result is None:
            print("Now install tensorflow for CPU instead.")
            # Install tensorflow for CPU
            run(f'"{sys.executable}" -m pip install tensorflow',
                f"Installing tensorflow for CPU.",
                f"Couldn't install tensorflow.")
    else:
        print("Please install ROCm to use AMD GPUs if you want to use GPU acceleration.")
        print("Now install tensorflow for CPU instead.")
        # Install tensorflow for CPU
        run(f'"{sys.executable}" -m pip install tensorflow',
            f"Installing tensorflow for CPU.",
            f"Couldn't install tensorflow.")
else:
    # Install tensorflow for CUDA
    run(f'"{sys.executable}" -m pip install tensorflow',
        f"Installing tensorflow for CUDA.",
        f"Couldn't install tensorflow.")