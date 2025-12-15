

# BMS2 Quick Start

## Requirements

The following core software is required to build and run **BMS2**:

  * **Intel oneAPI Base Toolkit 2025.1.0** – You can use any installer method, e.g., offline/online installer, `apt` or `yum` package manager.
  * **CUDA toolkit (12.3)** – With Codeplay plug-in matching the oneAPI version if running tests on NVIDIA GPU (required for BMS2 assessment).
  * **ROCm (7.0.0)** – With Codeplay plug-in matching the oneAPI version if running tests on AMD GPU.
  * **Python ≥ 3.9** – Packages are defined in `scripts/requirements.txt` and will be installed automatically.
  * **CMake ≥ 3.10**
  * **g++ 11.4.0** – Different versions may lead to a fail during compilation.
  * **git** – For cloning repositories and submodules.
  * **NVIDIA DCGM and NVIDA NCU** – Optional, but required for GPU metrics collection.

## Installing oneAPI

You can install the **Intel oneAPI Base Toolkit** using the official guide. The recommended way is to use the online installer, which will download the required components during installation. Alternatively, you can use the offline installer if you have a stable internet connection.

After downloading the installer, run it with the following command:

```bash
sudo sh ./intel-oneapi-base-toolkit-2025.2.0.592.sh -a --silent --eula accept
```

*This will install the oneAPI Base Toolkit in the default location (`/opt/intel/oneapi`). Alternatively, you can install it in your home directory by running the installer without `sudo`.*

### Installing Codeplay Plug-ins (for NVIDIA/AMD GPUs)

After downloading and installing the oneAPI Base Toolkit, on systems equipped with NVIDIA or AMD GPUs, you will also need to install the Codeplay plug-in to enable SYCL support.

**For NVIDIA GPUs:**

```bash
curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&filters[]=2025.1.0&filters[]=linux"
chmod +x oneapi-for-nvidia-gpus-2025.1.0-rocm-all-linux.sh
sudo ./oneapi-for-nvidia-gpus-2025.1.0-rocm-all-linux.sh
```

**For AMD GPUs:**

```bash
curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=amd&filters[]=2025.1.0&filters[]=linux"
chmod +x oneapi-for-amd-gpus-2025.1.0-rocm-6.3-linux.sh
sudo ./oneapi-for-amd-gpus-2025.1.0-rocm-6.3-linux.sh
```

> **Note:** The Codeplay plug-in is required for BMS2 to work with NVIDIA and AMD GPUs. If you are using Intel GPUs, you can skip this step. The plug-in version must match the oneAPI version you have installed. Avoid `sudo` if you installed oneAPI in your home directory.

-----

## 1\. Compilation

First, load the environment (example for Intel oneAPI):

```bash
source /opt/intel/oneapi/setvars.sh # If oneAPI is installed globally
# OR
source ~/intel/oneapi/setvars.sh    # If oneAPI is installed locally
```

Then, use `build.sh` to compile BMS2 and/or other frameworks:

```bash
./build.sh --sigmo-arch=nvidia_gpu_sm_80 # Specify the GPU architecture (e.g., NVIDIA A100)
```

**Supported Architectures:**

  * `nvidia_gpu_sm_70` for NVIDIA V100 (and V100S)
  * `amd_gpu_gfx908` for AMD MI100
  * `intel_gpu_pvc` for Intel Max 1100
  * *(See Intel LLVM User Manual for a full list)*

## 2\. Dataset Preparation

Use `generate.sh` to initialize and process the dataset:

```bash
./generate.sh

## 3\. Running Experiments

Once the project is built and the dataset is ready, use `run.sh` to execute the experiments:

```bash
./run.sh
```
