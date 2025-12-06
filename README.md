# GPT-OSS-120B on Intel MAX GPU 1550

This repository contains everything needed to build and run OpenAI's GPT-OSS-120B model on **a single Intel Data Center GPU Max 1550** (Ponte Vecchio) using llama.cpp with SYCL backend.

## Overview

GPT-OSS-120B is a 116.8B parameter Mixture of Experts (MoE) model released by OpenAI with 60 GB on a **single Intel XPU** using memory mapping and the SYCL programming model.

### Hardware Requirements
- Intel Data Center GPU Max 1550 (Ponte Vecchio)
- 64 GB System RAM (for memory-mapped model)
- x86_64 Linux system

### Software Requirements
This will only work for ... 
- Intel oneAPI Base Toolkit 25.190.0
- Vulkan SDK 1.4.321.1
- CMake 3.18+
- GCC 7+ or Intel C++ Compiler (icx/icpx)
- ~60 GB free disk space for model (do not put in `/home`)

## Quick Start

### 1. Clone repo
Navigate to directory (not `home`) and clone repo.
```
module load frameworks
git clone git@github.com:7shoe/gpt-oss-120b-intel-max-gpu.git
cd gpt-oss-120b-intel-max-gpu
```
### 2. Setup Vulkan SDK
Create directory `1.4.321.1` via 
```bash
cd scripts
./setup-vulkan-sdk.sh
```

### 3. Build llama.cpp with SYCL
Clone the llama.cpp repo, re-set it to the target date October 5th.
Tehn, it will apply necessary patches for Intel GPU compatibility and build with SYCL backend using Intel compiler.
```bash
./build-llama-cpp.sh
```

### 4. Download GPT-OSS-120B Model
The quantized model is provided on unsloth's [Huggingface repo](https://huggingface.co/unsloth/gpt-oss-120b-GGUF).
It is the model with weights `gpt-oss-120b-Q4_K_M-00001-of-00002.gguf` (47 GB) and `gpt-oss-120b-Q4_K_M-00002-of-00002.gguf` (13 GB).
```bash
./download-gpt-oss-120b.sh
```

### 4. Verify functionality
See if it works by running 
```bash
./run-inference.sh "What is the meaning of life?"
```

**Note**: First load takes 60-90 seconds as the model initializes.

## Model Specifications

### Single XPU Inference

```bash
# use XPU device 0
export ONEAPI_DEVICE_SELECTOR="level_zero:0"

# run inference
./llama.cpp/build/bin/llama-cli \
    -m ../models/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
    -p "What is 9*8 + 9?" \
    -ngl 80 \
    -c 2048 \
    -n 200
```

### Parameters
- `-m`: Path to first part of model (second part auto-detected)
- `-p`: Prompt text
- `-ngl`: Number of GPU layers (80 = all layers)
- `-c`: Context size (up to 131072)
- `-n`: Number of tokens to generate

### Tips for 120B Model

1. **Allow time for loading**: First inference takes 60-90 seconds
2. **Sufficient RAM**: Ensure 60+ GB free RAM before starting
3. **SSD recommended**: Faster model loading from NVMe storage
4. **Smaller contexts**: Use `-c 512` or `-c 1024` for faster initial responses

## Directory Structure

```
gpt-oss-120b-intel-max-gpu/
├── README.md              # This file
├── patches/               # Source code patches (same as 20B)
│   ├── 001-fix-tokenization-byte-fallback.patch
│   ├── 002-link-stdc++fs.patch
│   ├── 003-experimental-filesystem-support.patch
│   └── README.md
├── scripts/               # Build and utility scripts
│   ├── setup-vulkan-sdk.sh
│   ├── build-llama-cpp.sh
│   ├── download-gpt-oss-120b.sh  # Downloads 60 GB model
│   └── run-inference.sh
├── docs/                  # Additional documentation
│   ├── INSTALLATION.md
│   ├── TROUBLESHOOTING.md
│   └── PERFORMANCE.md
├── models/                # Model files (downloaded)
│   ├── gpt-oss-120b-Q4_K_M-00001-of-00002.gguf
│   └── gpt-oss-120b-Q4_K_M-00002-of-00002.gguf
└── test-results/          # Test outputs and benchmarks
    ├── test-2plus2.log
    └── README.md
```

## Troubleshooting

### "Out of memory" during model load

**Issue**: Insufficient system RAM

**Solution**:
- Ensure 60+ GB free RAM: `free -h`
- Close other applications
- Consider using smaller quantization (Q2_K ~40 GB)

### Slow loading (>5 minutes)

**Issue**: Slow storage device

**Solution**:
- Move model to SSD/NVMe storage
- Enable mmap: `--mmap` (default)

### Low performance (<10 tok/s)

**Issue**: Not all layers on GPU

**Solution**:
```bash
# Ensure all layers offloaded
-ngl 80  # or higher

# Verify in output:
load_tensors: offloaded 37/37 layers to GPU
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more issues and solutions.

## Patches Explained

Same patches as 20B model (fully compatible):

### Tokenization Fix (001)
Fixes crashes when byte tokens missing from vocabulary by replacing `.at()` with `.find()` and falling back to unknown token.

### Filesystem Library Linking (002)
Links experimental filesystem library (`libstdc++fs.a`) required for GCC 7 compatibility.

### Experimental Filesystem Support (003)
Adds conditional compilation for `<experimental/filesystem>` on older GCC versions.

See [patches/README.md](patches/README.md) for detailed information.

## Citations

```bibtex
@misc{openai2025gptoss,
  title={GPT-OSS: Open Source Language Models},
  author={OpenAI},
  year={2025},
  url={https://openai.com/index/introducing-gpt-oss/}
}
```

## References

- [GPT-OSS Announcement](https://openai.com/index/introducing-gpt-oss/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Intel oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)
- [Model on HuggingFace](https://huggingface.co/unsloth/gpt-oss-120b-GGUF)

## License

This repository: MIT License

The GPT-OSS-120B model: Apache 2.0

llama.cpp: MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- OpenAI for releasing GPT-OSS-120B
- ggml-org for llama.cpp
- Intel for oneAPI and SYCL support
- Unsloth for GGUF conversions

---

**Single GPU. 120 Billion Parameters. Full Speed.** 🚀
