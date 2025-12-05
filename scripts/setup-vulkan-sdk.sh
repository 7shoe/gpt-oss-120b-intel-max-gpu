#!/bin/bash
# Setup script for Vulkan SDK
# Required for SYCL GPU support

set -e

echo "=== Setting up Vulkan SDK ==="

# fixed Vulkan version (Oct 2025)
VULKAN_VERSION="1.4.321.1"
INSTALL_DIR="${VULKAN_VERSION}"

# Download Vulkan SDK if not already present
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Downloading Vulkan SDK..."
    curl -L -o vulkan_sdk.tar.xz "https://sdk.lunarg.com/sdk/download/${VULKAN_VERSION}/linux/vulkan_sdk.tar.xz"

    echo "Extracting Vulkan SDK..."
    tar -xf vulkan_sdk.tar.xz

    echo "Vulkan SDK extracted to: ${INSTALL_DIR}"
    rm vulkan_sdk.tar.xz
else
    echo "Vulkan SDK already installed at: ${INSTALL_DIR}"
fi

echo ""
echo "To use Vulkan SDK, run:"
echo "  source ${INSTALL_DIR}/setup-env.sh"
echo ""
echo "=== Vulkan SDK setup complete ==="
