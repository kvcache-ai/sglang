#!/bin/bash
#
# CI script for installing KT-Kernel (ktransformers) from source
#
# This script is used in GitHub Actions CI to install the kt-kernel package
# which provides AMX-optimized CPU kernels for hybrid CPU-GPU inference.
#
# Environment Variables:
#   KT_KERNEL_REPO: Git repository URL (default: https://github.com/kvcache-ai/ktransformers)
#   KT_KERNEL_BRANCH: Git branch/tag to checkout (default: main)
#   GITHUB_WORKSPACE: CI workspace directory (set by GitHub Actions)
#
# Usage:
#   bash scripts/ci/ci_install_kt_kernel.sh

set -euxo pipefail

echo "=========================================="
echo "Installing KT-Kernel from source"
echo "=========================================="

# Configuration
KT_REPO=${KT_KERNEL_REPO:-"https://github.com/kvcache-ai/ktransformers"}
KT_BRANCH=${KT_KERNEL_BRANCH:-"main"}
WORK_DIR=${GITHUB_WORKSPACE:-$(pwd)}
INSTALL_DIR="/tmp/ktransformers-install"

echo "Configuration:"
echo "  Repository: ${KT_REPO}"
echo "  Branch: ${KT_BRANCH}"
echo "  Work directory: ${WORK_DIR}"
echo "  Install directory: ${INSTALL_DIR}"
echo ""

# Step 1: Clean up previous installation
echo "[1/6] Cleaning up previous installation..."
rm -rf ${INSTALL_DIR}
pip uninstall -y ktransformers kt-kernel || true
echo "✓ Cleanup complete"
echo ""

# Step 2: Clone repository
echo "[2/6] Cloning ktransformers repository..."
git clone --depth 1 --branch ${KT_BRANCH} ${KT_REPO} ${INSTALL_DIR}
cd ${INSTALL_DIR}
echo "✓ Repository cloned"
echo ""

# Step 3: Initialize and update submodules
echo "[3/6] Initializing and updating submodules..."
git submodule update --init --recursive
echo "✓ Submodules initialized"
echo ""

# Step 4: Check if install.sh exists
echo "[4/6] Checking for kt-kernel/install.sh..."
if [ ! -f "kt-kernel/install.sh" ]; then
    echo "❌ Error: kt-kernel/install.sh not found"
    echo "Directory contents:"
    ls -la
    if [ -d "kt-kernel" ]; then
        echo "kt-kernel directory contents:"
        ls -la kt-kernel/
    fi
    exit 1
fi
echo "✓ Found kt-kernel/install.sh"
echo ""

# Step 5: Run kt-kernel installation script
echo "[5/6] Running kt-kernel/install.sh..."
cd kt-kernel

# Make install.sh executable
chmod +x install.sh

# Run the installation script
./install.sh

cd ${INSTALL_DIR}
echo "✓ kt-kernel installed via install.sh"
echo ""

# Step 6: Verify installation
echo "[6/6] Verifying installation..."

# Try to import kt_kernel
python3 << 'EOF'
import sys

try:
    from kt_kernel import KTMoEWrapper
    print("✓ Successfully imported KTMoEWrapper from kt_kernel")
except ImportError as e:
    print(f"❌ Failed to import kt_kernel: {e}")
    sys.exit(1)

# Check AMX support
try:
    import torch
    if torch._C._cpu._is_amx_tile_supported():
        print("✓ AMX tile instructions are supported")
    else:
        print("⚠ AMX tile instructions are NOT supported on this CPU")
        print("  KT-kernel will fall back to non-AMX CPU kernels")
except Exception as e:
    print(f"⚠ Could not check AMX support: {e}")

# Print version info if available
try:
    import ktransformers
    if hasattr(ktransformers, '__version__'):
        print(f"✓ ktransformers version: {ktransformers.__version__}")
except:
    pass

print("\n✓ KT-Kernel installation verified successfully")
EOF

VERIFY_RESULT=$?

# Cleanup and return to work directory
cd ${WORK_DIR}

if [ $VERIFY_RESULT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "KT-Kernel installation completed successfully"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "KT-Kernel installation verification failed"
    echo "=========================================="
    exit 1
fi
