#!/usr/bin/env bash
# Bootstrap a fresh Brev (or any Ubuntu) instance for SmolVLA co-training.
# Usage on a new instance:
#   wget https://raw.githubusercontent.com/ielminawi/lerobot/smolvla-cotrain/scripts/setup_brev.sh
#   bash setup_brev.sh
#
# Idempotent — safe to re-run on the same instance.

set -e

REPO_URL="https://github.com/ielminawi/lerobot.git"
BRANCH="smolvla-cotrain"
TARGET_DIR="$HOME/lerobot"

echo "==> Updating apt + installing ffmpeg ..."
sudo apt update -qq
sudo apt install -y ffmpeg python-is-python3

echo "==> Configuring PATH for user-local pip installs ..."
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Cloning/updating fork ..."
if [ -d "$TARGET_DIR/.git" ]; then
    cd "$TARGET_DIR"
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone -b "$BRANCH" "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
fi

echo "==> Installing lerobot (editable) + extra deps ..."
pip install -e . --break-system-packages
pip install num2words "transformers<5.4" --break-system-packages

echo "==> Configuring passwordless sudo for shutdown (so auto-shutdown works) ..."
if ! sudo -n true 2>/dev/null; then
    echo "$USER ALL=(ALL) NOPASSWD: /sbin/shutdown" | sudo tee /etc/sudoers.d/no-passwd-shutdown >/dev/null
fi

echo ""
echo "============================================"
echo "Setup complete. Next steps:"
echo ""
echo "  1. Authenticate with HuggingFace:"
echo "       hf auth login"
echo ""
echo "  2. (optional) Authenticate with wandb:"
echo "       wandb login"
echo ""
echo "  3. Verify the tests pass:"
echo "       cd ~/lerobot"
echo "       python tests/test_cotraining_forward.py"
echo "       python tests/test_mixed_training.py"
echo ""
echo "  4. Run training / diagnostics / rollout as you like."
echo "============================================"
