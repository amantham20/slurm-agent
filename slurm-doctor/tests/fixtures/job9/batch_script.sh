#!/bin/bash
#SBATCH --job-name=segfault
#SBATCH --partition=cpu
#SBATCH --mem=100M
#SBATCH --time=00:02:00
#SBATCH --output=/data/jobs/%x-%j.out

# Dereference NULL. Prefer a tiny C program; fall back to ctypes when the
# node has no compiler (this cluster's runtime image does not).
if command -v gcc >/dev/null 2>&1; then
    src=$(mktemp /tmp/segv-XXXX.c)
    cat > "$src" <<'EOF'
int main(void) { int *p = 0; return *p; }
EOF
    gcc -o /tmp/segv-bin "$src" && exec /tmp/segv-bin
fi
exec python3 -c 'import ctypes; ctypes.string_at(0)'
