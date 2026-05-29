#!/bin/bash
#SBATCH --job-name=sd_segfault
#SBATCH --output=/data/sd_segfault_%j.out
#SBATCH --error=/data/sd_segfault_%j.err
#SBATCH --time=00:01:00
#SBATCH --mem=200M
# Compile a tiny program that dereferences NULL, then run it -> SIGSEGV.
src="$(mktemp --suffix=.c)"; bin="$(mktemp)"
printf 'int main(void){volatile int *p=0;*p=42;return 0;}\n' > "$src"
gcc -O0 -o "$bin" "$src" || { echo "compile failed" >&2; exit 1; }
"$bin"
