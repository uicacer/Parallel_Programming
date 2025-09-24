#!/bin/bash
#SBATCH --job-name=sequential_pipeline
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=ts_acer_chi
#SBATCH --output=serial_pipeline_results_%j.txt

echo "=========================================="
echo "SEQUENTIAL MATHEMATICAL PROCESSING PIPELINE"
echo "=========================================="
echo "Purpose: Establish performance baseline using 1 CPU core"
echo "Approach: Process each number completely before moving to next"
echo ""

# Load compiler
module purge  # Clear any previously loaded modules to avoid conflicts 
module load gcc/11.2.0


# BEST PRACTICE: Navigate to the job submission directory
# Some clusters start jobs in different directories than where you submitted from.
# This ensures we're in the same directory as our source files.

cd $SLURM_SUBMIT_DIR

echo "=== Compiling Sequential Version ==="
gcc -O2 serial_c_code.c -o serial_c_code

# Run the compiled code
./serial_c_code



# GCC Compilation Command Breakdown
#gcc -O2 serial_c_code.c -o serial_c_code
#│  │   │               │  │
#│  │   │               │  └─ Output filename flag
#│  │   │               └──── Specify name of executable file to create
#│  │   └─────────────────── Source code filename (input)
#│  └───────────────────── Optimization level flag
#└──────────────────────── GNU C Compiler command

# Detailed explanation of each component:

# gcc
# - The GNU C Compiler
# - Transforms C source code into executable machine code
# - Part of GCC (GNU Compiler Collection)

# -O2
# - Optimization level 2 (capital O, not zero)
# - Enables most optimization without excessive compilation time
# - Makes code run faster by optimizing:
#   * Loop unrolling
#   * Function inlining
#   * Dead code elimination
#   * Common subexpression elimination
#   * Register optimization
# - Balances compilation speed vs runtime performance
# - Safe optimizations that don't change program behavior

# serial_c_code.c
# - Input source file written in C
# - Must have .c extension for C code
# - Contains the source code to be compiled

# -o
# - Output flag (lowercase o)
# - Specifies the name of the executable file to create
# - Without this flag, GCC creates default executable named "a.out"

# serial_c_code
# - Name of the output executable file
# - No file extension needed on Linux/Unix systems
# - This is what you run with: ./serial_c_code

# Alternative optimization levels:
# -O0  : No optimization (default, fastest compilation)
# -O1  : Basic optimization
# -O2  : Standard optimization (recommended for most cases)
# -O3  : Aggressive optimization (may increase binary size)
# -Os  : Optimize for size rather than speed
# -Ofast: Maximum optimization (may break IEEE compliance)
