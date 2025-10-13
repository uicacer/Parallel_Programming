#!/bin/bash

# ============================================================================
# SLURM RESOURCE ALLOCATION PARAMETERS
# These tell the cluster scheduler what resources your job needs
# ============================================================================

#SBATCH --job-name=openmp_data_pipeline
#SBATCH --time=00:10:00

# ============================================================================
# UNDERSTANDING --nodes vs --ntasks: THE KEY DIFFERENCE
# ============================================================================
#
# --nodes = Number of PHYSICAL COMPUTERS (compute nodes)
# --ntasks = Number of SEPARATE PROCESSES (program instances)
#
# The relationship between these parameters differs for OpenMP vs MPI:
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ FOR OpenMP (SHARED MEMORY - like this script):                      │
# ├─────────────────────────────────────────────────────────────────────┤
# │ --nodes=1   ✓ MUST be 1 (OpenMP can only use ONE computer)         │
# │ --ntasks=1  ✓ MUST be 1 (one process with multiple threads)        │
# │                                                                       │
# │ Why? OpenMP threads share memory, which only works on one machine.  │
# │ Parallelism comes from --cpus-per-task (threads, not processes)     │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ FOR MPI (DISTRIBUTED MEMORY):                                        │
# │ ** PRIMARY USE CASE: Running across MULTIPLE machines **            │
# ├─────────────────────────────────────────────────────────────────────┤
# │ SCENARIO 1: Running on a single node (testing/development only)     │
# │   --nodes=1                                                          │
# │   --ntasks=4        ← 4 MPI processes on the SAME computer         │
# │   Result: All 4 processes on one machine, communicate via MPI       │
# │   Note: If using only 1 node, OpenMP is usually better!            │
# │                                                                       │
# │ SCENARIO 2: Running across multiple nodes (typical MPI usage)       │
# │   --nodes=4                                                          │
# │   --ntasks=4        ← 4 MPI processes across 4 DIFFERENT computers │
# │   Result: 1 process per node (distributed across 4 machines)        │
# │                                                                       │
# │ SCENARIO 3: Multiple processes per node (hybrid distribution)       │
# │   --nodes=2                                                          │
# │   --ntasks=8        ← 8 MPI processes across 2 computers           │
# │   Result: 4 processes per node (8 total across 2 machines)          │
# │                                                                       │
# │ Why MPI? Designed for problems that:                                │
# │   • Need more compute power than 1 machine can provide              │
# │   • Have datasets too large to fit in 1 machine's memory            │
# │   • Benefit from distributing work across many computers            │
# │                                                                       │
# │ MPI processes don't share memory - they communicate via             │
# │ message passing, so they can run on different computers.            │
# └─────────────────────────────────────────────────────────────────────┘

#SBATCH --nodes=1
# Number of physical computers (compute nodes) to allocate
#
# For THIS script (OpenMP): KEEP as 1
#   - OpenMP uses shared memory parallelism
#   - All threads must run on the SAME physical computer
#   - Using >1 would waste resources (extra nodes sit idle)
#
# For MPI: Can be >1
#   - MPI processes can run across multiple computers
#   - Example: --nodes=4 allocates 4 different machines
#   - Useful when computation won't fit on one machine

#SBATCH --ntasks=1
# Number of separate program instances (MPI processes) to launch
#
# For THIS script (OpenMP): KEEP as 1
#   - You want ONE program that creates multiple threads internally
#   - Using >1 would launch multiple separate OpenMP programs (usually wrong)
#   - Example: --ntasks=4 would run 4 independent copies of your program
#
# For MPI: Set to desired number of MPI processes
#   - This is how you control MPI parallelism
#   - Example: --ntasks=8 runs 8 MPI processes
#   - Combined with --nodes, SLURM distributes tasks across nodes
#   - Distribution: (ntasks / nodes) = processes per node
#
# KEY INSIGHT: When BOTH are 1 (like OpenMP):
#   - You get 1 process on 1 computer
#   - Parallelism comes from --cpus-per-task (threads within that process)
#
# When ntasks > 1 but nodes = 1 (single-node MPI):
#   - You get multiple processes on the SAME computer
#   - Still using MPI message passing (not shared memory)
#   - Less efficient than using all nodes, but useful for testing

#SBATCH --cpus-per-task=4
# Number of CPU cores allocated to EACH task (process)
#
# For THIS script (OpenMP): This is your parallelism control
#   - Determines maximum number of OpenMP threads you can create
#   - Example: --cpus-per-task=8 allows up to 8 parallel threads
#   - Set OMP_NUM_THREADS to this value in your code
#   - More CPUs = more parallel threads = potentially faster
#
# For MPI (with OpenMP hybrid):
#   - Allocates cores per MPI process for thread-level parallelism
#   - Example: --ntasks=4 --cpus-per-task=2 = 4 MPI processes × 2 threads each
#   - Total cores used = ntasks × cpus-per-task
#
# Performance tip: Choose based on available cores per node
#   - Don't exceed cores available on the node
#   - Check with: sinfo -o "%P %n %c" (shows partition, nodes, and core counts)
#   - Or: sinfo -o "%P %n %c %m" (adds memory in MB - megabytes)

#SBATCH --account=ts_acer_chi
# Your billing/allocation account on the cluster
# This is required for job accounting and resource tracking
# Change this to: your actual account name (ask your sysadmin)

#SBATCH --output=openmp_results_%j.txt
# File where job output will be saved
# %j = job ID number (automatically replaced by SLURM)
# Captures both stdout and stderr by default
# Change to: any filename you prefer (e.g., "results_%j.out")

# ============================================================================
# VISUAL SUMMARY: Resource Allocation Patterns
# ============================================================================
#
# OpenMP (this script):
# ┌─────────────────┐
# │   Node 1        │
# │  ┌───────────┐  │
# │  │ Process 1 │  │  ← 1 process with 4 threads
# │  │ 🧵🧵🧵🧵   │  │
# │  └───────────┘  │
# └─────────────────┘
#
# MPI with --nodes=1 --ntasks=4:
# ┌─────────────────┐
# │   Node 1        │
# │  ┌───┐┌───┐    │
# │  │ P1││ P2│    │  ← 4 processes on same node
# │  └───┘└───┘    │
# │  ┌───┐┌───┐    │
# │  │ P3││ P4│    │
# │  └───┘└───┘    │
# └─────────────────┘
#
# MPI with --nodes=4 --ntasks=4:
# ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
# │Node1│  │Node2│  │Node3│  │Node4│
# │┌───┐│  │┌───┐│  │┌───┐│  │┌───┐│
# ││ P1││  ││ P2││  ││ P3││  ││ P4││  ← 1 process per node
# │└───┘│  │└───┘│  │└───┘│  │└───┘│
# └─────┘  └─────┘  └─────┘  └─────┘

# ============================================================================
# USEFUL CLUSTER INFORMATION COMMANDS
# ============================================================================
#
# Check available resources on your cluster:
#
# 1. Basic info (partition, node, cores):
#    $ sinfo -o "%P %n %c"
#    Example output:
#    PARTITION    NODELIST         CPUS
#    compute*     node001          48
#    compute*     node002          48
#
# 2. Add memory in MEGABYTES (MB):
#    $ sinfo -o "%P %n %c %m"
#    Example output:
#    PARTITION    NODELIST         CPUS  MEMORY
#    compute*     node001          48    385657    ← 385657 MB (~377 GB)
#    compute*     node002          48    385657
#
# Note: Memory is displayed in MB. To convert to GB, divide by 1024.
#
# ============================================================================

# ============================================================================
# MODULE LOADING
# ============================================================================

# Load gcc compiler (includes OpenMP support)
# OpenMP is built into gcc - no separate module needed
module load gcc/11.2.0

# Note: openmpi is NOT needed for this OpenMP program
# Only load openmpi for MPI-based programs:
# module load openmpi/mlnx/gcc/64/4.1.5a1

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Set the number of OpenMP threads to match allocated CPUs
# This ensures your program uses all available cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================"
echo "SLURM Job Configuration"
echo "========================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Nodes allocated:  $SLURM_JOB_NUM_NODES"
echo "Tasks (processes):$SLURM_NTASKS"
echo "CPUs per task:    $SLURM_CPUS_PER_TASK"
echo "OpenMP threads:   $OMP_NUM_THREADS"
echo "Node list:        $SLURM_JOB_NODELIST"
echo "========================================"
echo ""

# ============================================================================
# COMPILATION
# ============================================================================

echo "=== COMPILING OPENMP PROGRAM ==="
echo "Compiling data_parallel_shared_memory_c_code.c..."
echo ""

# Compile the C program with OpenMP support
# -fopenmp: Enables OpenMP parallelization
# -o: Output executable name
# -lm: Links math library (needed for M_PI, fabs, etc.)
gcc -fopenmp -o data_parallel_shared_memory_c_code data_parallel_shared_memory_c_code.c -lm

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
else
    echo "✗ ERROR: Compilation failed!"
    echo "Check that data_parallel_shared_memory_c_code.c exists in the current directory"
    exit 1
fi

# ============================================================================
# EXECUTION
# ============================================================================

echo "=== RUNNING OPENMP PROGRAM ==="
echo "Executing with $OMP_NUM_THREADS threads..."
echo ""

# Run the compiled program
# OpenMP will automatically use OMP_NUM_THREADS for parallelization
./data_parallel_shared_memory_c_code

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Job completed successfully"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "✗ Job failed during execution"
    echo "========================================"
    exit 1
fi
