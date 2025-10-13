#!/bin/bash

# ============================================================================
# SLURM RESOURCE ALLOCATION FOR MPI (DISTRIBUTED MEMORY) PROGRAM
# ============================================================================

#SBATCH --job-name=mpi_pipeline
#SBATCH --time=00:10:00              # Maximum runtime (10 minutes)
                                     # MPI has overhead for inter-node communication
                                     # so allow more time than equivalent OpenMP job

# ============================================================================
# KEY MPI PARAMETERS: --nodes and --ntasks-per-node
# ============================================================================

#SBATCH --nodes=4
# Number of physical computers (compute nodes) to use
# For MPI: This is how many separate machines you're using
# Example:
#   --nodes=4 → Use 4 different physical computers
#   --nodes=1 → Use only 1 computer (not typical for MPI)

#SBATCH --ntasks-per-node=1
# Number of MPI processes to run ON EACH NODE
# This is the KEY parameter that determines your total parallelism!
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ CRITICAL DIFFERENCE: --ntasks-per-node=4 vs --ntasks-per-node=1    │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                       │
# │ OPTION 1: --ntasks-per-node=1 (your original setting)               │
# │   Result: 4 nodes × 1 process/node = 4 TOTAL MPI processes         │
# │   Visual:                                                            │
# │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                          │
# │   │Node 1│  │Node 2│  │Node 3│  │Node 4│                          │
# │   │  P0  │  │  P1  │  │  P2  │  │  P3  │  ← 1 process per node   │
# │   └──────┘  └──────┘  └──────┘  └──────┘                          │
# │   Use this when:                                                     │
# │   - You want maximum memory per process                             │
# │   - Each process needs lots of resources                            │
# │   - Problem size is VERY large per process                          │
# │                                                                       │
# │ OPTION 2: --ntasks-per-node=4 (CURRENT setting)                    │
# │   Result: 4 nodes × 4 processes/node = 16 TOTAL MPI processes      │
# │   Visual:                                                            │
# │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
# │   │ Node 1  │  │ Node 2  │  │ Node 3  │  │ Node 4  │             │
# │   │ P0  P1  │  │ P4  P5  │  │ P8  P9  │  │ P12 P13 │             │
# │   │ P2  P3  │  │ P6  P7  │  │ P10 P11 │  │ P14 P15 │ ← 4 per node│
# │   └─────────┘  └─────────┘  └─────────┘  └─────────┘             │
# │   Use this when:                                                     │
# │   - You want MORE parallelism (faster computation)                  │
# │   - Each process doesn't need entire node's resources               │
# │   - Better load balancing across nodes                              │
# │                                                                       │
# │ FORMULA: Total MPI processes = nodes × ntasks-per-node             │
# │   Example 1: 4 nodes × 1 task/node  = 4 total processes            │
# │   Example 2: 4 nodes × 4 tasks/node = 16 total processes           │
# │   Example 3: 4 nodes × 8 tasks/node = 32 total processes           │
# └─────────────────────────────────────────────────────────────────────┘
#
# PERFORMANCE CONSIDERATION:
# More processes = more parallelism BUT also more communication overhead
# - Processes on SAME node: Fast communication (shared memory hardware)
# - Processes on DIFFERENT nodes: Slower (network communication)
#
# CURRENT CONFIGURATION BREAKDOWN:
# - 4 processes per node means each node runs 4 separate MPI processes
# - These 4 processes share the node's memory and resources
# - They communicate via MPI message passing (even though on same node)
# - Total: 16 processes communicating across 4 machines

#SBATCH --cpus-per-task=1
# Number of CPU cores allocated to EACH MPI process
#
# For PURE MPI (no OpenMP threading): Keep this at 1
#   - Each MPI process uses exactly 1 CPU core
#   - Total cores used = nodes × ntasks-per-node × cpus-per-task
#   - With current settings: 4 × 4 × 1 = 16 cores total
#
# For HYBRID MPI+OpenMP: Set this >1
#   - Each MPI process can spawn OpenMP threads
#   - Example: --cpus-per-task=4 gives each MPI process 4 cores
#   - Then each MPI process creates 4 OpenMP threads
#   - Total cores: 4 nodes × 4 MPI/node × 4 cores/MPI = 64 cores

#SBATCH --account=ts_acer_chi
# Your billing account on the cluster
# Required for job accounting and resource tracking

#SBATCH --partition=batch
# Which partition (queue) to submit the job to
# Different partitions may have different:
#   - Available node types
#   - Time limits
#   - Priority levels
# Check available partitions: sinfo -o "%P %l %N"

#SBATCH --output=mpi_results_%j.out
# File where job output will be saved
# %j = job ID number (automatically replaced by SLURM)
# All stdout and stderr from all MPI processes goes here
# NOTE: MPI output from different processes may interleave

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Move to the directory where the job was submitted from
# This ensures we can find our source files and save output in the right place
cd $SLURM_SUBMIT_DIR

# ============================================================================
# MODULE LOADING - MPI ENVIRONMENT
# ============================================================================

# Load OpenMPI library (MPI implementation)
# OpenMPI provides the mpicc compiler and mpirun launcher
# Version 4.1.6 compiled with GCC 13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0

# Load UCX (Unified Communication X) library
# UCX is a communication framework for high-performance networking
# Used by OpenMPI for fast inter-node communication
module load UCX/1.16.0-GCCcore-13.3.0

# ============================================================================
# OPENMPI COMMUNICATION CONFIGURATION
# These environment variables tune MPI performance for your cluster
# ============================================================================

# Configure OpenMPI communication settings for optimal performance
export OMPI_MCA_btl='^uct,ofi'
# BTL = Byte Transfer Layer (low-level communication)
# ^uct,ofi means "exclude UCT and OFI transport methods"
# This tells OpenMPI NOT to use certain communication protocols
# Reason: These may conflict with UCX or be slower on this cluster

export OMPI_MCA_pml='ucx'
# PML = Point-to-Point Messaging Layer
# Forces OpenMPI to use UCX for inter-process communication
# UCX is optimized for high-performance networks (InfiniBand, RoCE)
# This typically gives better performance than default TCP

export OMPI_MCA_mtl='^ofi'
# MTL = Matching Transport Layer
# Exclude OFI (libfabric) matching transport
# Similar to BTL exclusion - avoids conflicts with UCX

# WHY THESE SETTINGS?
# Modern clusters use specialized hardware for fast networking
# UCX provides a unified interface to these high-speed networks
# These settings ensure OpenMPI uses the fastest path available

# ============================================================================
# DISPLAY JOB INFORMATION
# ============================================================================

echo "========================================"
echo "MPI Job Configuration"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Job Name:              $SLURM_JOB_NAME"
echo "Partition:             $SLURM_JOB_PARTITION"
echo "Number of nodes:       $SLURM_JOB_NUM_NODES"
echo "Tasks per node:        $SLURM_NTASKS_PER_NODE"
echo "Total MPI processes:   $SLURM_NTASKS"
echo "CPUs per task:         ${SLURM_CPUS_PER_TASK:-1}"
echo "Total cores used:      $(($SLURM_NTASKS * ${SLURM_CPUS_PER_TASK:-1}))"
echo "Node list:             $SLURM_JOB_NODELIST"
echo "========================================"
echo ""
echo "Configuration breakdown:"
echo "  $SLURM_JOB_NUM_NODES nodes × $SLURM_NTASKS_PER_NODE tasks/node = $SLURM_NTASKS total MPI processes"
echo ""

# ============================================================================
# COMPILATION
# ============================================================================

echo "=== COMPILING MPI PROGRAM ==="
echo "Compiling data_parallel_dist_memory_c_code.c..."
echo ""

# Compile the MPI program using mpicc
# mpicc: MPI C compiler wrapper (automatically links MPI libraries)
# -O2: Optimization level 2 (good balance of speed and compile time)
# -o: Output executable name
# -lm: Link math library (for functions like sqrt, sin, M_PI, etc.)
mpicc -O2 data_parallel_dist_memory_c_code.c -o data_parallel_dist_memory_c_code -lm

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
else
    echo "✗ ERROR: Compilation failed!"
    echo "Check that data_parallel_dist_memory_c_code.c exists in the current directory"
    exit 1
fi

# ============================================================================
# EXECUTION
# ============================================================================

echo "=== RUNNING MPI PROGRAM ==="
echo "Launching $SLURM_NTASKS MPI processes across $SLURM_JOB_NUM_NODES nodes..."
echo "Distribution: $SLURM_NTASKS_PER_NODE processes per node"
echo ""

# Run the MPI program across all requested processes
# mpirun: MPI launcher that starts processes across nodes
# 
# How mpirun works:
# 1. Reads SLURM allocation (knows which nodes are assigned)
# 2. Spawns processes on each node according to --ntasks-per-node
# 3. Sets up MPI communication channels between all processes
# 4. Each process gets a unique rank (0, 1, 2, ..., N-1)
# 5. Processes can communicate using MPI_Send, MPI_Recv, etc.
#
# SLURM automatically configures mpirun, so you don't need to specify:
# - Number of processes (uses $SLURM_NTASKS)
# - Which nodes to use (uses $SLURM_JOB_NODELIST)
# - How to distribute processes (uses $SLURM_NTASKS_PER_NODE)
#
# Alternative: Can use srun instead of mpirun
#   srun data_parallel_dist_memory_c_code
# srun is SLURM's native launcher (sometimes faster startup)
mpirun data_parallel_dist_memory_c_code

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

# ============================================================================
# NOTES ON MPI PERFORMANCE
# ============================================================================
#
# Communication Patterns:
# - Intra-node (same machine): Very fast (shared memory hardware)
# - Inter-node (different machines): Slower (network, even with InfiniBand)
#
# With --ntasks-per-node=4:
# - 4 processes on each node communicate FAST (intra-node)
# - Processes on different nodes communicate SLOWER (inter-node)
# - This is usually a good trade-off between parallelism and communication
#
# Performance Tips:
# 1. More tasks/node = more parallelism but more intra-node communication
# 2. Fewer tasks/node = less parallelism but each process has more resources
# 3. Optimal setting depends on:
#    - Problem size
#    - Communication vs computation ratio
#    - Available cores per node
#    - Memory requirements per process
#
# Testing Recommendations:
# Try different configurations to find optimal performance:
#   --nodes=4 --ntasks-per-node=1  → 4 total processes
#   --nodes=4 --ntasks-per-node=2  → 8 total processes
#   --nodes=4 --ntasks-per-node=4  → 16 total processes (CURRENT)
#   --nodes=4 --ntasks-per-node=8  → 32 total processes
#
# ============================================================================
