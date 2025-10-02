#!/bin/bash

# SLURM RESOURCE ALLOCATION PARAMETERS
# These tell the cluster scheduler what resources your job needs

#SBATCH --job-name=openmp_data_pipeline
# Job name that appears in queue listings
# Change this to: any descriptive name (e.g., "my_openmp_test")

#SBATCH --time=00:10:00
# Maximum wall-clock time your job can run (HH:MM:SS format)
# Change this to: longer time if your job needs more (e.g., "00:10:00" for 10 minutes)
# If job exceeds this time, SLURM will kill it

#SBATCH --nodes=1
# Number of physical computers (nodes) to use
# For OpenMP: ALWAYS keep this as 1 (shared memory requires single computer)
# Change this to: >1 only for MPI jobs that need multiple computers

#SBATCH --ntasks=1
# Number of separate program instances (processes) to run
# For OpenMP: ALWAYS use 1 (you want 1 program using multiple threads)
# Change this to: >1 for MPI (e.g., --ntasks=4 runs 4 copies of your program)
# Example: --ntasks=4 would run 4 separate instances of your program

#SBATCH --cpus-per-task=4
# Number of CPU cores allocated to each task
# For OpenMP: This determines how many threads you can create
# Change this to: 2, 8, 16, etc. based on available cores and your needs
# More CPUs = more parallel threads = potentially faster execution
# Example: --cpus-per-task=8 gives you 8 cores for OpenMP threads

#SBATCH --account=ts_acer_chi
# Your billing account on the cluster
# Change this to: your actual account name (required for job submission)

#SBATCH --output=openmp_results_%j.txt
# Where to save job output (%j gets replaced with job ID number)
# Change this to: any filename pattern you prefer
# Example: "my_results_%j.out" or "output_file.txt"

echo "=== OPENMP PARALLEL PROCESSING ==="

# MODULE LOADING
# Clear any previously loaded modules to avoid conflicts
module purge
module load gcc/11.2.0  # Load GCC compiler (includes OpenMP support)

# Navigate to job submission directory
cd $SLURM_SUBMIT_DIR

# OPENMP CONFIGURATION
# Tell OpenMP how many threads to use (should match --cpus-per-task)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# If you change --cpus-per-task to 8, OpenMP will automatically use 8 threads

echo "Compiling with OpenMP support..."
gcc -fopenmp -O2 data_parallel_shared_memory_c_code.c -o data_parallel_shared_memory_c_code

echo "Running OpenMP version..."
./data_parallel_shared_memory_c_code

# WHAT HAPPENS IF YOU CHANGE PARAMETERS:

# If you change --cpus-per-task=2:
#   - OpenMP uses 2 threads instead of 4
#   - Job runs slower (less parallelism)
#   - Uses fewer cluster resources

# If you change --cpus-per-task=8:
#   - OpenMP uses 8 threads instead of 4  
#   - Job may run faster (more parallelism)
#   - Uses more cluster resources
#   - May wait longer in queue (requesting more resources)

# If you change --ntasks=4 (DON'T DO THIS FOR OPENMP):
#   - SLURM runs 4 copies of your program simultaneously
#   - Each copy would compete for the same data
#   - Results would be wrong/corrupted
#   - This is for MPI, not OpenMP

# If you change --nodes=2 (DON'T DO THIS FOR OPENMP):
#   - SLURM tries to use 2 computers
#   - OpenMP can't share memory across computers
#   - Job will likely fail
#   - This is for MPI, not OpenMP
