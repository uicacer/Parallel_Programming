#!/bin/bash
# hybrid_job.sh
#SBATCH --job-name=hybrid_pipeline
#SBATCH --time=00:10:00
#SBATCH --nodes=2                    # Distributed: 4 separate computers
#SBATCH --ntasks-per-node=1         # 1 MPI process per computer (true distributed)
#SBATCH --cpus-per-task=4           # 4 OpenMP threads per MPI process
#SBATCH --account=ts_acer_chi
#SBATCH --partition=batch
#SBATCH --output=hybrid_results_%j.out

# Move to submission directory
cd $SLURM_SUBMIT_DIR

# Setup MPI programming environment
module load OpenMPI/4.1.6-GCC-13.2.0 
module load UCX/1.16.0-GCCcore-13.3.0

# Configure OpenMPI communication settings
export OMPI_MCA_btl='^uct,ofi'
export OMPI_MCA_pml='ucx'
export OMPI_MCA_mtl='^ofi'

# Configure OpenMP (CRITICAL for hybrid performance)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Set OpenMP threads per process

# DIAGNOSTIC: Print OpenMP settings
echo "DIAGNOSTIC: OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "DIAGNOSTIC: SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "DIAGNOSTIC: Expected total cores = $((4 * $SLURM_CPUS_PER_TASK))"

# Compile with both MPI and OpenMP support
mpicc -fopenmp -O2 data_parallel_hybrid_c_code.c -o data_parallel_hybrid_c_code

# Run the hybrid program
mpirun data_parallel_hybrid_c_code
