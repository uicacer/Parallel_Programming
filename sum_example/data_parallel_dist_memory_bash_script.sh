#!/bin/bash
# mpi_job.sh
#SBATCH --job-name=mpi_pipeline
#SBATCH --time=00:10:00              # Allow more time for MPI setup
#SBATCH --nodes=4                    # Use 4 separate compute nodes for true distributed memory
#SBATCH --ntasks-per-node=1         # Run 1 MPI process per node (total: 4 processes)
#SBATCH --account=ts_acer_chi
#SBATCH --partition=batch
#SBATCH --output=mpi_results_%j.out

# Move to submission directory
cd $SLURM_SUBMIT_DIR

# Setup MPI programming environment
module load OpenMPI/4.1.6-GCC-13.2.0 
module load UCX/1.16.0-GCCcore-13.3.0

# Configure OpenMPI communication settings for optimal performance
export OMPI_MCA_btl='^uct,ofi'      # Exclude UCT and OFI byte transport layers
export OMPI_MCA_pml='ucx'           # Use UCX as point-to-point messaging layer
export OMPI_MCA_mtl='^ofi'          # Exclude OFI matching transport layer

# Compile the MPI program
mpicc -O2 data_parallel_dist_memory_c_code.c -o data_parallel_dist_memory_c_code

# Run the MPI program across all requested processes
# mpirun automatically distributes across nodes based on SLURM allocation
mpirun data_parallel_dist_memory_c_code
