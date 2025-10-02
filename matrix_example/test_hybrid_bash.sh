#!/bin/bash
# simple_hybrid_test.sh
#SBATCH --job-name=simple_hybrid_test
#SBATCH --time=00:05:00
#SBATCH --nodes=4                    # Use 4 separate computers
#SBATCH --ntasks-per-node=1         # 1 MPI process per computer
#SBATCH --cpus-per-task=4           # 4 OpenMP threads per MPI process
#SBATCH --account=ts_acer_chi
#SBATCH --partition=batch
#SBATCH --output=simple_hybrid_test_%j.out

# Move to submission directory
cd $SLURM_SUBMIT_DIR

# Setup MPI programming environment
module load OpenMPI/4.1.6-GCC-13.2.0 
module load UCX/1.16.0-GCCcore-13.3.0

# Configure OpenMPI communication settings
export OMPI_MCA_btl='^uct,ofi'
export OMPI_MCA_pml='ucx'
export OMPI_MCA_mtl='^ofi'

# Configure OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Print diagnostics
echo "=== SIMPLE HYBRID TEST DIAGNOSTICS ==="
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "Expected total processes = $SLURM_NTASKS"
echo "Expected threads per process = $OMP_NUM_THREADS"
echo "Expected total cores = $((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "========================================="

# Create the simple test file
cat > test_hybrid.c << 'EOF'
// Simple test file: test_hybrid.c
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int rank, size;
    int thread_support;
    char hostname[256];
    
    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get hostname to see which computer this is
    gethostname(hostname, 255);
    
    printf("=== PROCESS %d DIAGNOSTICS ===\n", rank);
    printf("Process %d running on host: %s\n", rank, hostname);
    printf("Process %d: Thread support level = %d\n", rank, thread_support);
    printf("Process %d: OpenMP max threads = %d\n", rank, omp_get_max_threads());
    printf("Process %d: Before OpenMP test\n", rank);
    
    // Test OpenMP parallel region
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Use critical section to avoid jumbled output
        #pragma omp critical
        {
            printf("Process %d (host %s), Thread %d of %d threads\n", 
                   rank, hostname, thread_id, num_threads);
        }
    }
    
    printf("Process %d: After OpenMP test\n", rank);
    
    // Test a simple parallel computation
    int n = 1000000;
    double sum = 0.0;
    
    printf("Process %d: Testing OpenMP computation...\n", rank);
    
    double start_time = MPI_Wtime();
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += (double)i * (double)i;
    }
    
    double end_time = MPI_Wtime();
    
    printf("Process %d: Computed sum = %f in %f seconds using %d threads\n", 
           rank, sum, end_time - start_time, omp_get_max_threads());
    
    MPI_Finalize();
    return 0;
}
EOF

echo "=== COMPILING SIMPLE HYBRID TEST ==="
mpicc -fopenmp -O2 test_hybrid.c -o test_hybrid

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "=== RUNNING SIMPLE HYBRID TEST ==="
mpirun test_hybrid

echo "=== SIMPLE HYBRID TEST COMPLETED ==="
