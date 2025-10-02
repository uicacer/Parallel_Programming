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
