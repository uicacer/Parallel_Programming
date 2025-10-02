// hybrid_matrix_multiply.c
// HYBRID PARALLELISM: MPI processes + OpenMP threads within each process

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    const int N = 6000;
    
    // =====================================================================
    // STEP 1: MPI INITIALIZATION WITH THREAD SUPPORT
    // Enable MPI to work with OpenMP threads
    // =====================================================================
    
    int thread_support;
    MPI_Init_thread(&argc, &argv,           // Initialize MPI with threading
                    MPI_THREAD_FUNNELED,    // Only main thread makes MPI calls
                    &thread_support);       // Actual thread support level
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Verify threading support
    if (thread_support < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            printf("Error: MPI does not support threading\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // =====================================================================
    // STEP 2: HYBRID WORK DISTRIBUTION
    // MPI level: Divide rows among processes
    // OpenMP level: Each process will further divide its rows among threads
    // =====================================================================
    
    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;
    if (rank == size - 1) {
        end_row = N;
    }
    int local_rows = end_row - start_row;
    
    // HYBRID HIERARCHY EXAMPLE (4 MPI processes × 8 OpenMP threads = 32 cores):
    // MPI Process 0: rows 0-499, divided among 8 threads (each thread: ~62 rows)
    // MPI Process 1: rows 500-999, divided among 8 threads  
    // MPI Process 2: rows 1000-1499, divided among 8 threads
    // MPI Process 3: rows 1500-1999, divided among 8 threads
    
    // =====================================================================
    // STEP 3: DISTRIBUTED MEMORY ALLOCATION (same as MPI version)
    // =====================================================================
    
    double **local_A = malloc(local_rows * sizeof(double*));
    double **full_B = malloc(N * sizeof(double*));
    double **local_C = malloc(local_rows * sizeof(double*));
    
    for (int i = 0; i < local_rows; i++) {
        local_A[i] = malloc(N * sizeof(double));
        local_C[i] = malloc(N * sizeof(double));
    }
    for (int i = 0; i < N; i++) {
        full_B[i] = malloc(N * sizeof(double));
    }
    
    if (!local_A || !full_B || !local_C) {
        printf("Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // =====================================================================
    // STEP 4: MATRIX INITIALIZATION (same as MPI version)
    // =====================================================================
    
    for (int i = 0; i < local_rows; i++) {
        int global_row = start_row + i;
        for (int j = 0; j < N; j++) {
            local_A[i][j] = (global_row + j) % 100 + 1.0;
        }
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            full_B[i][j] = (i * j) % 100 + 1.0;
        }
    }
    
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // =====================================================================
    // STEP 5: HYBRID COMPUTATION - MPI + OPENMP
    // Two-level parallelism: MPI distributes among processes, 
    // OpenMP distributes within each process
    // =====================================================================
    
    if (rank == 0) {
        printf("Starting hybrid matrix multiplication: %d processes × %d threads = %d cores...\n",
               size, omp_get_max_threads(), size * omp_get_max_threads());
    }
    
    double start_time = MPI_Wtime();
    
    // HYBRID PARALLEL COMPUTATION
    #pragma omp parallel for schedule(static) shared(local_A,full_B,local_C)
    for (int i = 0; i < local_rows; i++) {     // OpenMP threads divide local_rows
        
        // THREAD WORK DISTRIBUTION within each MPI process:
        // If process has 500 rows and 8 threads, each thread gets ~62 rows
        // Thread 0: rows 0-62, Thread 1: rows 63-125, etc.
        
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
            
            // Inner loop: computational hotspot (same algorithm)
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * full_B[k][j];
                // HYBRID SAFETY: Different MPI processes write to different memory
                // Different OpenMP threads write to different rows within process
            }
        }
    }
    // HYBRID SYNCHRONIZATION: 
    // OpenMP threads synchronize here (within each process)
    // MPI processes continue independently
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // =====================================================================
    // STEP 6: HYBRID RESULT COLLECTION
    // OpenMP reduction within each process, then MPI reduction across processes
    // =====================================================================
    
    // OPENMP REDUCTION: Combine results from threads within this process
    double local_checksum = 0.0;
    #pragma omp parallel for reduction(+:local_checksum)
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_checksum += local_C[i][j];
        }
    }
    
    // MPI REDUCTION: Combine results from all processes  
    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // =====================================================================
    // STEP 7: RESULTS (master process only)
    // =====================================================================
    
    if (rank == 0) {
        double total_operations = 2.0 * N * N * N;
        double flops = total_operations / max_time;
        
        printf("Hybrid Results:\n");
        printf("Time: %.4f seconds\n", max_time);
        printf("Performance: %.1f GFLOPS\n", flops / 1e9);
        printf("Checksum: %.2f\n", global_checksum);
    }
    
    // =====================================================================
    // STEP 8: CLEANUP
    // =====================================================================
    
    for (int i = 0; i < local_rows; i++) {
        free(local_A[i]);
        free(local_C[i]);
    }
    for (int i = 0; i < N; i++) {
        free(full_B[i]);
    }
    free(local_A);
    free(full_B);
    free(local_C);
    
    MPI_Finalize();
    return 0;
}
