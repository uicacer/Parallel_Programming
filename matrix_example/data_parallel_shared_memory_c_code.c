// openmp_matrix_multiply.c  
// SHARED MEMORY PARALLELISM: Multiple threads share matrices, compute different rows

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    const int N = 6000;                    // Matrix dimension
    
    // =====================================================================
    // STEP 1: SHARED MEMORY ALLOCATION
    // All threads will access the same matrices (shared memory model)
    // =====================================================================
    
    double **A = malloc(N * sizeof(double*));
    double **B = malloc(N * sizeof(double*));
    double **C = malloc(N * sizeof(double*));
    
    for (int i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(double));
        B[i] = malloc(N * sizeof(double));
        C[i] = malloc(N * sizeof(double));
    }
    
    if (!A || !B || !C) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    // =====================================================================
    // STEP 2: MATRIX INITIALIZATION (done by master thread)
    // =====================================================================
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (i + j) % 100 + 1.0;
            B[i][j] = (i * j) % 100 + 1.0;
            C[i][j] = 0.0;
        }
    }
    
    // =====================================================================
    // STEP 3: PARALLEL MATRIX MULTIPLICATION WITH OPENMP
    // Each thread processes different rows of the outer loop
    // =====================================================================
    
    printf("Starting OpenMP matrix multiplication with %d threads...\n", omp_get_max_threads());
    double start_time = omp_get_wtime();
    
    // PARALLEL DIRECTIVE: Distribute outer loop iterations among threads
    #pragma omp parallel for schedule(static) shared(A,B,C)
    for (int i = 0; i < N; i++) {          // Each thread gets different values of i
        
        // THREAD WORK DISTRIBUTION:
        // With 4 threads: Thread 0 gets rows 0-499, Thread 1 gets 500-999, etc.
        // Each thread executes the inner loops for its assigned rows
        
        for (int j = 0; j < N; j++) {      
            C[i][j] = 0.0;                 
            
            // Inner loop: computational hotspot (same as sequential)
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
                // THREAD SAFETY: Each thread writes to different C[i][j] elements
                // No race conditions because threads handle different rows
            }
        }
    }
    // BARRIER: All threads synchronize here before continuing
    
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    
    // =====================================================================
    // STEP 4: VERIFICATION (master thread only)
    // =====================================================================
    
    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    
    double total_operations = 2.0 * N * N * N;
    double flops = total_operations / elapsed_time;
    
    printf("OpenMP Results:\n");
    printf("Time: %.4f seconds\n", elapsed_time);
    printf("Performance: %.1f GFLOPS\n", flops / 1e9);
    printf("Checksum: %.2f\n", checksum);
    
    // =====================================================================
    // STEP 5: CLEANUP
    // =====================================================================
    
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    return 0;
}
