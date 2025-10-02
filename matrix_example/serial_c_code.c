// sequential_matrix_multiply.c
// SEQUENTIAL PROCESSING: One CPU core computes entire matrix multiplication

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    const int N = 6000;                    // Matrix dimension (2000 × 2000)
    
    // =====================================================================
    // STEP 1: MEMORY ALLOCATION
    // Create three 2D matrices: A (input), B (input), C (result)
    // =====================================================================
    
    // Allocate array of pointers (one for each row)
    double **A = malloc(N * sizeof(double*));
    double **B = malloc(N * sizeof(double*));
    double **C = malloc(N * sizeof(double*));
    
    // Allocate memory for each row
    for (int i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(double));  // Row i of matrix A
        B[i] = malloc(N * sizeof(double));  // Row i of matrix B  
        C[i] = malloc(N * sizeof(double));  // Row i of matrix C (result)
    }
    
    // Check if memory allocation succeeded
    if (!A || !B || !C) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    // =====================================================================
    // STEP 2: MATRIX INITIALIZATION
    // Fill matrices A and B with test data, initialize C to zero
    // =====================================================================
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Initialize with simple patterns for reproducible results
            A[i][j] = (i + j) % 100 + 1.0;      // Values 1-100
            B[i][j] = (i * j) % 100 + 1.0;      // Values 1-100
            C[i][j] = 0.0;                      // Result matrix starts at zero
        }
    }
    
    // =====================================================================
    // STEP 3: MATRIX MULTIPLICATION ALGORITHM
    // Triple nested loop: C[i][j] = sum(A[i][k] * B[k][j]) for all k
    // =====================================================================
    
    printf("Starting sequential matrix multiplication...\n");
    clock_t start_time = clock();
    
    // OUTER LOOP: Process each row of result matrix C
    for (int i = 0; i < N; i++) {
        
        // MIDDLE LOOP: Process each column of result matrix C  
        for (int j = 0; j < N; j++) {
            
            C[i][j] = 0.0;  // Initialize this result element
            
            // INNER LOOP: Compute dot product of row i of A with column j of B
            // This is where most computation happens (N operations per element)
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];   // Multiply and accumulate
                // This line executes N³ times total (2000³ = 8 billion times)
            }
        }
    }
    
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    // =====================================================================
    // STEP 4: VERIFICATION AND PERFORMANCE MEASUREMENT
    // =====================================================================
    
    // Compute checksum to verify correctness
    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    
    // Calculate and display performance metrics
    double total_operations = 2.0 * N * N * N;  // Each C[i][j] needs N multiply-adds
    double flops = total_operations / elapsed_time;
    
    printf("Sequential Results:\n");
    printf("Time: %.4f seconds\n", elapsed_time);
    printf("Performance: %.1f GFLOPS\n", flops / 1e9);
    printf("Checksum: %.2f\n", checksum);
    
    // =====================================================================
    // STEP 5: CLEANUP MEMORY
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
