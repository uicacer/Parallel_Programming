// mpi_matrix_multiply.c
// DISTRIBUTED MEMORY PARALLELISM: Multiple processes on different computers

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;                        // MPI process ID and total process count
    const int N = 6000;                    // Matrix dimension
    
    // =====================================================================
    // STEP 1: MPI INITIALIZATION
    // Start MPI system and get process information
    // =====================================================================
    
    MPI_Init(&argc, &argv);                            // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);             // Get my process ID (0, 1, 2, ...)
    MPI_Comm_size(MPI_COMM_WORLD, &size);             // Get total number of processes
    
    // =====================================================================
    // STEP 2: WORK DISTRIBUTION CALCULATION
    // Divide matrix rows among MPI processes
    // =====================================================================
    
    // Calculate how many rows each process handles
    int rows_per_process = N / size;        // Base number of rows per process
    int start_row = rank * rows_per_process; // First row for this process
    int end_row = start_row + rows_per_process; // Last row + 1
    
    // Last process handles any remainder rows
    if (rank == size - 1) {
        end_row = N;
    }
    int local_rows = end_row - start_row;   // Number of rows this process handles
    
    // EXAMPLE with 4 processes:
    // Process 0: rows 0-499 (500 rows)
    // Process 1: rows 500-999 (500 rows)  
    // Process 2: rows 1000-1499 (500 rows)
    // Process 3: rows 1500-1999 (500 rows)
    
    // =====================================================================
    // STEP 3: DISTRIBUTED MEMORY ALLOCATION
    // Each process allocates memory for only its portion of the computation
    // =====================================================================
    
    // local_A: Only the rows this process will compute (local_rows × N)
    double **local_A = malloc(local_rows * sizeof(double*));
    
    // full_B: All processes need the complete B matrix (N × N)
    double **full_B = malloc(N * sizeof(double*));
    
    // local_C: Only the result rows this process computes (local_rows × N)  
    double **local_C = malloc(local_rows * sizeof(double*));
    
    // Allocate individual rows
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
    // STEP 4: MATRIX INITIALIZATION
    // Each process initializes its portion of A and the full B matrix
    // =====================================================================
    
    // Initialize my portion of matrix A
    for (int i = 0; i < local_rows; i++) {
        int global_row = start_row + i;     // Convert local row to global row index
        for (int j = 0; j < N; j++) {
            local_A[i][j] = (global_row + j) % 100 + 1.0;
        }
    }
    
    // Initialize complete B matrix (same on all processes)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            full_B[i][j] = (i * j) % 100 + 1.0;
        }
    }
    
    // Initialize result matrix
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
        }
    }
    
    // Synchronize all processes before computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // =====================================================================
    // STEP 5: DISTRIBUTED MATRIX MULTIPLICATION
    // Each process computes only its assigned rows of result matrix C
    // =====================================================================
    
    if (rank == 0) {
        printf("Starting MPI matrix multiplication with %d processes...\n", size);
    }
    
    double start_time = MPI_Wtime();
    
    // DISTRIBUTED COMPUTATION: Each process handles different rows
    for (int i = 0; i < local_rows; i++) {     // Process only my local rows
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
            
            // Inner loop: same computational pattern as sequential
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * full_B[k][j];
            }
        }
    }
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // =====================================================================
    // STEP 6: RESULT COLLECTION AND VERIFICATION
    // Combine partial results from all processes
    // =====================================================================
    
    // Each process computes checksum of its portion
    double local_checksum = 0.0;
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_checksum += local_C[i][j];
        }
    }
    
    // MPI COMMUNICATION: Combine checksums from all processes
    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum,      // What this process is sending
               &global_checksum,     // Where combined result goes (on rank 0)
               1,                    // Number of elements
               MPI_DOUBLE,           // Data type
               MPI_SUM,              // Operation (sum all contributions)
               0,                    // Destination process (rank 0)
               MPI_COMM_WORLD);      // All processes participate
    
    // Get maximum time across all processes (slowest process determines total time)
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // =====================================================================
    // STEP 7: RESULTS (only master process prints)
    // =====================================================================
    
    if (rank == 0) {
        double total_operations = 2.0 * N * N * N;
        double flops = total_operations / max_time;
        
        printf("MPI Results:\n");
        printf("Time: %.4f seconds\n", max_time);
        printf("Performance: %.1f GFLOPS\n", flops / 1e9);
        printf("Checksum: %.2f\n", global_checksum);
    }
    
    // =====================================================================
    // STEP 8: CLEANUP AND MPI FINALIZATION
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
    
    MPI_Finalize();                         // Shut down MPI system
    return 0;
}
