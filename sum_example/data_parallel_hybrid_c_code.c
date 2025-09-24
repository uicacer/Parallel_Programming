// hybrid_data_pipeline.c
// This demonstrates: DISTRIBUTED MEMORY (MPI) + SHARED MEMORY (OpenMP) + DATA PARALLELISM
// Multiple processes, each using multiple threads, all processing different data

#include <mpi.h>     // MPI library for distributed memory programming
#include <omp.h>     // OpenMP library for shared memory programming  
#include <stdio.h>   // Standard input/output functions
#include <stdlib.h>  // Memory allocation functions

int main(int argc, char **argv) {
    int rank, size;                          // MPI variables: rank and size
    const int ARRAY_SIZE = 10000000;         // 10 million numbers (same as other versions)
    const int THRESHOLD = 50;

    // STEP 1: Initialize MPI with thread support for hybrid programming
    int thread_support;                      // Variable to store the level of thread support provided
    MPI_Init_thread(&argc, &argv,           // Initialize with threading support
                    MPI_THREAD_FUNNELED,    // Thread level: only main thread makes MPI calls
                    &thread_support);       // Returns the actual thread support level provided

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get my process ID (0, 1, 2, 3...)
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    if (rank == 0) {
        printf("=== HYBRID PARALLEL PIPELINE (MPI + OpenMP) ===\n");
        printf("Using %d computers × %d threads = %d total cores\n",
               size, omp_get_max_threads(), size * omp_get_max_threads());
        printf("Processing %d numbers (%.1f million)\n", ARRAY_SIZE, ARRAY_SIZE/1000000.0);

        // DIAGNOSTIC: Check if OpenMP is actually working
        printf("DIAGNOSTIC: OpenMP max threads = %d\n", omp_get_max_threads());
        if (omp_get_max_threads() == 1) {
            printf("WARNING: OpenMP is NOT working! Only 1 thread detected.\n");
            printf("Check: 1) OMP_NUM_THREADS environment variable\n");
            printf("       2) Compilation with -fopenmp flag\n");
            printf("       3) SLURM --cpus-per-task setting\n");
        }
    }

    // STEP 2: MPI LEVEL - Divide work among computers with optimal load balancing
    int base_per_process = ARRAY_SIZE / size;
    int remainder = ARRAY_SIZE % size;
    
    int my_count = base_per_process + (rank < remainder ? 1 : 0);
    int my_start = rank * base_per_process + (rank < remainder ? rank : remainder);

    // STEP 3: Each computer allocates its own private memory
    int *numbers = malloc(my_count * sizeof(int));  // Use int for input numbers
    if (!numbers) {
        printf("Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Each process initializes its portion of the number sequence
    for (int i = 0; i < my_count; i++) {
        numbers[i] = my_start + i + 1;
    }

    // STEP 4: Start timing
    double start_time = MPI_Wtime();

    // STEP 5: HYBRID COMPUTATION - MPI + OpenMP Data Parallelism
    double total_sum = 0.0;               // Use double to match other versions
    int count_above_threshold = 0;

    // OpenMP DATA PARALLELISM within each computer
    #pragma omp parallel for reduction(+:total_sum,count_above_threshold)
    for (int i = 0; i < my_count; i++) {
        // PIPELINE STAGE 1: Square the number
        double squared = (double)numbers[i] * (double)numbers[i];  // Use double arithmetic
        
        // PIPELINE STAGE 2: Add 10 to the squared result  
        double plus_ten = squared + 10.0;
        
        // ADDITIONAL COMPUTATIONAL WORK to slow down the pipeline
        // Simulate more complex mathematical operations
        for (int work = 0; work < 1000; work++) {
            plus_ten = plus_ten + (work % 3) - 1;
        }
        
        // PIPELINE STAGE 3: Add to running sum
        total_sum += plus_ten;
        
        // PIPELINE STAGE 4: Count if above threshold
        if (plus_ten > THRESHOLD) {
            count_above_threshold++;
        }
    }

    // STEP 6: MPI LEVEL - Combine results from all computers
    double global_sum = 0.0;                // Use double for global sum
    int global_count = 0;

    // Each computer sends its results to the master computer
    MPI_Reduce(&total_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);  // Use MPI_DOUBLE
    MPI_Reduce(&count_above_threshold, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // STEP 7: Display results (only master computer)
    if (rank == 0) {
        printf("\n=== RESULTS ===\n");
        printf("Total sum: %.0f\n", global_sum);  // Use %.0f for double
        printf("Count > %d: %d\n", THRESHOLD, global_count);
        printf("Processing time: %.4f seconds\n", elapsed_time);
        printf("Numbers processed: %d (%.1f million)\n", ARRAY_SIZE, ARRAY_SIZE/1000000.0);
    }

    // STEP 8: Cleanup
    free(numbers);
    MPI_Finalize();
    return 0;
}
