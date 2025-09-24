// mpi_data_pipeline.c
// This demonstrates: DISTRIBUTED MEMORY + DATA PARALLELISM with optimal load balancing
// Multiple processes process different number ranges through identical pipeline

#include <mpi.h>     // MPI library for distributed memory programming
#include <stdio.h>   // Standard input/output functions
#include <stdlib.h>  // Memory allocation functions

int main(int argc, char **argv) {
    int rank, size;                          // rank = my process ID, size = total processes
    const int ARRAY_SIZE = 10000000;          // 10 million numbers for testing (change to 160M for production)
    const int THRESHOLD = 50;
    
    // STEP 1: Initialize MPI distributed memory system
    MPI_Init(&argc, &argv);                  // Start MPI system - this MUST be the first MPI call
                                             // &argc, &argv: Pass addresses so MPI can read command-line args
                                             // MPI uses these to configure runtime (process count, hostnames, etc.)
                                             // After this call, MPI runtime is active and this process can communicate
                                             
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get my unique process ID number (rank)
                                             // MPI_COMM_WORLD: The "communicator" containing all MPI processes
                                             // &rank: Address where MPI will store my ID (0, 1, 2, 3...)
                                             // Rank 0 is typically the "master", others are "workers"
                                             // Each process gets different rank, so they can do different work
                                             
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get total number of processes in the job
                                             // MPI_COMM_WORLD: Same communicator as above (all processes)
                                             // &size: Address where MPI will store the total count
                                             // If you launched with "mpirun -np 4", then size = 4
                                             // This tells each process how many others exist
    
    // Only master process (rank 0) prints header information
    if (rank == 0) {
        printf("=== MPI DATA PARALLEL PIPELINE ===\n");
        printf("Using %d processes on %d separate computers\n", size, size);
        printf("Processing %d numbers (%.1f million)\n", ARRAY_SIZE, ARRAY_SIZE/1000000.0);
    }
    
    // STEP 2: Calculate work division with optimal load balancing
    int base_numbers_per_process = ARRAY_SIZE / size;       // Base amount each process gets
                                                            // Example: 4,000,000 ÷ 4 = 1,000,000 per process
                                                            
    int remainder = ARRAY_SIZE % size;                      // How many numbers are left over  
                                                            // Example: 4,000,000 % 4 = 0 (no remainder)
                                                            
    int my_count = base_numbers_per_process;                // Start with base amount for this process
    if (rank < remainder) {                                 // First 'remainder' processes get +1 extra number
        my_count++;                                         // Only applies if there's a remainder
    }
    
    // Calculate start position - this is where the previous bug was!
    // We need to calculate the exact starting number for this process
    int my_start = rank * base_numbers_per_process;         // Simple calculation works when no remainder
    if (rank < remainder) {                                 // If this process gets extra work
        my_start += rank;                                   // Add how many extra numbers previous processes got
    } else {                                                // If this process doesn't get extra work  
        my_start += remainder;                              // Add total extra numbers given to earlier processes
    }
    
    // DEBUG: Print work division for verification
    printf("DEBUG Process %d: my_start=%d, my_count=%d, numbers %d to %d\n", 
           rank, my_start, my_count, my_start + 1, my_start + my_count);
    
    // STEP 3: Each process allocates memory for its portion only
    int *numbers = malloc(my_count * sizeof(int));         // Use int for input numbers
    
    if (!numbers) {
        printf("Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);  // Stop all processes across all computers if one fails
    }
    
    // Each process initializes its own portion of the number sequence
    for (int i = 0; i < my_count; i++) {                   // Generate only my assigned numbers
        numbers[i] = my_start + i + 1;                     // Convert from 0-based to 1-based numbering
                                                           // Process 0: generates 1, 2, 3, ..., 1,000,000
                                                           // Process 1: generates 1,000,001, 1,000,002, ..., 2,000,000
                                                           // Process 2: generates 2,000,001, 2,000,002, ..., 3,000,000
                                                           // Process 3: generates 3,000,001, 3,000,002, ..., 4,000,000
    }
    
    // Start timing (MPI provides its own timing function)
    double start_time = MPI_Wtime();  // MPI_Wtime() returns wall clock time in seconds
    
    // STEP 4: Each process processes its numbers through the complete pipeline
    double total_sum = 0.0;              // Each process's partial sum (use double to avoid overflow)
    int count_above_threshold = 0;       // Each process's partial count
    
    for (int i = 0; i < my_count; i++) {                   // Process only my assigned numbers
        // Each process runs the IDENTICAL pipeline on DIFFERENT data
        
        // PIPELINE STAGE 1: Square the number
        double squared = (double)numbers[i] * (double)numbers[i];  // Use double arithmetic to avoid overflow
        
        // PIPELINE STAGE 2: Add 10
        double plus_ten = squared + 10.0;
        
        // ADDITIONAL COMPUTATIONAL WORK (consistent with all versions)
        for (int work = 0; work < 1000; work++) {
            plus_ten = plus_ten + (work % 3) - 1;
        }
        
        // PIPELINE STAGE 3: Add to this process's partial sum
        total_sum += plus_ten;
        
        // PIPELINE STAGE 4: Count if above threshold
        if (plus_ten > THRESHOLD) {
            count_above_threshold++;
        }
    }
    
    // DEBUG: Print partial results before combining
    printf("DEBUG Process %d: partial_sum=%.0f, partial_count=%d\n", 
           rank, total_sum, count_above_threshold);
    
    // STEP 5: DISTRIBUTED MEMORY COMMUNICATION - combine results from all computers
    double global_sum = 0.0;         // Final combined sum (use double, only meaningful on computer with rank 0)
    int global_count = 0;            // Final combined count (only meaningful on computer with rank 0)
    
    // MPI_Reduce: All processes send their partial results over network to process 0
    MPI_Reduce(&total_sum,               // What I'm sending (my local sum)
               &global_sum,              // Where combined result goes (only valid on root computer)
               1,                        // Sending 1 number
               MPI_DOUBLE,               // Data type is double (changed from MPI_LONG_LONG to avoid overflow)
               MPI_SUM,                  // Operation: add all partial sums together
               0,                        // Send results to process 0 (master computer)
               MPI_COMM_WORLD);          // All computers participate in this communication
    
    MPI_Reduce(&count_above_threshold,   // What I'm sending (my local count)
               &global_count,            // Where combined result goes
               1,                        // Sending 1 number  
               MPI_INT,                  // Data type is int
               MPI_SUM,                  // Operation: add all partial counts
               0,                        // Send to computer with rank 0
               MPI_COMM_WORLD);          // All computers participate in this communication
    
    double end_time = MPI_Wtime();   // Get end time using MPI timing function
    double elapsed_time = end_time - start_time;
    
    // STEP 6: Only master process (rank 0) displays final results
    if (rank == 0) {
        printf("\n=== RESULTS ===\n");
        printf("Global sum: %.0f\n", global_sum);        // Use %.0f for double
        printf("Global count > %d: %d\n", THRESHOLD, global_count);
        printf("Processing time: %.4f seconds\n", elapsed_time);
        printf("Numbers processed: %d (%.1f million)\n", ARRAY_SIZE, ARRAY_SIZE/1000000.0);
        
        // Display load balancing info - FIXED VERSION
        printf("\n=== LOAD BALANCING ===\n");
        printf("Base numbers per process: %d\n", base_numbers_per_process);
        printf("Remainder to distribute: %d\n", remainder);
        if (remainder > 0) {
            printf("Processes 0-%d handle: %d numbers each\n", 
                   remainder-1, base_numbers_per_process + 1);
            printf("Processes %d-%d handle: %d numbers each\n", 
                   remainder, size-1, base_numbers_per_process);
        } else {
            printf("All processes 0-%d handle: %d numbers each\n", 
                   size-1, base_numbers_per_process);
        }
    }
    
    // STEP 7: Cleanup
    free(numbers);               // Free the memory allocated for my portion
    MPI_Finalize();              // Shut down MPI system (must be called before program ends)
    return 0;
}
