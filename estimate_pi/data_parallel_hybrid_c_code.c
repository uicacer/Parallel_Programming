// Monte Carlo π estimation using hybrid MPI + OpenMP parallelism
// Multiple processes (MPI), each using multiple threads (OpenMP)

// =====================================================================
// HEADER FILES - COMBINING BOTH MPI AND OPENMP
// =====================================================================
#include <stdio.h>   // Standard I/O
#include <stdlib.h>  // Standard library
#include <mpi.h>     // MPI for distributed memory (between computers)
#include <omp.h>     // OpenMP for shared memory (within each computer)
#include <math.h>    // Math functions
#include <time.h>    // Time functions

int main(int argc, char** argv) {
    int rank, size;
    const long NUM_SAMPLES = 500000000;  // 500 million total samples
    
    // =====================================================================
    // HYBRID INITIALIZATION - MPI WITH THREAD SUPPORT
    // =====================================================================
    
    // HYBRID PROGRAMS REQUIRE SPECIAL MPI INITIALIZATION
    // Regular MPI_Init() is NOT sufficient when using OpenMP threads!
    
    int thread_support;  // Variable to store level of thread support MPI provides
    
    // MPI_Init_thread() - Special initialization for MPI + threads
    //
    // WHY DIFFERENT FROM MPI_Init()?
    // - MPI was originally designed assuming one process = one thread
    // - When mixing MPI with OpenMP, we have multiple threads per process
    // - MPI needs to know about threads to handle communication safely
    //
    // FOUR PARAMETERS:
    MPI_Init_thread(
        &argc, &argv,           // Command-line arguments (same as MPI_Init)
        
        MPI_THREAD_FUNNELED,    // THREAD LEVEL REQUESTED:
                                // This specifies how threads will use MPI
                                //
                                // MPI_THREAD_FUNNELED means:
                                // - Only the MAIN thread makes MPI calls
                                // - Worker threads do computation but no MPI
                                // - This is safest and most common for hybrid
                                //
                                // Other levels exist but are more complex:
                                // - MPI_THREAD_SINGLE: No threads at all
                                // - MPI_THREAD_SERIALIZED: Threads can call MPI
                                //   but only one at a time
                                // - MPI_THREAD_MULTIPLE: Any thread can call MPI
                                //   anytime (most complex, rarely needed)
        
        &thread_support         // OUTPUT: Actual thread support level provided
                                // MPI stores here what level it can actually support
                                // Might be less than what we requested!
    );
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // =====================================================================
    // HYBRID ARCHITECTURE INFORMATION
    // =====================================================================
    
    if (rank == 0) {
        printf("=== HYBRID MPI+OPENMP MONTE CARLO PI ESTIMATION ===\n");
        printf("Number of samples: %ld (%.1f million)\n", 
               NUM_SAMPLES, NUM_SAMPLES/1000000.0);
        
        // UNDERSTANDING THE HYBRID HIERARCHY:
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("Total cores: %d\n", size * omp_get_max_threads());
        
        // EXAMPLE ARCHITECTURE (4 MPI processes × 4 OpenMP threads):
        //
        // Computer 1: MPI Process 0 with 4 OpenMP threads
        // Computer 2: MPI Process 1 with 4 OpenMP threads
        // Computer 3: MPI Process 2 with 4 OpenMP threads
        // Computer 4: MPI Process 3 with 4 OpenMP threads
        //
        // Total: 16 cores working simultaneously
        //        4 processes (distributed across computers)
        //        16 threads (4 per computer)
    }
    
    // =====================================================================
    // TWO-LEVEL WORK DISTRIBUTION
    // =====================================================================
    
    // LEVEL 1: MPI divides work among processes (computers)
    long samples_per_process = NUM_SAMPLES / size;
    long my_samples = samples_per_process;
    
    if (rank == size - 1) {
        my_samples += NUM_SAMPLES % size;
    }
    
    // HYBRID WORK DISTRIBUTION EXAMPLE (500M samples, 4 processes, 4 threads):
    //
    // MPI LEVEL (between computers):
    // Process 0: samples 0 to 124,999,999 (125M samples)
    // Process 1: samples 125,000,000 to 249,999,999 (125M)
    // Process 2: samples 250,000,000 to 374,999,999 (125M)
    // Process 3: samples 375,000,000 to 499,999,999 (125M)
    //
    // OPENMP LEVEL (within each process, e.g., Process 0):
    // Thread 0: samples 0 to 31,249,999 (31.25M)
    // Thread 1: samples 31,250,000 to 62,499,999 (31.25M)
    // Thread 2: samples 62,500,000 to 93,749,999 (31.25M)
    // Thread 3: samples 93,750,000 to 124,999,999 (31.25M)
    //
    // This pattern repeats for each MPI process
    
    // =====================================================================
    // HYBRID PARALLEL COMPUTATION
    // =====================================================================
    
    double start_time = MPI_Wtime();
    
    long local_count = 0;  // Count for THIS MPI process (all its threads combined)
    
    // OPENMP PARALLEL REGION within each MPI process
    // This happens INDEPENDENTLY on each computer
    #pragma omp parallel reduction(+:local_count)
    {
        // ----------------------------------------------------------------
        // HYBRID RANDOM SEED STRATEGY - CRITICAL!
        // ----------------------------------------------------------------
        
        // CHALLENGE: Need UNIQUE seeds across BOTH dimensions:
        // 1. Different MPI processes (on different computers)
        // 2. Different OpenMP threads (within each process)
        //
        // SOLUTION: Combine rank, thread number, and time
        
        unsigned int seed = rank * 1000 +           // Separates MPI processes
                           omp_get_thread_num() +   // Separates threads within process
                           time(NULL);              // Adds time variation
        
        // WHY "rank * 1000"?
        // - Multiplying by 1000 creates large gaps between processes
        // - Ensures process 0's threads (0-7) don't overlap with process 1's (1000-1007)
        //
        // EXAMPLE SEEDS (time(NULL) = 1234567890):
        // Process 0, Thread 0: 0*1000 + 0 + 1234567890 = 1234567890
        // Process 0, Thread 1: 0*1000 + 1 + 1234567890 = 1234567891
        // Process 0, Thread 2: 0*1000 + 2 + 1234567890 = 1234567892
        // Process 0, Thread 3: 0*1000 + 3 + 1234567890 = 1234567893
        // Process 1, Thread 0: 1*1000 + 0 + 1234567890 = 1234568890
        // Process 1, Thread 1: 1*1000 + 1 + 1234567890 = 1234568891
        // ... and so on
        //
        // All 16 threads get UNIQUE seeds!
        
        // ----------------------------------------------------------------
        // NESTED PARALLEL LOOP
        // ----------------------------------------------------------------
        
        #pragma omp for
        for (long i = 0; i < my_samples; i++) {
            // EXECUTION HIERARCHY:
            // 1. Each MPI process executes this loop for its my_samples
            // 2. Within each process, OpenMP divides iterations among threads
            // 3. Each thread uses rand_r with its unique seed
            
            // Generate random point using thread-safe rand_r
            double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            
            // Check if inside circle
            double distance_squared = x * x + y * y;
            
            if (distance_squared <= 1.0) {
                local_count++;  // Thread-safe via reduction
            }
        }
        // OpenMP threads synchronize here (end of parallel for)
    }
    // OpenMP threads join here (end of parallel region)
    // 
    // AT THIS POINT:
    // - All threads within this process have finished
    // - local_count contains combined count from all threads in THIS process
    // - But we still need to combine counts from all MPI processes!
    
    // =====================================================================
    // TWO-LEVEL RESULT COMBINATION
    // =====================================================================
    
    // LEVEL 1 (already done): OpenMP reduction combined thread results
    //          within each process into local_count
    //
    // LEVEL 2 (now): MPI_Reduce combines local_count from all processes
    
    long global_count = 0;
    MPI_Reduce(&local_count,      // Each process's combined count
               &global_count,     // Final total (on rank 0)
               1, 
               MPI_LONG, 
               MPI_SUM, 
               0, 
               MPI_COMM_WORLD);
    
    // HYBRID REDUCTION EXAMPLE (conceptual flow):
    //
    // Computer 1 (Process 0):
    //   Thread 0 count: 24,543,123
    //   Thread 1 count: 24,541,876
    //   Thread 2 count: 24,542,991
    //   Thread 3 count: 24,543,456
    //   OpenMP reduction → local_count = 98,171,446
    //
    // Computer 2 (Process 1):
    //   Threads similarly combined → local_count = 98,169,234
    //
    // Computer 3 (Process 2):
    //   Threads similarly combined → local_count = 98,170,123
    //
    // Computer 4 (Process 3):
    //   Threads similarly combined → local_count = 98,168,897
    //
    // MPI_Reduce → global_count = 392,679,700 (on Process 0)
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // =====================================================================
    // DISPLAY RESULTS (only master process)
    // =====================================================================
    
    if (rank == 0) {
        double pi_estimate = 4.0 * global_count / NUM_SAMPLES;
        double error = fabs(pi_estimate - M_PI);
        
        printf("\nResults:\n");
        printf("Estimated π: %.10f\n", pi_estimate);
        printf("Actual π:    %.10f\n", M_PI);
        printf("Error:       %.10f\n", error);
        printf("Time:        %.4f seconds\n", max_time);
    }
    
    MPI_Finalize();
    return 0;
}

// =====================================================================
// KEY HYBRID CONCEPTS SUMMARY FOR STUDENTS:
// =====================================================================
//
// 1. TWO-LEVEL PARALLELISM
//    - MPI level: Parallelism BETWEEN computers (distributed memory)
//    - OpenMP level: Parallelism WITHIN each computer (shared memory)
//    - Combines benefits of both approaches
//
// 2. WHEN TO USE HYBRID
//    - Have access to multiple computers (cluster/supercomputer)
//    - Each computer has multiple cores
//    - Problem is large enough to benefit from both levels
//    - Want to maximize use of all available computational resources
//
// 3. COMPLEXITY TRADE-OFFS
//    Advantages:
//    - Uses ALL available cores efficiently
//    - Best possible performance for large problems
//    - Scales to thousands of cores
//    
//    Disadvantages:
//    - More complex to program (must understand both MPI and OpenMP)
//    - More complex to debug (two types of parallelism)
//    - Only beneficial when problem is large enough
//
// 4. INITIALIZATION DIFFERENCES
//    - Must use MPI_Init_thread() instead of MPI_Init()
//    - Must specify thread support level
//    - MPI needs to know about OpenMP threads
//
// 5. UNIQUE SEED STRATEGY
//    - Must ensure uniqueness across TWO dimensions
//    - Combine: rank (process ID) + thread_num (thread ID) + time
//    - Critical for correctness in hybrid programs
//
// 6. NESTED REDUCTIONS
//    - OpenMP reduction: combines threads within each process
//    - MPI reduction: combines processes across computers
//    - Two-stage process ensures all work is combined correctly
//
// 7. TYPICAL HYBRID ARCHITECTURE
//    - Cluster with N nodes (computers)
//    - Each node has M cores
//    - Use N MPI processes (one per node)
//    - Use M OpenMP threads (one per core on each node)
//    - Total: N × M cores working simultaneously
//
// 8. PERFORMANCE EXPECTATIONS
//    - Near-linear speedup if problem is large enough
//    - Best performance for problems requiring distributed memory
//    - May not show advantage over pure MPI for small problems
//
// =====================================================================
