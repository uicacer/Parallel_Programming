// data_parallel_shared_memory_c_code.c
// Monte Carlo π estimation using OpenMP (Shared Memory Parallelism)
// Distributes work across multiple CPU cores on a SINGLE computer

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>        // OpenMP library for parallel processing

/*
 * OPENMP PARALLEL PROGRAMMING MODEL:
 * 
 * What is OpenMP?
 * - Open Multi-Processing: API for shared-memory parallel programming
 * - Uses multiple THREADS on a single computer
 * - All threads share the same memory space (can access same variables)
 * - Ideal for parallelizing loops and independent computations
 * 
 * Key Concepts:
 * - THREADS: Lightweight execution units within a single process
 * - SHARED MEMORY: All threads can read/write to the same memory
 * - PARALLEL REGION: Section of code executed by multiple threads
 * - REDUCTION: Combining results from multiple threads safely
 * 
 * This program parallelizes Monte Carlo π estimation:
 * - Each thread generates its own random points
 * - Each thread counts points in its portion of work
 * - Results are combined at the end (reduction)
 */

int main() {
    // =====================================================================
    // CONFIGURATION
    // =====================================================================
    
    const long NUM_SAMPLES = 500000000;  // 500 million samples
                                         // Same total work as sequential version
                                         // Will be divided among threads
    
    long count_inside = 0;               // Total points inside circle
                                         // Will be updated by multiple threads
                                         // (using atomic operations or reduction)

    printf("=== OPENMP PARALLEL MONTE CARLO PI ESTIMATION ===\n");
    printf("Number of samples: %ld (%.1f million)\n",
           NUM_SAMPLES, NUM_SAMPLES/1000000.0);

    // =====================================================================
    // OPENMP CONFIGURATION
    // =====================================================================
    
    // Get the number of threads from environment variable OMP_NUM_THREADS
    // This is typically set in the SLURM script or system environment
    // Example: export OMP_NUM_THREADS=4
    int num_threads = omp_get_max_threads();
    printf("Number of OpenMP threads: %d\n", num_threads);
    printf("Samples per thread: ~%ld\n\n", NUM_SAMPLES / num_threads);

    // =====================================================================
    // PARALLEL MONTE CARLO SIMULATION
    // This is where the magic happens!
    // =====================================================================

    double start_time = omp_get_wtime();  // OpenMP's high-precision wall clock timer

    // ┌─────────────────────────────────────────────────────────────────┐
    // │ OPENMP PARALLEL REGION                                          │
    // │                                                                  │
    // │ #pragma omp parallel: Creates a team of threads                │
    // │ - Each thread executes the code block independently             │
    // │ - Threads share the same memory space                           │
    // │ - Work is distributed automatically                             │
    // │                                                                  │
    // │ reduction(+:count_inside): Safely combines results              │
    // │ - Each thread has its own private copy of count_inside          │
    // │ - At the end, all copies are summed together                    │
    // │ - This avoids race conditions (multiple threads writing         │
    // │   to the same variable simultaneously)                          │
    // └─────────────────────────────────────────────────────────────────┘
    
    #pragma omp parallel reduction(+:count_inside)
    {
        // ================================================================
        // THREAD-LOCAL VARIABLES
        // Each thread gets its own copy of these variables
        // ================================================================
        
        // Get this thread's unique ID (0, 1, 2, 3, etc.)
        int thread_id = omp_get_thread_num();
        
        // Seed random number generator differently for each thread
        // This ensures different random sequences per thread
        // Using thread_id + time ensures uniqueness
        unsigned int seed = time(NULL) + thread_id;
        
        // ================================================================
        // PARALLEL FOR LOOP
        // Work is automatically divided among threads
        // ================================================================
        
        // #pragma omp for: Distributes loop iterations across threads
        // - OpenMP automatically divides NUM_SAMPLES iterations
        // - Each thread processes its assigned chunk
        // - schedule(static): Divides iterations into equal-sized chunks
        //   Example with 4 threads and 1000 iterations:
        //   - Thread 0: iterations 0-249
        //   - Thread 1: iterations 250-499
        //   - Thread 2: iterations 500-749
        //   - Thread 3: iterations 750-999
        
        #pragma omp for schedule(static)
        for (long i = 0; i < NUM_SAMPLES; i++) {
            
            // Generate random point (x, y) in range [-1, 1] × [-1, 1]
            // Using rand_r() instead of rand() for thread-safety
            // rand_r() takes a seed pointer, allowing each thread
            // to maintain its own random number sequence
            double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;

            // Calculate squared distance from origin
            // Point is inside unit circle if x² + y² ≤ 1
            double distance_squared = x * x + y * y;

            // Check if point is inside circle
            if (distance_squared <= 1.0) {
                count_inside++;  // Safe because of reduction clause
                                // Each thread increments its private copy
            }
        }
        
        // ================================================================
        // IMPLICIT BARRIER
        // All threads wait here until everyone finishes their work
        // Then OpenMP automatically combines all count_inside values
        // ================================================================
        
    } // End of parallel region - reduction happens here automatically

    double end_time = omp_get_wtime();  // Stop timing

    // =====================================================================
    // CALCULATE AND DISPLAY RESULTS
    // =====================================================================

    double elapsed_time = end_time - start_time;

    // Estimate π using Monte Carlo formula
    double pi_estimate = 4.0 * count_inside / NUM_SAMPLES;

    // Calculate error
    double error = fabs(pi_estimate - M_PI);
    double percent_error = (error / M_PI) * 100.0;

    // =====================================================================
    // OUTPUT RESULTS
    // =====================================================================
    
    printf("=== RESULTS ===\n");
    printf("Points inside circle: %ld\n", count_inside);
    printf("Total points tested:  %ld\n", NUM_SAMPLES);
    printf("Ratio (inside/total): %.10f\n", (double)count_inside / NUM_SAMPLES);
    printf("\n");
    printf("Estimated π:          %.10f\n", pi_estimate);
    printf("Actual π:             %.10f\n", M_PI);
    printf("Absolute error:       %.10f\n", error);
    printf("Percent error:        %.6f%%\n", percent_error);
    printf("\n");
    printf("Computation time:     %.2f seconds\n", elapsed_time);
    printf("Samples per second:   %.2f million\n", 
           (NUM_SAMPLES / elapsed_time) / 1000000.0);
    printf("\n");
    printf("Speedup vs sequential: Compare with sequential_results.txt\n");
    printf("Efficiency:            Speedup / num_threads = parallel efficiency\n");

    return 0;
}

/*
 * ============================================================================
 * OPENMP COMPILATION AND EXECUTION
 * ============================================================================
 * 
 * Compilation:
 *   gcc -fopenmp -o program data_parallel_shared_memory_c_code.c -lm
 *   
 *   -fopenmp: Enables OpenMP support
 *   -lm: Links math library (for M_PI constant)
 * 
 * Execution:
 *   export OMP_NUM_THREADS=4    # Set number of threads
 *   ./program
 * 
 * Or in SLURM script:
 *   #SBATCH --cpus-per-task=4
 *   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
 *   ./program
 * 
 * ============================================================================
 * KEY OPENMP DIRECTIVES EXPLAINED
 * ============================================================================
 * 
 * #pragma omp parallel
 *   - Creates a team of threads
 *   - All threads execute the code block
 *   - Number of threads set by OMP_NUM_THREADS
 * 
 * reduction(+:variable)
 *   - Each thread gets private copy of variable
 *   - At end, all copies are combined using + operator
 *   - Thread-safe way to accumulate results
 *   - Avoids race conditions
 * 
 * #pragma omp for
 *   - Distributes loop iterations across threads
 *   - Must be inside a parallel region
 *   - Automatic load balancing
 * 
 * schedule(static)
 *   - Divides iterations into equal-sized chunks at compile time
 *   - Each thread gets a contiguous block of iterations
 *   - Good for uniform workload (like this program)
 *   - Alternative: schedule(dynamic) for uneven workloads
 * 
 * ============================================================================
 * THREAD SAFETY CONSIDERATIONS
 * ============================================================================
 * 
 * Why rand_r() instead of rand()?
 *   - rand() uses global state (not thread-safe)
 *   - Multiple threads calling rand() simultaneously causes race conditions
 *   - rand_r() takes a seed pointer, each thread has its own seed
 *   - Result: Each thread has independent random number sequence
 * 
 * Why reduction for count_inside?
 *   - Without reduction: Multiple threads writing to same variable = race condition
 *   - Race condition example:
 *     Thread 1 reads count_inside (100)
 *     Thread 2 reads count_inside (100) ← Same value!
 *     Thread 1 writes 101
 *     Thread 2 writes 101 ← Should be 102!
 *   - Reduction creates private copies, then safely combines them
 * 
 * ============================================================================
 * PERFORMANCE EXPECTATIONS
 * ============================================================================
 * 
 * Ideal speedup with N threads: N× faster than sequential
 *   - 4 threads: 4× speedup
 *   - 8 threads: 8× speedup
 * 
 * Reality: Usually 0.8N to 0.95N speedup due to:
 *   - Thread creation/synchronization overhead
 *   - Memory bandwidth limitations
 *   - Cache effects
 * 
 * This algorithm is "embarrassingly parallel":
 *   - No dependencies between iterations
 *   - Minimal communication between threads
 *   - Should achieve near-linear speedup
 * 
 * Example with 4 threads on 4 cores:
 *   - Sequential time: 10 seconds
 *   - Parallel time: ~2.5-2.7 seconds
 *   - Speedup: 3.7-4.0× (excellent!)
 * 
 * ============================================================================
 * SHARED MEMORY vs DISTRIBUTED MEMORY
 * ============================================================================
 * 
 * OpenMP (Shared Memory) - THIS PROGRAM:
 *   ✓ Multiple threads on ONE computer
 *   ✓ All threads share same memory
 *   ✓ Fast communication (direct memory access)
 *   ✓ Limited by single machine's resources
 *   ✓ Easiest to program and debug
 *   - Best for: Single-node parallelism
 * 
 * MPI (Distributed Memory) - See dist_memory version:
 *   ✓ Multiple processes across MULTIPLE computers
 *   ✓ Each process has separate memory
 *   ✓ Communication via message passing (slower)
 *   ✓ Can scale to thousands of nodes
 *   ✓ More complex programming model
 *   - Best for: Multi-node, large-scale parallelism
 * 
 * Hybrid (MPI + OpenMP) - See hybrid version:
 *   ✓ MPI processes across multiple nodes
 *   ✓ OpenMP threads within each node
 *   ✓ Best of both worlds
 *   - Best for: Maximum performance on clusters
 */
