// Monte Carlo π estimation using OpenMP shared memory parallelism
// Multiple threads work on different samples simultaneously

// =====================================================================
// HEADER FILES (Libraries we need)
// =====================================================================
#include <stdio.h>   // Standard Input/Output: printf, etc.
#include <stdlib.h>  // Standard Library: rand(), RAND_MAX, etc.
#include <omp.h>     // OpenMP: Parallel programming functions
#include <math.h>    // Math functions: fabs() for absolute value
#include <time.h>    // Time functions: time() for random seed

int main() {
    // =====================================================================
    // PROBLEM SETUP
    // =====================================================================
    
    const long NUM_SAMPLES = 500000000;  // Total dart throws (500 million)
                                         // 'const' means this value cannot change
                                         // 'long' is a data type for large integers
    
    long count_inside = 0;               // Counter for darts inside circle
                                         // Initially zero, will be incremented
    
    // Print information about what we're doing
    printf("=== OPENMP MONTE CARLO PI ESTIMATION ===\n");
    printf("Number of samples: %ld (%.1f million)\n", 
           NUM_SAMPLES, NUM_SAMPLES/1000000.0);
    
    // omp_get_max_threads() is an OpenMP function that returns
    // how many threads (parallel workers) we can use
    printf("Number of threads: %d\n", omp_get_max_threads());
    
    // =====================================================================
    // TIMING: Start measuring how long computation takes
    // =====================================================================
    
    // omp_get_wtime() is OpenMP's wall-clock timer function
    // Returns current time in seconds as a decimal number
    double start_time = omp_get_wtime();
    
    // =====================================================================
    // OPENMP PARALLEL REGION: This is where parallelism happens
    // =====================================================================
    
    // #pragma omp parallel - This is a COMPILER DIRECTIVE (not regular C code)
    // It tells the compiler: "make this block run in parallel"
    // 
    // reduction(+:count_inside) - This is CRITICAL for correctness
    // What it means:
    // - Each thread gets its own PRIVATE copy of count_inside (starts at 0)
    // - Each thread counts independently (no conflicts)
    // - At the end, OpenMP AUTOMATICALLY adds all private counts together
    // - The final sum goes into the original count_inside variable
    //
    // WITHOUT reduction: threads would interfere with each other (race condition)
    // WITH reduction: threads work safely and results are combined correctly
    
    #pragma omp parallel reduction(+:count_inside)
    {
        // INSIDE THE PARALLEL REGION:
        // This code block is executed by MULTIPLE threads simultaneously
        // If you have 4 threads, this entire block runs 4 times at once
        
        // ----------------------------------------------------------------
        // THREAD-SPECIFIC RANDOM NUMBER SETUP
        // ----------------------------------------------------------------
        
        // IMPORTANT CONCEPT: Random Number Generators Need Unique Seeds
        // If all threads use the same seed, they generate IDENTICAL random numbers!
        // This would give us wrong results (not truly random sampling)
        
        // omp_get_thread_num() returns this thread's ID number (0, 1, 2, 3...)
        // time(NULL) returns current time in seconds since 1970 (a large number)
        // Adding them together gives each thread a UNIQUE seed
        unsigned int seed = omp_get_thread_num() + time(NULL);
        
        // EXAMPLE: If time(NULL) = 1234567890
        // Thread 0 gets seed = 0 + 1234567890 = 1234567890
        // Thread 1 gets seed = 1 + 1234567890 = 1234567891
        // Thread 2 gets seed = 2 + 1234567890 = 1234567892
        // Thread 3 gets seed = 3 + 1234567890 = 1234567893
        
        // ----------------------------------------------------------------
        // PARALLEL FOR LOOP: Divide work among threads
        // ----------------------------------------------------------------
        
        // #pragma omp for - Another compiler directive
        // Tells OpenMP: "divide loop iterations among threads"
        // 
        // HOW IT DIVIDES WORK (with 4 threads and 500 million samples):
        // Thread 0: iterations 0 to 124,999,999 (125 million samples)
        // Thread 1: iterations 125,000,000 to 249,999,999 (125 million)
        // Thread 2: iterations 250,000,000 to 374,999,999 (125 million)
        // Thread 3: iterations 375,000,000 to 499,999,999 (125 million)
        //
        // Each thread does its portion SIMULTANEOUSLY (in parallel)
        
        #pragma omp for
        for (long i = 0; i < NUM_SAMPLES; i++) {
            // EACH THREAD EXECUTES THIS LOOP BODY for its assigned iterations
            
            // ----------------------------------------------------------------
            // STEP 1: Generate random point coordinates
            // ----------------------------------------------------------------
            
            // rand_r(&seed) is a THREAD-SAFE random number generator
            //
            // ═══════════════════════════════════════════════════════════════
            // WHAT DOES "THREAD-SAFE" MEAN? (DETAILED EXPLANATION)
            // ═══════════════════════════════════════════════════════════════
            //
            // ANALOGY: Imagine a shared bank account vs individual accounts
            //
            // NOT THREAD-SAFE (like rand()):
            // - All threads share ONE random number generator (like one bank account)
            // - Generator has internal state that gets updated with each call
            // - When multiple threads call rand() simultaneously:
            //   
            //   Time 1: Thread 1 reads state → calculates → about to update state
            //   Time 2: Thread 2 reads SAME state → calculates → updates state
            //   Time 3: Thread 1 updates state (but based on OLD reading!)
            //   
            //   Result: STATE GETS CORRUPTED! Numbers aren't truly random anymore
            //   This is called a RACE CONDITION (threads "race" to update shared data)
            //
            // THREAD-SAFE (like rand_r()):
            // - Each thread has its OWN random number generator (like individual bank accounts)
            // - Each thread passes its own seed using &seed
            // - The '&' means "address of" - telling rand_r where THIS thread's seed is stored
            // - Each thread's seed is stored in a different memory location
            // - Threads never interfere with each other's seeds
            //
            // VISUAL COMPARISON:
            //
            // rand() - NOT SAFE:
            //   Thread 1 ──┐
            //   Thread 2 ──┼──► SHARED Generator ──► CONFLICTS!
            //   Thread 3 ──┘    (all updating same state)
            //
            // rand_r() - SAFE:
            //   Thread 1 ──► Generator with seed1 ──► No conflicts
            //   Thread 2 ──► Generator with seed2 ──► Independent
            //   Thread 3 ──► Generator with seed3 ──► Separate
            //
            // WHY THIS MATTERS:
            // - Using rand() would give wrong π estimates (corrupted randomness)
            // - Using rand_r() gives correct π estimates (true randomness per thread)
            //
            // SYNTAX: rand_r(&seed)
            // - rand_r is the function name
            // - &seed means "the memory address where my seed is stored"
            // - Each thread has 'seed' stored in different memory location
            // - So each thread's random numbers are independent
            //
            // ═══════════════════════════════════════════════════════════════
            
            // Now generate random x coordinate in range [-1, 1]
            // TRANSFORMATION BREAKDOWN:
            // rand_r(&seed) → random integer [0, RAND_MAX]
            // (double)rand_r(&seed) / RAND_MAX → convert to [0.0, 1.0]
            // * 2.0 → scale to [0.0, 2.0]
            // - 1.0 → shift to [-1.0, 1.0]
            
            double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            
            // Now we have a random point (x, y) somewhere in the square
            // from -1 to 1 on both axes
            
            // ----------------------------------------------------------------
            // STEP 2: Check if point is inside the unit circle
            // ----------------------------------------------------------------
            
            // Circle equation: x² + y² ≤ 1
            // If distance from origin ≤ 1, point is inside circle
            
            double distance_squared = x * x + y * y;
            
            // ----------------------------------------------------------------
            // STEP 3: Count if inside circle
            // ----------------------------------------------------------------
            
            if (distance_squared <= 1.0) {
                // Point is inside the circle!
                
                // count_inside++ means: count_inside = count_inside + 1
                // 
                // THREAD SAFETY: This is safe because of reduction(+:count_inside)
                // Each thread increments its OWN private copy
                // No conflicts between threads
                // At the end of parallel region, all counts are added together
                
                count_inside++;
            }
            // If distance_squared > 1.0, point is outside, don't count it
        }
        // End of parallel for loop
        
    }
    // End of parallel region
    // 
    // AUTOMATIC ACTIONS BY OPENMP AT THIS POINT:
    // 1. Wait for all threads to finish (barrier/synchronization)
    // 2. Add up all the private count_inside values from each thread
    // 3. Store the total in the original count_inside variable
    // 4. All threads terminate, back to single-threaded execution
    
    // =====================================================================
    // TIMING: Stop measuring time
    // =====================================================================
    
    double end_time = omp_get_wtime();           // Get end time
    double elapsed_time = end_time - start_time; // Calculate duration
    
    // =====================================================================
    // CALCULATE PI ESTIMATE
    // =====================================================================
    
    // Formula: π ≈ 4 × (points inside circle) / (total points)
    // 
    // Why 4.0 and not 4?
    // - 4.0 is a double (floating-point number)
    // - Using 4.0 ensures floating-point division (keeps decimal places)
    // - If we used 4, it would be integer division (truncates decimals)
    
    double pi_estimate = 4.0 * count_inside / NUM_SAMPLES;
    
    // Calculate error: How far off are we from actual π?
    // M_PI is a constant defined in <math.h> = 3.14159265358979323846...
    // fabs() returns absolute value (always positive)
    double error = fabs(pi_estimate - M_PI);
    
    // =====================================================================
    // DISPLAY RESULTS
    // =====================================================================
    
    printf("\nResults:\n");
    printf("Estimated π: %.10f\n", pi_estimate);  // %.10f = 10 decimal places
    printf("Actual π:    %.10f\n", M_PI);
    printf("Error:       %.10f\n", error);
    printf("Time:        %.4f seconds\n", elapsed_time);  // %.4f = 4 decimal places
    
    return 0;  // Program finished successfully
}

// =====================================================================
// KEY OPENMP CONCEPTS SUMMARY FOR STUDENTS:
// =====================================================================
//
// 1. #pragma omp parallel
//    - Creates multiple threads that run simultaneously
//    - Code inside {...} executes on all threads at once
//
// 2. #pragma omp for
//    - Divides loop iterations among threads
//    - Each thread gets a portion of iterations
//
// 3. reduction(+:variable)
//    - Each thread gets private copy of variable
//    - Threads work independently (no conflicts)
//    - At end, private copies are combined using specified operation (+)
//
// 4. Thread-safe functions
//    - rand_r() instead of rand() (each thread uses its own seed)
//    - Thread-safe = multiple threads can use it simultaneously without conflicts
//    - omp_get_thread_num() returns thread ID (0, 1, 2, ...)
//    - omp_get_max_threads() returns total number of threads
//    - omp_get_wtime() returns wall-clock time for timing
//
// 5. Shared vs Private variables
//    - Variables declared OUTSIDE parallel region = SHARED (all threads see same copy)
//    - Variables declared INSIDE parallel region = PRIVATE (each thread gets own copy)
//    - reduction() clause makes variable start private, then combines at end
//
// 6. Race Conditions
//    - Occur when multiple threads access shared data without protection
//    - Can cause incorrect results, crashes, or unpredictable behavior
//    - Avoided by: reduction(), private variables, or thread-safe functions
//
// =====================================================================
