// openmp_pi.c
// Monte Carlo π estimation using OpenMP shared memory parallelism
// Multiple threads work on different samples simultaneously

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

int main() {
    const long NUM_SAMPLES = 500000000;  // 500 million samples
    long count_inside = 0;
    
    printf("=== OPENMP MONTE CARLO PI ESTIMATION ===\n");
    printf("Number of samples: %ld (%.1f million)\n", 
           NUM_SAMPLES, NUM_SAMPLES/1000000.0);
    printf("Number of threads: %d\n", omp_get_max_threads());
    
    // =====================================================================
    // OPENMP PARALLEL MONTE CARLO SIMULATION
    // Each thread processes different samples using its own random seed
    // =====================================================================
    
    double start_time = omp_get_wtime();
    
    // PARALLEL LOOP: Each thread processes a portion of samples
    #pragma omp parallel reduction(+:count_inside)
    {
        // Each thread needs its own random number generator seed
        unsigned int seed = omp_get_thread_num() + time(NULL);
        
        #pragma omp for
        for (long i = 0; i < NUM_SAMPLES; i++) {
            // Generate random point (x, y) using thread-safe rand_r
            double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            
            // Check if point is inside unit circle
            double distance_squared = x * x + y * y;
            
            if (distance_squared <= 1.0) {
                count_inside++;  // Safely accumulated via reduction
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    
    // =====================================================================
    // CALCULATE AND DISPLAY RESULTS
    // =====================================================================
    
    double pi_estimate = 4.0 * count_inside / NUM_SAMPLES;
    double error = fabs(pi_estimate - M_PI);
    
    printf("\nResults:\n");
    printf("Estimated π: %.10f\n", pi_estimate);
    printf("Actual π:    %.10f\n", M_PI);
    printf("Error:       %.10f\n", error);
    printf("Time:        %.4f seconds\n", elapsed_time);
    
    return 0;
}
