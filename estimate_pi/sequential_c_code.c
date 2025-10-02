// Monte Carlo π estimation using sequential processing
// Throws random darts and counts how many fall inside a unit circle

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main() {
    const long NUM_SAMPLES = 500000000;  // 500 million samples for accuracy
    long count_inside = 0;               // Count of points inside circle
    
    printf("=== SEQUENTIAL MONTE CARLO PI ESTIMATION ===\n");
    printf("Number of samples: %ld (%.1f million)\n", 
           NUM_SAMPLES, NUM_SAMPLES/1000000.0);
    
    // =====================================================================
    // RANDOM NUMBER GENERATOR SETUP
    // =====================================================================
    
    // Seed the random number generator with current time
    srand(time(NULL));
    
    // =====================================================================
    // SEQUENTIAL MONTE CARLO SIMULATION
    // Process each sample one at a time
    // =====================================================================
    
    clock_t start_time = clock();
    
    for (long i = 0; i < NUM_SAMPLES; i++) {
        // Generate random point (x, y) in range [-1, 1]
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;  // Random x in [-1, 1]
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;  // Random y in [-1, 1]
        
        // Check if point is inside unit circle: x² + y² ≤ 1
        double distance_squared = x * x + y * y;
        
        if (distance_squared <= 1.0) {
            count_inside++;  // Point is inside circle
        }
    }
    
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    // =====================================================================
    // CALCULATE AND DISPLAY RESULTS
    // =====================================================================
    
    // Estimate π: π ≈ 4 × (points inside circle) / (total points)
    double pi_estimate = 4.0 * count_inside / NUM_SAMPLES;
    double error = fabs(pi_estimate - M_PI);  // M_PI is actual π value
    
    printf("\nResults:\n");
    printf("Estimated π: %.10f\n", pi_estimate);
    printf("Actual π:    %.10f\n", M_PI);
    printf("Error:       %.10f\n", error);
    printf("Time:        %.4f seconds\n", elapsed_time);
    
    return 0;
}
