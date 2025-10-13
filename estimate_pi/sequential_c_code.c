// Monte Carlo π estimation using sequential processing
// Throws random darts and counts how many fall inside a unit circle

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*
 * MONTE CARLO METHOD FOR ESTIMATING π:
 * 
 * The mathematical principle:
 * - Imagine a circle of radius 1 inscribed in a square of side 2
 * - Area of circle = π × r² = π × 1² = π
 * - Area of square = 2 × 2 = 4
 * - Ratio = (Area of circle) / (Area of square) = π/4
 * 
 * The simulation:
 * - Generate random points uniformly distributed in the square [-1,1] × [-1,1]
 * - Count how many points fall inside the circle (distance from origin ≤ 1)
 * - Ratio of points inside / total points ≈ π/4
 * - Therefore: π ≈ 4 × (points inside circle) / (total points)
 * 
 * As the number of random samples increases, the estimate converges to π
 */

int main() {
    // =====================================================================
    // CONFIGURATION
    // =====================================================================
    
    const long NUM_SAMPLES = 500000000;  // 500 million samples for accuracy
                                         // More samples = better approximation
                                         // but takes longer to compute
    
    long count_inside = 0;               // Counter for points that fall inside
                                         // the unit circle (x² + y² ≤ 1)

    // =====================================================================
    // DISPLAY SIMULATION INFO
    // =====================================================================
    
    printf("=== SEQUENTIAL MONTE CARLO PI ESTIMATION ===\n");
    printf("Number of samples: %ld (%.1f million)\n",
           NUM_SAMPLES, NUM_SAMPLES/1000000.0);

    // =====================================================================
    // RANDOM NUMBER GENERATOR SETUP
    // =====================================================================

    // Seed the random number generator with current time
    // This ensures different random sequences each time the program runs
    // Without seeding, rand() would produce the same sequence every time
    srand(time(NULL));

    // =====================================================================
    // SEQUENTIAL MONTE CARLO SIMULATION
    // Process each sample one at a time (no parallelization)
    // =====================================================================

    // Start timing the computation
    clock_t start_time = clock();

    // Main simulation loop: generate and test NUM_SAMPLES random points
    for (long i = 0; i < NUM_SAMPLES; i++) {
        // Generate random point (x, y) in the range [-1, 1] × [-1, 1]
        // This covers the square that contains our unit circle
        
        // Step 1: rand() returns integer in [0, RAND_MAX]
        // Step 2: Divide by RAND_MAX to get value in [0, 1]
        // Step 3: Multiply by 2.0 to get [0, 2]
        // Step 4: Subtract 1.0 to shift to [-1, 1]
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;  // Random x in [-1, 1]
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;  // Random y in [-1, 1]

        // Calculate the squared distance from the origin (0, 0)
        // We use x² + y² instead of sqrt(x² + y²) for efficiency
        // A point is inside the unit circle if x² + y² ≤ 1
        double distance_squared = x * x + y * y;

        // Check if the point falls inside the unit circle
        if (distance_squared <= 1.0) {
            count_inside++;  // Increment our counter for points inside the circle
        }
         
    }

    // Stop timing the computation
    clock_t end_time = clock();

    // =====================================================================
    // CALCULATE AND DISPLAY RESULTS
    // =====================================================================

    // Calculate the elapsed time in seconds
    // CLOCKS_PER_SEC converts clock ticks to seconds
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Estimate π using the Monte Carlo formula:
    // π ≈ 4 × (points inside circle) / (total points)
    // 
    // Why multiply by 4?
    // - Ratio of points inside/total ≈ (area of circle)/(area of square)
    // - This ratio = π/4, so we multiply by 4 to get π
    double pi_estimate = 4.0 * count_inside / NUM_SAMPLES;

    // Calculate the error compared to the mathematical constant M_PI
    // M_PI is defined in math.h as a high-precision value of π
    double error = fabs(pi_estimate - M_PI);
    
    // Calculate percent error for easier interpretation
    double percent_error = (error / M_PI) * 100.0;

    // =====================================================================
    // OUTPUT RESULTS
    // =====================================================================
    
    printf("\n=== RESULTS ===\n");
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

    // =====================================================================
    // EXPLANATION OF ACCURACY
    // =====================================================================
    
    printf("\n=== NOTES ===\n");
    printf("Monte Carlo method accuracy improves with √N\n");
    printf("where N is the number of samples.\n");
    printf("To cut error in half, you need 4× more samples.\n");

    return 0;
}

/*
 * SEQUENTIAL vs PARALLEL:
 * 
 * This is a SEQUENTIAL implementation - it processes one sample at a time.
 * Characteristics:
 * - Simple and straightforward
 * - Uses only one CPU core
 * - Good for learning/understanding the algorithm
 * - Slower for large numbers of samples
 * 
 * For better performance, this algorithm is "embarrassingly parallel":
 * - Each random point can be tested independently
 * - No data dependencies between iterations
 * - Ideal candidate for parallelization (OpenMP, MPI, CUDA, etc.)
 * - Parallel versions can achieve near-linear speedup with multiple cores/GPUs
 * 
 */
