// openmp_data_pipeline.c
// This demonstrates: SHARED MEMORY + DATA PARALLELISM
// Multiple threads process different numbers through identical pipeline

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    const int ARRAY_SIZE = 10000000;        // 10 million numbers (same as sequential)
    const int THRESHOLD = 50;                // Count numbers above this value
    
    printf("=== OPENMP PARALLEL PROCESSING ===\n");
    printf("Processing %d numbers (%.1f million)\n", ARRAY_SIZE, ARRAY_SIZE/1000000.0);
    
    // STEP 1: Create and initialize input array (shared among all threads)
    int *numbers = malloc(ARRAY_SIZE * sizeof(int));  // Use int for input numbers
    if (!numbers) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    // Fill array with consecutive numbers: 1, 2, 3, 4, ...
    for (int i = 0; i < ARRAY_SIZE; i++) {
        numbers[i] = i + 1;
    }
    
    // Start timing the computational pipeline
    double start_time = omp_get_wtime();     // Use OpenMP timing function
    
    // STEP 2: Process each number through the complete pipeline
    double total_sum = 0.0;               // Use double to avoid overflow
    int count_above_threshold = 0;        // Count values above threshold
    
    // OpenMP parallel processing: each thread handles different iterations
    #pragma omp parallel for reduction(+:total_sum,count_above_threshold)
    for (int i = 0; i < ARRAY_SIZE; i++) {
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
    
    // End timing
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    
    // STEP 3: Display results
    printf("\n=== RESULTS ===\n");
    printf("Total sum: %.0f\n", total_sum);  // Use %.0f for double
    printf("Count > %d: %d\n", THRESHOLD, count_above_threshold);
    printf("Processing time: %.4f seconds\n", elapsed_time);
    printf("Numbers processed: %d (%.1f million)\n", ARRAY_SIZE, ARRAY_SIZE/1000000.0);
    
    free(numbers);
    return 0;
}
