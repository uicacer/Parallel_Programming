// Monte Carlo π estimation using MPI distributed memory parallelism
// Multiple processes on different computers work independently

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;
    const long NUM_SAMPLES = 500000000;  // 500 million total samples
    
    // =====================================================================
    // MPI INITIALIZATION
    // =====================================================================
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get my process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes
    
    if (rank == 0) {
        printf("=== MPI MONTE CARLO PI ESTIMATION ===\n");
        printf("Number of samples: %ld (%.1f million)\n", 
               NUM_SAMPLES, NUM_SAMPLES/1000000.0);
        printf("Number of processes: %d\n", size);
    }
    
    // =====================================================================
    // WORK DISTRIBUTION
    // Divide samples among MPI processes
    // =====================================================================
    
    long samples_per_process = NUM_SAMPLES / size;
    long my_samples = samples_per_process;
    
    // Last process handles remainder
    if (rank == size - 1) {
        my_samples += NUM_SAMPLES % size;
    }
    
    // =====================================================================
    // MPI PARALLEL MONTE CARLO SIMULATION
    // Each process works independently on its portion
    // =====================================================================
    
    // Each process needs unique random seed
    srand(rank + time(NULL));
    
    double start_time = MPI_Wtime();
    
    long local_count = 0;  // Count for this process
    
    for (long i = 0; i < my_samples; i++) {
        // Generate random point
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        
        // Check if inside circle
        double distance_squared = x * x + y * y;
        
        if (distance_squared <= 1.0) {
            local_count++;
        }
    }
    
    // =====================================================================
    // MPI COMMUNICATION: COMBINE RESULTS
    // All processes send their counts to master process (rank 0)
    // =====================================================================
    
    long global_count = 0;
    MPI_Reduce(&local_count,      // What I'm sending
               &global_count,     // Where combined result goes
               1,                 // Number of elements
               MPI_LONG,          // Data type
               MPI_SUM,           // Operation (sum all counts)
               0,                 // Destination (rank 0)
               MPI_COMM_WORLD);   // All processes participate
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // Get maximum time across all processes
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
