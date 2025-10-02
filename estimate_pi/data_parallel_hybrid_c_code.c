// Monte Carlo π estimation using hybrid MPI + OpenMP parallelism
// Multiple processes (MPI), each using multiple threads (OpenMP)

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;
    const long NUM_SAMPLES = 500000000;  // 500 million total samples
    
    // =====================================================================
    // MPI INITIALIZATION WITH THREAD SUPPORT
    // =====================================================================
    
    int thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("=== HYBRID MPI+OPENMP MONTE CARLO PI ESTIMATION ===\n");
        printf("Number of samples: %ld (%.1f million)\n", 
               NUM_SAMPLES, NUM_SAMPLES/1000000.0);
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("Total cores: %d\n", size * omp_get_max_threads());
    }
    
    // =====================================================================
    // WORK DISTRIBUTION (MPI level)
    // Divide samples among MPI processes
    // =====================================================================
    
    long samples_per_process = NUM_SAMPLES / size;
    long my_samples = samples_per_process;
    
    if (rank == size - 1) {
        my_samples += NUM_SAMPLES % size;
    }
    
    // =====================================================================
    // HYBRID PARALLEL MONTE CARLO SIMULATION
    // MPI level: Different processes work on different sample ranges
    // OpenMP level: Threads within each process divide that process's work
    // =====================================================================
    
    double start_time = MPI_Wtime();
    
    long local_count = 0;  // Count for this process
    
    // OPENMP PARALLEL REGION within each MPI process
    #pragma omp parallel reduction(+:local_count)
    {
        // Each thread gets unique random seed
        unsigned int seed = rank * 1000 + omp_get_thread_num() + time(NULL);
        
        // Each thread processes a portion of this process's samples
        #pragma omp for
        for (long i = 0; i < my_samples; i++) {
            // Generate random point
            double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            
            // Check if inside circle
            double distance_squared = x * x + y * y;
            
            if (distance_squared <= 1.0) {
                local_count++;  // Thread-safe via reduction
            }
        }
    }
    
    // =====================================================================
    // MPI COMMUNICATION: COMBINE RESULTS FROM ALL PROCESSES
    // =====================================================================
    
    long global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
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
