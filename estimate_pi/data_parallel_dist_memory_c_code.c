// Monte Carlo π estimation using MPI distributed memory parallelism
// Multiple processes on different computers work independently

// =====================================================================
// HEADER FILES (Libraries we need)
// =====================================================================
#include <stdio.h>   // Standard Input/Output: printf, etc.
#include <stdlib.h>  // Standard Library: rand(), srand(), RAND_MAX
#include <mpi.h>     // MPI: Distributed memory parallel programming
#include <math.h>    // Math functions: fabs() for absolute value
#include <time.h>    // Time functions: time() for random seed


// int argc = Argument Count (number of command-line arguments)
// char** argv = Argument Vector (array of strings containing the arguments)
// MPI requires these to read configuration details when the program starts
int main(int argc, char** argv) {
    // =====================================================================
    // MPI VARIABLE DECLARATIONS
    // =====================================================================
    
    // 'rank' = This process's unique ID number (like an employee ID)
    // 'size' = Total number of processes running (like total employees)
    int rank, size;
    
    const long NUM_SAMPLES = 500000000;  // Total dart throws (500 million)
    
    // =====================================================================
    // MPI INITIALIZATION - REQUIRED FOR ALL MPI PROGRAMS
    // =====================================================================
    
    // MPI_Init() MUST be the first MPI function called
    // It starts up the MPI system and prepares processes for communication
    // 
    // &argc and &argv are passed so MPI can read command-line arguments
    // (MPI uses these to configure how processes communicate)
    MPI_Init(&argc, &argv);
    
    // MPI_Comm_rank() - Gets this process's unique ID number
    // 
    // WHAT IS A RANK?
    // - Like an employee ID number: unique identifier for each process
    // - Process 0 (rank 0) is typically the "master" or "coordinator"
    // - Other processes (rank 1, 2, 3...) are typically "workers"
    // - Each process gets a different rank, so they know who they are
    //
    // MPI_COMM_WORLD = The "communicator" containing ALL processes
    // Think of it as "everyone in this parallel job"
    //
    // &rank = Address where MPI will store my rank number
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // MPI_Comm_size() - Gets total number of processes
    //
    // If you launched with: mpirun -np 4 mpi_pi
    // Then size = 4 (four processes running)
    //
    // &size = Address where MPI will store the total count
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // =====================================================================
    // CONDITIONAL OUTPUT - ONLY MASTER PROCESS PRINTS
    // =====================================================================
    
    // KEY CONCEPT: All processes run the SAME code
    // But we only want ONE process to print the header (avoid duplicate output)
    // So we use: if (rank == 0) to make only the master print
    
    if (rank == 0) {
        printf("=== MPI MONTE CARLO PI ESTIMATION ===\n");
        printf("Number of samples: %ld (%.1f million)\n", 
               NUM_SAMPLES, NUM_SAMPLES/1000000.0);
        printf("Number of processes: %d\n", size);
    }
    
    // =====================================================================
    // WORK DISTRIBUTION - DIVIDE SAMPLES AMONG PROCESSES
    // =====================================================================
    
    // DISTRIBUTED MEMORY CONCEPT:
    // Unlike OpenMP where threads share memory, MPI processes have
    // COMPLETELY SEPARATE memory spaces (like different computers)
    // Each process can only access its own memory
    // So we explicitly divide the work among processes
    
    // Calculate base amount each process handles
    long samples_per_process = NUM_SAMPLES / size;
    
    // Start with base amount for this process
    long my_samples = samples_per_process;
    
    // Handle remainder: if NUM_SAMPLES doesn't divide evenly
    // Last process takes any extra samples
    // 
    // EXAMPLE with 500,000,000 samples and 4 processes:
    // samples_per_process = 500,000,000 / 4 = 125,000,000
    // NUM_SAMPLES % size = 500,000,000 % 4 = 0 (no remainder)
    //
    // Process 0: my_samples = 125,000,000
    // Process 1: my_samples = 125,000,000
    // Process 2: my_samples = 125,000,000
    // Process 3: my_samples = 125,000,000
    //
    // EXAMPLE with 500,000,003 samples and 4 processes:
    // samples_per_process = 500,000,003 / 4 = 125,000,000
    // NUM_SAMPLES % size = 500,000,003 % 4 = 3 (remainder of 3)
    //
    // Process 0: my_samples = 125,000,000
    // Process 1: my_samples = 125,000,000
    // Process 2: my_samples = 125,000,000
    // Process 3: my_samples = 125,000,000 + 3 = 125,000,003
    
    if (rank == size - 1) {
        my_samples += NUM_SAMPLES % size;
    }
    
    // =====================================================================
    // RANDOM NUMBER GENERATOR SETUP - EACH PROCESS NEEDS UNIQUE SEED
    // =====================================================================
    
    // CRITICAL: Each process must generate different random numbers!
    // If all processes use the same seed, they generate IDENTICAL sequences
    // This would mean we're not really sampling 500 million points,
    // but the same 125 million points repeated 4 times!
    //
    // SOLUTION: Each process gets unique seed
    // rank = process ID (0, 1, 2, 3...)
    // time(NULL) = current time in seconds (a large number)
    //
    // Process 0: seed = 0 + time(NULL) = 1234567890
    // Process 1: seed = 1 + time(NULL) = 1234567891
    // Process 2: seed = 2 + time(NULL) = 1234567892
    // Process 3: seed = 3 + time(NULL) = 1234567893
    //
    // srand() initializes the random number generator with this seed
    srand(rank + time(NULL));
    
    // =====================================================================
    // TIMING START
    // =====================================================================
    
    // MPI_Wtime() returns wall-clock time in seconds
    // (Similar to omp_get_wtime() but for MPI)
    double start_time = MPI_Wtime();
    
    // =====================================================================
    // MONTE CARLO COMPUTATION - EACH PROCESS WORKS INDEPENDENTLY
    // =====================================================================
    
    // local_count = number of points THIS process finds inside circle
    // Each process has its own separate local_count in its own memory
    // (Unlike OpenMP where we needed reduction for shared memory)
    long local_count = 0;
    
    // INDEPENDENT COMPUTATION:
    // Each process runs this loop for its assigned number of samples
    // Process 0: iterates 125 million times
    // Process 1: iterates 125 million times
    // Process 2: iterates 125 million times
    // Process 3: iterates 125 million times
    // All happening SIMULTANEOUSLY on different computers
    
    for (long i = 0; i < my_samples; i++) {
        
        // ----------------------------------------------------------------
        // Generate random point (x, y) in range [-1, 1]
        // ----------------------------------------------------------------
        
        // rand() is okay here (unlike OpenMP) because:
        // - Each process has its OWN copy of rand()'s internal state
        // - Processes have separate memory spaces
        // - No process can interfere with another process's rand()
        //
        // TRANSFORMATION: rand() → [0, RAND_MAX] → [0, 1] → [0, 2] → [-1, 1]
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        
        // ----------------------------------------------------------------
        // Check if point is inside unit circle
        // ----------------------------------------------------------------
        
        // Circle equation: x² + y² ≤ 1
        double distance_squared = x * x + y * y;
        
        if (distance_squared <= 1.0) {
            local_count++;  // Increment THIS process's count
        }
    }
    // End of computation loop
    
    // At this point, each process has its own local_count:
    // Process 0: local_count = (some number of hits)
    // Process 1: local_count = (some number of hits)
    // Process 2: local_count = (some number of hits)
    // Process 3: local_count = (some number of hits)
    //
    // But these are all in SEPARATE memory on SEPARATE computers!
    // We need to COMMUNICATE to combine them.
    
    // =====================================================================
    // MPI COMMUNICATION - COMBINE RESULTS FROM ALL PROCESSES
    // =====================================================================
    
    // This is where MPI differs fundamentally from OpenMP:
    // We must EXPLICITLY send messages between processes to share data
    
    long global_count = 0;  // Variable to hold combined result
    
    // MPI_Reduce() - THE KEY MPI COMMUNICATION FUNCTION
    //
    // PURPOSE: Collect data from all processes and combine it
    //
    // ANALOGY: 
    // Imagine 4 workers (processes) counting inventory in different warehouses
    // Each worker counts items in their warehouse (local_count)
    // Then all workers report their counts to the manager (rank 0)
    // Manager adds up all counts to get total (global_count)
    //
    // WHAT HAPPENS:
    // 1. Each process sends its local_count to process 0
    // 2. Process 0 receives all local_counts and adds them together
    // 3. Result is stored in global_count on process 0
    // 4. Other processes: global_count remains 0 (they don't receive result)
    
    MPI_Reduce(
        &local_count,      // SEND: Address of what I'm sending
                          //       Each process sends its own local_count
                          
        &global_count,     // RECEIVE: Address where result goes
                          //           Only meaningful on rank 0 (destination)
                          //           Other processes: this stays 0
                          
        1,                 // COUNT: Number of elements to send
                          //        We're sending 1 number (the count)
                          
        MPI_LONG,          // DATATYPE: What type of data
                          //           MPI_LONG for 'long' integers
                          //           Other types: MPI_INT, MPI_DOUBLE, etc.
                          
        MPI_SUM,           // OPERATION: How to combine data
                          //            MPI_SUM = add all values together
                          //            Other ops: MPI_MAX, MPI_MIN, MPI_PROD
                          
        0,                 // DESTINATION: Which process receives result
                          //              0 = master process (rank 0)
                          
        MPI_COMM_WORLD     // COMMUNICATOR: Who participates
                          //               MPI_COMM_WORLD = all processes
    );
    
    // AFTER MPI_Reduce:
    // Process 0: global_count = sum of all local_counts ✓
    // Process 1: global_count = 0 (didn't receive result)
    // Process 2: global_count = 0 (didn't receive result)
    // Process 3: global_count = 0 (didn't receive result)
    
    // =====================================================================
    // TIMING END - COLLECT MAXIMUM TIME
    // =====================================================================
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // Different processes might finish at slightly different times
    // We want the SLOWEST process's time (the bottleneck)
    // So we use another MPI_Reduce with MPI_MAX operation
    
    double max_time;
    MPI_Reduce(&elapsed_time,  // Each process sends its time
               &max_time,       // Master receives maximum
               1,                // 1 number
               MPI_DOUBLE,       // Double precision
               MPI_MAX,          // Find MAXIMUM (not sum)
               0,                // Send to rank 0
               MPI_COMM_WORLD);  // All participate
    
    // max_time on process 0 = slowest process's time
    
    // =====================================================================
    // DISPLAY RESULTS - ONLY MASTER PROCESS
    // =====================================================================
    
    // Only rank 0 has the correct global_count and max_time
    // So only rank 0 should print results
    
    if (rank == 0) {
        // Calculate π estimate using combined data from all processes
        double pi_estimate = 4.0 * global_count / NUM_SAMPLES;
        
        // Calculate error
        // M_PI = actual π value (3.14159265...)
        // fabs() = absolute value (always positive)
        double error = fabs(pi_estimate - M_PI);
        
        printf("\nResults:\n");
        printf("Estimated π: %.10f\n", pi_estimate);
        printf("Actual π:    %.10f\n", M_PI);
        printf("Error:       %.10f\n", error);
        printf("Time:        %.4f seconds\n", max_time);
    }
    
    // =====================================================================
    // MPI FINALIZATION - REQUIRED CLEANUP
    // =====================================================================
    
    // MPI_Finalize() MUST be called before program ends
    // It shuts down the MPI system and cleans up communication resources
    // After this, no more MPI functions can be called
    MPI_Finalize();
    
    return 0;  // Program finished successfully
}

// =====================================================================
// KEY MPI CONCEPTS SUMMARY FOR STUDENTS:
// =====================================================================
//
// 1. DISTRIBUTED MEMORY MODEL
//    - Each process has completely separate memory space
//    - Processes cannot directly access each other's variables
//    - Like having multiple separate computers
//    - Must explicitly send messages to share data
//
// 2. PROCESS IDENTIFICATION
//    - rank = unique ID for each process (0, 1, 2, ...)
//    - size = total number of processes
//    - Rank 0 typically designated as "master" or coordinator
//
// 3. SPMD (Single Program, Multiple Data)
//    - All processes run the SAME program
//    - But they process DIFFERENT data
//    - Use rank to determine what each process does
//    - Example: if (rank == 0) { master tasks } else { worker tasks }
//
// 4. MPI COMMUNICATION FUNCTIONS
//    - MPI_Reduce() = Collect data from all processes and combine
//    - Data flows: all processes → one process
//    - Must specify: what to send, how to combine, who receives
//
// 5. SYNCHRONIZATION
//    - MPI_Reduce() is a COLLECTIVE operation
//    - All processes must call it (synchronization point)
//    - Like a meeting where everyone must show up
//    - Program waits until all processes reach this point
//
// 6. UNIQUE SEEDS CRITICAL
//    - Each process must use different random seed
//    - Use: srand(rank + time(NULL))
//    - Ensures different random sequences per process
//
// 7. REQUIRED MPI FUNCTIONS
//    - MPI_Init() = Must be first MPI call (starts system)
//    - MPI_Finalize() = Must be last MPI call (cleanup)
//    - Program will fail if these are missing or out of order
//
// 8. COMPARISON WITH OPENMP
//    - OpenMP: Shared memory, automatic data sharing, simpler
//    - MPI: Distributed memory, explicit messaging, more scalable
//    - OpenMP: Limited to one computer
//    - MPI: Can use thousands of computers
//
// =====================================================================
