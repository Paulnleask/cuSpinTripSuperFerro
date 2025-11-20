// =========================
// main_modular.cpp
// =========================
#include "simulation.hpp"
#include "renderer_gl.hpp"
#include <cstdio>
#include <chrono>

// ---------------- Optimization main function ----------------
int main(int argc, char** argv) {
    
    // Initialise the optimizer
    Simulation optimizer{};
    optimizer.init();

    // Number of fields consistency check
    if(optimizer.p.number_total_fields != no_fields){
        std::printf("Error: simulation cannot run\n");
        std::printf("Reason: no_fields in kernels.hpp must match number_total_fields in params.hpp\n");
        exit(1);
    }

    // Output initial data
    optimizer.output_data();

    // Print simulation details
    optimizer.details();

    // Begin timer
    auto sim_start = std::chrono::high_resolution_clock::now();

    // Minimize cost (energy)
    if(optimizer.p.real_time_visualization){
        run_gl_simulation(optimizer, &argc, argv);
    }
    else{
        while (optimizer.p.error > optimizer.p.tolerance) {
            optimizer.p.error = optimizer.step();
            optimizer.computeObservables();
            if ((optimizer.p.epochs % optimizer.p.loop_display) == 0) {
                std::printf("t=%.1f (%d epochs): error=%.6f, energy=%.6f, skyrmion=%.6f, vortex=%.6f\n", optimizer.p.epochs * optimizer.p.time_step, optimizer.p.epochs, optimizer.p.error, optimizer.p.energy, optimizer.p.skyrmion, optimizer.p.vortex);
            }
            if (optimizer.p.epochs >= optimizer.p.loops_max) break;
            ++optimizer.p.epochs;
            if ((optimizer.p.epochs % optimizer.p.loop_output) == 0) {
                optimizer.output_data();
            }
        }
    }

    // Runtime
	auto sim_end = std::chrono::high_resolution_clock::now();
	auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(sim_end - sim_start).count();

    // Output final data
    optimizer.output_data();

    std::printf("\nSimulation converged within tolerance");
    std::printf("\nProperties of converged solution: energy=%.6f, skyrmion=%.6f, vortex=%.6f", optimizer.p.energy, optimizer.p.skyrmion, optimizer.p.vortex);
    std::printf("\nRun time: t=%.1f seconds (%d epochs)\n", runtime / 1000000.0, optimizer.p.epochs);

    // Free up memory
    optimizer.uninit();

    return 0;
}