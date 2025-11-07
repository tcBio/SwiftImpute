#include "kernels/logsumexp.cuh"
#include "kernels/emission.hpp"
#include "kernels/transition.hpp"
#include "kernels/sampling.hpp"
#include "kernels/forward_backward.cuh"
#include "core/types.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace swiftimpute;
using namespace swiftimpute::kernels;

// Test harness
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

std::vector<TestResult> test_results;

void report_test(const std::string& name, bool passed, const std::string& msg = "") {
    test_results.push_back({name, passed, msg});
    if (passed) {
        std::cout << "[PASS] " << name << std::endl;
    } else {
        std::cout << "[FAIL] " << name;
        if (!msg.empty()) {
            std::cout << ": " << msg;
        }
        std::cout << std::endl;
    }
}

// Helper function to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Test logsumexp2 device function
__global__ void test_logsumexp2_kernel(prob_t* results) {
    // Test cases in log space
    results[0] = logsumexp2(logf(0.5f), logf(0.5f));  // Should be log(1.0) = 0.0
    results[1] = logsumexp2(logf(0.3f), logf(0.7f));  // Should be log(1.0) = 0.0
    results[2] = logsumexp2(logf(0.1f), logf(0.9f));  // Should be log(1.0) = 0.0
    results[3] = logsumexp2(-999.0f, logf(1.0f));     // Should be log(1.0) = 0.0
}

void test_logsumexp() {
    try {
        prob_t* d_results;
        CHECK_CUDA(cudaMalloc(&d_results, 4 * sizeof(prob_t)));

        test_logsumexp2_kernel<<<1, 1>>>(d_results);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<prob_t> results(4);
        CHECK_CUDA(cudaMemcpy(results.data(), d_results, 4 * sizeof(prob_t), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < 4; i++) {
            float expected = 0.0f;
            if (std::abs(results[i] - expected) > 0.01f) {
                passed = false;
                break;
            }
        }

        CHECK_CUDA(cudaFree(d_results));
        report_test("LogSumExp operations", passed);
    } catch (const std::exception& e) {
        report_test("LogSumExp operations", false, e.what());
    }
}

// Test EmissionComputer
void test_emission_computer() {
    try {
        const marker_t num_markers = 10;
        const haplotype_t num_haplotypes = 20;
        const uint32_t num_states = 4;
        const uint32_t num_samples = 2;

        // Create synthetic reference panel
        std::vector<allele_t> reference(num_markers * num_haplotypes, 0);
        for (size_t i = 0; i < reference.size(); i++) {
            reference[i] = (i % 2);  // Alternating 0s and 1s
        }

        // Create emission computer
        EmissionComputer computer(num_markers, num_haplotypes, num_states, 0);
        computer.set_reference_panel(reference.data());

        // Create synthetic genotype likelihoods (high confidence hom ref)
        std::vector<GenotypeLikelihoods> genotype_liks(num_samples * num_markers);
        for (auto& gl : genotype_liks) {
            gl.ll_00 = 0.0f;   // log10(1.0) = 0
            gl.ll_01 = -2.0f;  // log10(0.01)
            gl.ll_11 = -2.0f;  // log10(0.01)
        }

        // Create selected states (use first num_states haplotypes)
        std::vector<haplotype_t> selected_states(num_samples * num_markers * num_states * 2);
        for (size_t i = 0; i < selected_states.size(); i++) {
            selected_states[i] = (i % num_states) * 2;
        }

        // Allocate GPU memory
        GenotypeLikelihoods* d_genotype_liks;
        haplotype_t* d_selected_states;
        prob_t* d_emission_probs;

        CHECK_CUDA(cudaMalloc(&d_genotype_liks, genotype_liks.size() * sizeof(GenotypeLikelihoods)));
        CHECK_CUDA(cudaMalloc(&d_selected_states, selected_states.size() * sizeof(haplotype_t)));
        CHECK_CUDA(cudaMalloc(&d_emission_probs, num_samples * num_markers * num_states * sizeof(prob_t)));

        CHECK_CUDA(cudaMemcpy(d_genotype_liks, genotype_liks.data(),
                             genotype_liks.size() * sizeof(GenotypeLikelihoods), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_selected_states, selected_states.data(),
                             selected_states.size() * sizeof(haplotype_t), cudaMemcpyHostToDevice));

        // Compute emissions
        computer.compute(d_genotype_liks, d_selected_states, d_emission_probs, num_samples, true);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy results back
        std::vector<prob_t> emission_probs(num_samples * num_markers * num_states);
        CHECK_CUDA(cudaMemcpy(emission_probs.data(), d_emission_probs,
                             emission_probs.size() * sizeof(prob_t), cudaMemcpyDeviceToHost));

        // Verify emissions are in log space and reasonable
        bool passed = true;
        for (const auto& prob : emission_probs) {
            if (prob > 0.0f || prob < -10.0f) {  // Should be log probs, so negative
                passed = false;
                break;
            }
        }

        CHECK_CUDA(cudaFree(d_genotype_liks));
        CHECK_CUDA(cudaFree(d_selected_states));
        CHECK_CUDA(cudaFree(d_emission_probs));

        report_test("EmissionComputer basic test", passed);
    } catch (const std::exception& e) {
        report_test("EmissionComputer basic test", false, e.what());
    }
}

// Test TransitionComputer
void test_transition_computer() {
    try {
        const marker_t num_markers = 10;
        const uint32_t num_states = 4;

        // Create transition computer
        TransitionComputer computer(num_markers, num_states, 0);

        // Set genetic distances (uniform 0.01 cM between markers)
        std::vector<double> genetic_distances(num_markers, 0.0);
        for (marker_t m = 1; m < num_markers; m++) {
            genetic_distances[m] = m * 0.01;
        }
        computer.set_genetic_map(genetic_distances.data());

        // Set HMM parameters
        kernels::HMMParameters params;
        params.ne = 10000.0;
        params.rho_rate = 1e-8;
        params.theta = 0.001;
        computer.set_parameters(params);

        // Compute transitions
        computer.compute();

        // Get transition matrices
        const prob_t* transitions = computer.get_transition_matrices();

        bool passed = (transitions != nullptr);
        report_test("TransitionComputer compute", passed);

        // Verify memory usage is reasonable
        size_t mem_usage = computer.memory_usage();
        size_t expected_min = (num_markers - 1) * num_states * num_states * sizeof(prob_t);
        passed = (mem_usage >= expected_min);

        report_test("TransitionComputer memory usage", passed);
    } catch (const std::exception& e) {
        report_test("TransitionComputer", false, e.what());
    }
}

// Test HaplotypeSampler
void test_haplotype_sampler() {
    try {
        const marker_t num_markers = 10;
        const haplotype_t num_haplotypes = 20;
        const uint32_t num_states = 4;
        const uint32_t num_samples = 2;

        // Create synthetic reference panel
        std::vector<allele_t> reference(num_markers * num_haplotypes, 0);
        for (size_t i = 0; i < reference.size(); i++) {
            reference[i] = (i % 2);
        }

        // Create haplotype sampler
        HaplotypeSampler sampler(num_markers, num_haplotypes, num_states, 0);
        sampler.set_reference_panel(reference.data());
        sampler.initialize_rng(num_samples, 12345ULL);

        // Create synthetic posteriors (uniform distribution in log space)
        std::vector<prob_t> posteriors(num_samples * num_markers * num_states);
        for (auto& p : posteriors) {
            p = logf(1.0f / num_states);  // Uniform distribution
        }

        // Create selected states
        std::vector<haplotype_t> selected_states(num_samples * num_markers * num_states * 2);
        for (size_t i = 0; i < selected_states.size(); i++) {
            selected_states[i] = (i % num_haplotypes);
        }

        // Allocate GPU memory
        prob_t* d_posteriors;
        haplotype_t* d_selected_states;
        allele_t* d_output_haplotypes;

        CHECK_CUDA(cudaMalloc(&d_posteriors, posteriors.size() * sizeof(prob_t)));
        CHECK_CUDA(cudaMalloc(&d_selected_states, selected_states.size() * sizeof(haplotype_t)));
        CHECK_CUDA(cudaMalloc(&d_output_haplotypes, num_samples * 2 * num_markers * sizeof(allele_t)));

        CHECK_CUDA(cudaMemcpy(d_posteriors, posteriors.data(),
                             posteriors.size() * sizeof(prob_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_selected_states, selected_states.data(),
                             selected_states.size() * sizeof(haplotype_t), cudaMemcpyHostToDevice));

        // Sample haplotypes (deterministic mode for testing)
        sampler.sample(d_posteriors, d_selected_states, d_output_haplotypes, num_samples, true);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy results back
        std::vector<allele_t> output_haplotypes(num_samples * 2 * num_markers);
        CHECK_CUDA(cudaMemcpy(output_haplotypes.data(), d_output_haplotypes,
                             output_haplotypes.size() * sizeof(allele_t), cudaMemcpyDeviceToHost));

        // Verify output is valid (0 or 1)
        bool passed = true;
        for (const auto& allele : output_haplotypes) {
            if (allele != 0 && allele != 1) {
                passed = false;
                break;
            }
        }

        CHECK_CUDA(cudaFree(d_posteriors));
        CHECK_CUDA(cudaFree(d_selected_states));
        CHECK_CUDA(cudaFree(d_output_haplotypes));

        report_test("HaplotypeSampler basic test", passed);
    } catch (const std::exception& e) {
        report_test("HaplotypeSampler basic test", false, e.what());
    }
}

// Test forward-backward integration
void test_forward_backward_integration() {
    try {
        const uint32_t num_samples = 1;
        const marker_t num_markers = 5;
        const uint32_t num_states = 2;
        const uint32_t checkpoint_interval = 2;

        // Create synthetic emission probabilities (uniform in log space)
        std::vector<prob_t> emissions(num_samples * num_markers * num_states);
        for (auto& e : emissions) {
            e = logf(1.0f / num_states);
        }

        // Create uniform transition matrix
        std::vector<prob_t> transitions(num_states * num_states);
        for (auto& t : transitions) {
            t = logf(1.0f / num_states);
        }

        // Dummy selected states (not used in this simplified test)
        std::vector<marker_t> selected_states(num_markers * num_states, 0);

        // Allocate GPU memory
        prob_t* d_emissions;
        prob_t* d_transitions;
        marker_t* d_selected_states;
        prob_t* d_forward_checkpoints;
        prob_t* d_scaling_factors;
        prob_t* d_posteriors;

        uint32_t num_checkpoints = (num_markers + checkpoint_interval - 1) / checkpoint_interval;

        CHECK_CUDA(cudaMalloc(&d_emissions, emissions.size() * sizeof(prob_t)));
        CHECK_CUDA(cudaMalloc(&d_transitions, transitions.size() * sizeof(prob_t)));
        CHECK_CUDA(cudaMalloc(&d_selected_states, selected_states.size() * sizeof(marker_t)));
        CHECK_CUDA(cudaMalloc(&d_forward_checkpoints, num_samples * num_checkpoints * num_states * sizeof(prob_t)));
        CHECK_CUDA(cudaMalloc(&d_scaling_factors, num_samples * num_markers * sizeof(prob_t)));
        CHECK_CUDA(cudaMalloc(&d_posteriors, num_samples * num_markers * num_states * sizeof(prob_t)));

        CHECK_CUDA(cudaMemcpy(d_emissions, emissions.data(), emissions.size() * sizeof(prob_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_transitions, transitions.data(), transitions.size() * sizeof(prob_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_selected_states, selected_states.data(), selected_states.size() * sizeof(marker_t), cudaMemcpyHostToDevice));

        // Run forward pass
        launch_forward_pass(d_emissions, d_transitions, d_selected_states,
                          num_samples, num_markers, num_states, checkpoint_interval,
                          d_forward_checkpoints, d_scaling_factors, 0);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Run backward pass
        launch_backward_pass(d_emissions, d_transitions, d_selected_states,
                           d_forward_checkpoints, d_scaling_factors,
                           num_samples, num_markers, num_states, checkpoint_interval,
                           d_posteriors, 0);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy posteriors back
        std::vector<prob_t> posteriors(num_samples * num_markers * num_states);
        CHECK_CUDA(cudaMemcpy(posteriors.data(), d_posteriors,
                             posteriors.size() * sizeof(prob_t), cudaMemcpyDeviceToHost));

        // Verify posteriors are normalized (sum to 1 in probability space)
        bool passed = true;
        for (uint32_t s = 0; s < num_samples; s++) {
            for (marker_t m = 0; m < num_markers; m++) {
                prob_t sum = 0.0f;
                for (uint32_t st = 0; st < num_states; st++) {
                    size_t idx = (s * num_markers + m) * num_states + st;
                    sum += expf(posteriors[idx]);
                }
                if (std::abs(sum - 1.0f) > 0.01f) {
                    passed = false;
                    break;
                }
            }
        }

        CHECK_CUDA(cudaFree(d_emissions));
        CHECK_CUDA(cudaFree(d_transitions));
        CHECK_CUDA(cudaFree(d_selected_states));
        CHECK_CUDA(cudaFree(d_forward_checkpoints));
        CHECK_CUDA(cudaFree(d_scaling_factors));
        CHECK_CUDA(cudaFree(d_posteriors));

        report_test("Forward-Backward integration", passed);
    } catch (const std::exception& e) {
        report_test("Forward-Backward integration", false, e.what());
    }
}

int main() {
    std::cout << "SwiftImpute GPU Kernel Tests" << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << std::endl;

    // Run all tests
    test_logsumexp();
    test_emission_computer();
    test_transition_computer();
    test_haplotype_sampler();
    test_forward_backward_integration();

    // Summary
    std::cout << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "============" << std::endl;

    int passed = 0;
    int failed = 0;
    for (const auto& result : test_results) {
        if (result.passed) {
            passed++;
        } else {
            failed++;
        }
    }

    std::cout << "Passed: " << passed << "/" << test_results.size() << std::endl;
    std::cout << "Failed: " << failed << "/" << test_results.size() << std::endl;

    return (failed == 0) ? 0 : 1;
}
