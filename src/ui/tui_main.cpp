/**
 * SwiftImpute Terminal User Interface
 *
 * A modern TUI for GPU-accelerated genotype imputation.
 * Works over SSH and doesn't require X11/display.
 */

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/table.hpp>
#include <ftxui/component/event.hpp>

#include "api/imputer.hpp"
#include "api/checkpoint.hpp"
#include "core/types.hpp"
#include "io/binary_format.hpp"

#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <deque>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;
using namespace ftxui;
using namespace swiftimpute;

// ============================================================================
// Application State
// ============================================================================

struct AppState {
    // File paths
    std::string reference_path;
    std::string target_path;
    std::string output_path;
    std::string checkpoint_path;
    std::string genetic_map_path;

    // Configuration
    int preset_index = 0;
    int gpu_device = 0;
    int num_states = 8;
    int batch_size = 100;
    int ne = 10000;
    bool deterministic = false;
    bool batch_mode = true;
    bool rebuild_pbwt = false;

    // Progress tracking
    std::atomic<bool> running{false};
    std::atomic<bool> completed{false};
    std::atomic<bool> failed{false};
    std::atomic<double> overall_progress{0.0};
    std::atomic<double> chromosome_progress{0.0};
    std::string current_chromosome;
    int current_chrom_index = 0;
    int total_chromosomes = 0;
    std::chrono::steady_clock::time_point start_time;

    // GPU info
    std::string gpu_name;
    size_t gpu_memory_total = 0;
    size_t gpu_memory_used = 0;

    // Log messages
    std::deque<std::string> log_messages;
    std::mutex log_mutex;

    // Statistics
    uint32_t samples_processed = 0;
    uint32_t markers_processed = 0;
    std::string error_message;
};

// Global state
AppState g_state;

// ============================================================================
// Utility Functions
// ============================================================================

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && unit_idx < 4) {
        size /= 1024;
        unit_idx++;
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << size << " " << units[unit_idx];
    return ss.str();
}

std::string format_duration(std::chrono::seconds duration) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = duration;

    std::ostringstream ss;
    if (hours.count() > 0) {
        ss << hours.count() << "h ";
    }
    ss << minutes.count() << "m " << seconds.count() << "s";
    return ss.str();
}

void add_log(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_state.log_mutex);
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&time), "[%H:%M:%S] ") << message;
    g_state.log_messages.push_back(ss.str());

    // Keep last 100 messages
    while (g_state.log_messages.size() > 100) {
        g_state.log_messages.pop_front();
    }
}

std::vector<std::string> list_directory(const std::string& path, bool dirs_only = false) {
    std::vector<std::string> entries;

    try {
        if (!fs::exists(path)) {
            return entries;
        }

        // Add parent directory entry
        entries.push_back("..");

        for (const auto& entry : fs::directory_iterator(path)) {
            if (dirs_only && !entry.is_directory()) {
                continue;
            }

            std::string name = entry.path().filename().string();

            // Filter for VCF files if not dirs_only
            if (!dirs_only && entry.is_regular_file()) {
                if (name.find(".vcf") == std::string::npos &&
                    name.find(".bcf") == std::string::npos) {
                    continue;
                }
            }

            if (entry.is_directory()) {
                name += "/";
            }

            entries.push_back(name);
        }

        std::sort(entries.begin() + 1, entries.end());
    } catch (...) {
        // Ignore errors
    }

    return entries;
}

void detect_gpus() {
    try {
        int device_count = get_device_count();
        if (device_count > 0) {
            auto info = get_device_info(0);
            g_state.gpu_name = info.name;
            g_state.gpu_memory_total = info.total_memory;
            g_state.gpu_memory_used = 0;
        }
    } catch (...) {
        g_state.gpu_name = "No GPU detected";
    }
}

// ============================================================================
// Imputation Worker Thread
// ============================================================================

void run_imputation() {
    g_state.running = true;
    g_state.completed = false;
    g_state.failed = false;
    g_state.overall_progress = 0.0;
    g_state.error_message.clear();
    g_state.start_time = std::chrono::steady_clock::now();

    add_log("Starting imputation...");

    try {
        // Build configuration
        ImputationConfig config;

        // Apply preset
        const char* presets[] = {"default", "radseq", "small-ref", "biobank",
                                 "low-memory", "high-accuracy", "large-panel"};
        std::string preset = presets[g_state.preset_index];

        if (preset == "radseq") {
            config = ImputationConfig::radseq_preset();
        } else if (preset == "small-ref") {
            config = ImputationConfig::small_reference_preset();
        } else if (preset == "biobank") {
            config = ImputationConfig::biobank_preset();
        } else if (preset == "low-memory") {
            config = ImputationConfig::low_memory_preset();
        } else if (preset == "high-accuracy") {
            config = ImputationConfig::high_accuracy_preset();
        } else if (preset == "large-panel") {
            config = ImputationConfig::large_panel_preset();
        }

        // Apply user overrides
        config.device_id = g_state.gpu_device;
        config.hmm_params.num_states = g_state.num_states;
        config.batch_size = g_state.batch_size;
        config.hmm_params.ne = g_state.ne;
        config.deterministic = g_state.deterministic;
        config.rebuild_pbwt_per_window = g_state.rebuild_pbwt;

        add_log("Configuration: " + preset + " preset");
        add_log("  States: " + std::to_string(config.hmm_params.num_states));
        add_log("  Batch size: " + std::to_string(config.batch_size));
        add_log("  Ne: " + std::to_string(static_cast<int>(config.hmm_params.ne)));

        if (g_state.batch_mode) {
            // Use batch imputer with checkpointing
            add_log("Running in batch mode with checkpointing");

            BatchImputer imputer(
                g_state.reference_path,
                g_state.target_path,
                g_state.output_path,
                config
            );

            if (!g_state.checkpoint_path.empty()) {
                imputer.set_checkpoint_path(g_state.checkpoint_path);
            }

            // Set progress callback
            imputer.set_progress_callback(
                [](const std::string& chrom, uint32_t chrom_idx, uint32_t total,
                   double chrom_progress, double overall_progress) {
                    g_state.current_chromosome = chrom;
                    g_state.current_chrom_index = chrom_idx;
                    g_state.total_chromosomes = total;
                    g_state.chromosome_progress = chrom_progress;
                    g_state.overall_progress = overall_progress;
                }
            );

            bool success = imputer.run();

            if (success) {
                g_state.completed = true;
                add_log("Imputation completed successfully!");
            } else {
                g_state.failed = true;
                add_log("Imputation completed with some failures");
            }

        } else {
            // Standard single-pass imputation
            add_log("Loading reference panel...");
            auto reference = ReferencePanel::load_vcf(g_state.reference_path);
            add_log("  " + std::to_string(reference->num_samples()) + " samples, " +
                    std::to_string(reference->num_markers()) + " markers");

            add_log("Loading target data...");
            auto targets = TargetData::load_vcf(g_state.target_path);
            add_log("  " + std::to_string(targets->num_samples()) + " samples, " +
                    std::to_string(targets->num_markers()) + " markers");

            g_state.samples_processed = targets->num_samples();
            g_state.markers_processed = reference->num_markers();

            add_log("Initializing imputer...");
            Imputer imputer(*reference, config);

            add_log("Building PBWT index...");
            imputer.build_index();

            add_log("Running imputation...");
            auto result = imputer.impute_with_progress(
                *targets,
                [](uint32_t completed, uint32_t total) {
                    g_state.overall_progress = static_cast<double>(completed) / total;
                }
            );

            add_log("Writing output...");
            result->write_vcf(g_state.output_path, *targets, *reference, config);

            g_state.completed = true;
            add_log("Imputation completed successfully!");
        }

    } catch (const std::exception& e) {
        g_state.failed = true;
        g_state.error_message = e.what();
        add_log("ERROR: " + std::string(e.what()));
    }

    g_state.running = false;
}

// ============================================================================
// UI Components
// ============================================================================

// File browser component
Component FileBrowser(std::string* selected_path, const std::string& title) {
    static std::string current_dir = fs::current_path().string();
    static int selected_index = 0;
    static std::vector<std::string> entries;

    // Refresh entries
    entries = list_directory(current_dir);

    auto menu = Menu(&entries, &selected_index);

    return Renderer(menu, [=] {
        auto content = vbox({
            text(title) | bold,
            separator(),
            text("Directory: " + current_dir) | dim,
            separator(),
            menu->Render() | vscroll_indicator | frame | size(HEIGHT, LESS_THAN, 15),
            separator(),
            text("Selected: " + (selected_path->empty() ? "(none)" : *selected_path)) | dim,
        });

        return window(text(" File Browser "), content);
    }) | CatchEvent([=](Event event) {
        if (event == Event::Return && selected_index < static_cast<int>(entries.size())) {
            std::string entry = entries[selected_index];

            if (entry == "..") {
                current_dir = fs::path(current_dir).parent_path().string();
                entries = list_directory(current_dir);
                selected_index = 0;
            } else if (entry.back() == '/') {
                current_dir = (fs::path(current_dir) / entry.substr(0, entry.size() - 1)).string();
                entries = list_directory(current_dir);
                selected_index = 0;
            } else {
                *selected_path = (fs::path(current_dir) / entry).string();
            }
            return true;
        }
        return false;
    });
}

// Configuration panel
Component ConfigPanel() {
    static std::vector<std::string> presets = {
        "Default", "RAD-seq", "Small Reference", "Biobank",
        "Low Memory", "High Accuracy", "Large Panel (1000G)"
    };

    auto preset_dropdown = Dropdown(&presets, &g_state.preset_index);

    auto states_slider = Slider("HMM States: ", &g_state.num_states, 4, 64, 1);
    auto batch_slider = Slider("Batch Size: ", &g_state.batch_size, 10, 500, 10);
    auto ne_slider = Slider("Ne: ", &g_state.ne, 1000, 50000, 1000);

    auto deterministic_checkbox = Checkbox("Deterministic mode", &g_state.deterministic);
    auto batch_mode_checkbox = Checkbox("Batch mode (per-chromosome)", &g_state.batch_mode);
    auto rebuild_pbwt_checkbox = Checkbox("Rebuild PBWT per window", &g_state.rebuild_pbwt);

    auto container = Container::Vertical({
        preset_dropdown,
        states_slider,
        batch_slider,
        ne_slider,
        deterministic_checkbox,
        batch_mode_checkbox,
        rebuild_pbwt_checkbox,
    });

    return Renderer(container, [=] {
        return vbox({
            text(" Configuration ") | bold | center,
            separator(),
            hbox({text("Preset: "), preset_dropdown->Render()}),
            separator(),
            states_slider->Render(),
            batch_slider->Render(),
            ne_slider->Render(),
            separator(),
            deterministic_checkbox->Render(),
            batch_mode_checkbox->Render(),
            rebuild_pbwt_checkbox->Render(),
        }) | border;
    });
}

// Progress panel
Component ProgressPanel() {
    return Renderer([=] {
        double overall = g_state.overall_progress.load();
        double chrom = g_state.chromosome_progress.load();

        auto elapsed = std::chrono::steady_clock::now() - g_state.start_time;
        auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed);

        // Estimate remaining time
        std::string eta = "calculating...";
        if (overall > 0.01) {
            auto total_estimated = elapsed_sec.count() / overall;
            auto remaining = static_cast<int>(total_estimated * (1.0 - overall));
            eta = format_duration(std::chrono::seconds(remaining));
        }

        std::string status;
        Color status_color = Color::White;
        if (g_state.completed) {
            status = "COMPLETED";
            status_color = Color::Green;
        } else if (g_state.failed) {
            status = "FAILED";
            status_color = Color::Red;
        } else if (g_state.running) {
            status = "RUNNING";
            status_color = Color::Yellow;
        } else {
            status = "READY";
            status_color = Color::Blue;
        }

        Elements progress_elements = {
            hbox({
                text("Status: "),
                text(status) | color(status_color) | bold,
            }),
            separator(),
        };

        if (g_state.running || g_state.completed || g_state.failed) {
            progress_elements.push_back(
                hbox({
                    text("Overall: "),
                    gauge(overall) | flex,
                    text(" " + std::to_string(static_cast<int>(overall * 100)) + "%"),
                })
            );

            if (g_state.batch_mode && !g_state.current_chromosome.empty()) {
                progress_elements.push_back(
                    hbox({
                        text("Chromosome " + g_state.current_chromosome + ": "),
                        gauge(chrom) | flex,
                        text(" " + std::to_string(static_cast<int>(chrom * 100)) + "%"),
                    })
                );
                progress_elements.push_back(
                    text("  (" + std::to_string(g_state.current_chrom_index + 1) + "/" +
                         std::to_string(g_state.total_chromosomes) + " chromosomes)")
                );
            }

            progress_elements.push_back(separator());
            progress_elements.push_back(
                hbox({text("Elapsed: "), text(format_duration(elapsed_sec))})
            );

            if (g_state.running) {
                progress_elements.push_back(
                    hbox({text("ETA: "), text(eta)})
                );
            }
        }

        if (!g_state.error_message.empty()) {
            progress_elements.push_back(separator());
            progress_elements.push_back(
                text("Error: " + g_state.error_message) | color(Color::Red)
            );
        }

        return vbox(progress_elements) | border;
    });
}

// GPU status panel
Component GPUPanel() {
    return Renderer([=] {
        double mem_usage = 0;
        if (g_state.gpu_memory_total > 0) {
            mem_usage = static_cast<double>(g_state.gpu_memory_used) / g_state.gpu_memory_total;
        }

        return vbox({
            text(" GPU Status ") | bold | center,
            separator(),
            text(g_state.gpu_name),
            hbox({
                text("Memory: "),
                gauge(mem_usage) | flex,
                text(" " + format_bytes(g_state.gpu_memory_used) + "/" +
                     format_bytes(g_state.gpu_memory_total)),
            }),
        }) | border;
    });
}

// Log panel
Component LogPanel() {
    return Renderer([=] {
        Elements log_elements;
        {
            std::lock_guard<std::mutex> lock(g_state.log_mutex);
            for (const auto& msg : g_state.log_messages) {
                Color msg_color = Color::White;
                if (msg.find("ERROR") != std::string::npos) {
                    msg_color = Color::Red;
                } else if (msg.find("WARNING") != std::string::npos) {
                    msg_color = Color::Yellow;
                } else if (msg.find("completed") != std::string::npos) {
                    msg_color = Color::Green;
                }
                log_elements.push_back(text(msg) | color(msg_color));
            }
        }

        if (log_elements.empty()) {
            log_elements.push_back(text("(no messages)") | dim);
        }

        return vbox({
            text(" Log ") | bold | center,
            separator(),
            vbox(log_elements) | vscroll_indicator | frame | flex,
        }) | border;
    });
}

// File path display component
Component FilePathDisplay() {
    return Renderer([=] {
        auto path_row = [](const std::string& label, const std::string& path) {
            std::string display = path.empty() ? "(not selected)" : path;
            Color c = path.empty() ? Color::Red : Color::Green;
            return hbox({
                text(label) | size(WIDTH, EQUAL, 12),
                text(display) | color(c),
            });
        };

        return vbox({
            text(" Files ") | bold | center,
            separator(),
            path_row("Reference:", g_state.reference_path),
            path_row("Target:", g_state.target_path),
            path_row("Output:", g_state.output_path),
        }) | border;
    });
}

// ============================================================================
// Main Application
// ============================================================================

int main() {
    // Initialize
    detect_gpus();
    add_log("SwiftImpute UI started");
    add_log("GPU: " + g_state.gpu_name);

    // Create screen
    auto screen = ScreenInteractive::Fullscreen();

    // Tab selection
    int tab_index = 0;
    std::vector<std::string> tab_names = {"Files", "Configure", "Run", "Help"};
    auto tab_menu = Toggle(&tab_names, &tab_index);

    // File selection inputs
    auto ref_input = Input(&g_state.reference_path, "Reference VCF path");
    auto target_input = Input(&g_state.target_path, "Target VCF path");
    auto output_input = Input(&g_state.output_path, "Output VCF path");
    auto checkpoint_input = Input(&g_state.checkpoint_path, "Checkpoint path (optional)");

    // Run button
    auto run_button = Button("Start Imputation", [&] {
        if (g_state.running) {
            add_log("Imputation already running!");
            return;
        }

        if (g_state.reference_path.empty() || g_state.target_path.empty() ||
            g_state.output_path.empty()) {
            add_log("ERROR: Please select all required files");
            return;
        }

        // Start imputation in background thread
        std::thread(run_imputation).detach();
    });

    // Files tab content
    auto files_tab = Container::Vertical({
        Renderer([=] { return text(" Input Files ") | bold; }),
        ref_input,
        target_input,
        output_input,
        checkpoint_input,
    });

    // Configure tab
    auto config_tab = ConfigPanel();

    // Run tab content
    auto run_tab = Container::Vertical({
        FilePathDisplay(),
        ProgressPanel(),
        run_button,
        GPUPanel(),
        LogPanel(),
    });

    // Help tab content
    auto help_tab = Renderer([=] {
        return vbox({
            text(" SwiftImpute Help ") | bold | center,
            separator(),
            text(""),
            text("SwiftImpute is a GPU-accelerated genotype imputation tool.") | bold,
            text(""),
            text("Quick Start:"),
            text("  1. Go to 'Files' tab and enter paths to your VCF files"),
            text("  2. Go to 'Configure' tab to adjust settings"),
            text("  3. Go to 'Run' tab and click 'Start Imputation'"),
            text(""),
            text("Presets:"),
            text("  - Default: Standard settings for most use cases"),
            text("  - RAD-seq: Optimized for sparse RAD-seq data"),
            text("  - Small Reference: For reference panels < 1000 samples"),
            text("  - Biobank: For large biobank-scale data"),
            text("  - Low Memory: Minimize GPU memory usage (4-6 GB)"),
            text("  - High Accuracy: Maximum accuracy (slower)"),
            text("  - Large Panel: For 1000 Genomes or similar (>2000 samples)"),
            text(""),
            text("Tips:"),
            text("  - Enable 'Batch mode' for automatic checkpointing"),
            text("  - Use 'Rebuild PBWT per window' for large reference panels"),
            text("  - Increase 'HMM States' for better accuracy"),
            text(""),
            text("Keyboard shortcuts:"),
            text("  Tab      - Switch between input fields"),
            text("  Ctrl+C   - Exit application"),
            text("  Enter    - Activate buttons/selections"),
            text(""),
            text("For more help, visit: https://github.com/your-org/swiftimpute"),
        }) | border;
    });

    // Tab container
    auto tab_content = Container::Tab({
        files_tab,
        config_tab,
        run_tab,
        help_tab,
    }, &tab_index);

    // Main layout
    auto main_container = Container::Vertical({
        tab_menu,
        tab_content,
    });

    // Renderer
    auto main_renderer = Renderer(main_container, [&] {
        return vbox({
            // Header
            hbox({
                text("SwiftImpute") | bold | color(Color::Cyan),
                filler(),
                text("GPU-Accelerated Genotype Imputation") | dim,
            }) | border,

            // Tab selector
            tab_menu->Render() | center,

            separator(),

            // Tab content
            tab_content->Render() | flex,

            // Footer
            hbox({
                text(" Press Ctrl+C to exit ") | dim,
                filler(),
                text(" v1.0.0 ") | dim,
            }),
        });
    });

    // Refresh loop for progress updates
    std::atomic<bool> refresh_ui{true};
    std::thread refresh_thread([&] {
        while (refresh_ui) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });

    // Run the UI
    screen.Loop(main_renderer);

    // Cleanup
    refresh_ui = false;
    refresh_thread.join();

    return 0;
}
