#include "genetic_map.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <filesystem>

namespace swiftimpute {
namespace io {

namespace fs = std::filesystem;

// Helper to normalize chromosome names (chr1 -> 1, CHR1 -> 1, etc.)
std::string GeneticMap::normalize_chrom(const std::string& chrom) {
    std::string result = chrom;

    // Remove "chr" prefix (case-insensitive)
    if (result.size() >= 3) {
        std::string prefix = result.substr(0, 3);
        for (auto& c : prefix) c = std::tolower(c);
        if (prefix == "chr") {
            result = result.substr(3);
        }
    }

    return result;
}

GeneticMapFormat GeneticMap::detect_format(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open genetic map file: " + filename);
    }

    std::string line;
    std::getline(file, line);

    // Skip empty lines
    while (line.empty() || line[0] == '#') {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Empty genetic map file: " + filename);
        }
    }

    // Count columns
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    // Check for HapMap header
    if (tokens.size() >= 3 &&
        (tokens[0] == "position" || tokens[0] == "Position" ||
         tokens[0] == "pos" || tokens[0] == "Pos")) {
        return GeneticMapFormat::HAPMAP;
    }

    // 4 columns: likely PLINK format
    if (tokens.size() == 4) {
        // Check if column 3 looks like a genetic position
        try {
            std::stod(tokens[2]);
            std::stoll(tokens[3]);
            return GeneticMapFormat::PLINK;
        } catch (...) {}
    }

    // 3 columns: could be SHAPEIT or BEAGLE
    if (tokens.size() == 3) {
        // Check if first column is numeric (SHAPEIT) or has letters (BEAGLE)
        bool first_is_numeric = true;
        for (char c : tokens[0]) {
            if (!std::isdigit(c)) {
                first_is_numeric = false;
                break;
            }
        }

        if (first_is_numeric) {
            return GeneticMapFormat::SHAPEIT;
        } else {
            return GeneticMapFormat::BEAGLE;
        }
    }

    // Default to PLINK
    LOG_WARNING("Could not detect genetic map format, assuming PLINK format");
    return GeneticMapFormat::PLINK;
}

void GeneticMap::load(
    const std::string& filename,
    GeneticMapFormat format,
    const std::string& chromosome
) {
    LOG_INFO("Loading genetic map from: " + filename);

    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open genetic map file: " + filename);
    }

    // Auto-detect format if needed
    if (format == GeneticMapFormat::AUTO) {
        format = detect_format(filename);
        LOG_INFO("Auto-detected genetic map format: " +
                 std::string(format == GeneticMapFormat::PLINK ? "PLINK" :
                            format == GeneticMapFormat::HAPMAP ? "HapMap" :
                            format == GeneticMapFormat::SHAPEIT ? "SHAPEIT" :
                            "BEAGLE"));
    }

    // Parse based on format
    switch (format) {
        case GeneticMapFormat::PLINK:
            parse_plink_format(file, chromosome);
            break;
        case GeneticMapFormat::HAPMAP:
            parse_hapmap_format(file, chromosome);
            break;
        case GeneticMapFormat::SHAPEIT:
            parse_shapeit_format(file, chromosome);
            break;
        case GeneticMapFormat::BEAGLE:
            parse_beagle_format(file, chromosome);
            break;
        default:
            throw std::runtime_error("Unknown genetic map format");
    }

    // Sort entries by position for each chromosome
    for (auto& [chrom, map] : chromosome_maps_) {
        std::sort(map.entries.begin(), map.entries.end(),
            [](const GeneticMapEntry& a, const GeneticMapEntry& b) {
                return a.position < b.position;
            });

        LOG_INFO("Loaded " + std::to_string(map.entries.size()) +
                 " map entries for chromosome " + chrom);
    }
}

void GeneticMap::parse_plink_format(std::istream& input, const std::string& filter_chrom) {
    // PLINK format: chrom  id  cM  bp
    std::string line;
    std::string norm_filter = filter_chrom.empty() ? "" : normalize_chrom(filter_chrom);

    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string chrom, id;
        double cM;
        uint64_t pos;

        if (!(iss >> chrom >> id >> cM >> pos)) {
            continue;  // Skip malformed lines
        }

        std::string norm_chrom = normalize_chrom(chrom);

        // Filter by chromosome if specified
        if (!norm_filter.empty() && norm_chrom != norm_filter) {
            continue;
        }

        // Add entry
        GeneticMapEntry entry;
        entry.position = pos;
        entry.cM = cM;
        entry.rate = 0.0;  // Not provided in PLINK format

        chromosome_maps_[norm_chrom].chrom = norm_chrom;
        chromosome_maps_[norm_chrom].entries.push_back(entry);
    }
}

void GeneticMap::parse_hapmap_format(std::istream& input, const std::string& chrom) {
    // HapMap format: position rate(cM/Mb) cM
    // Has header line
    std::string line;

    // Skip header
    std::getline(input, line);

    std::string norm_chrom = chrom.empty() ? "unknown" : normalize_chrom(chrom);

    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        uint64_t pos;
        double rate, cM;

        if (!(iss >> pos >> rate >> cM)) {
            continue;
        }

        GeneticMapEntry entry;
        entry.position = pos;
        entry.rate = rate;
        entry.cM = cM;

        chromosome_maps_[norm_chrom].chrom = norm_chrom;
        chromosome_maps_[norm_chrom].entries.push_back(entry);
    }
}

void GeneticMap::parse_shapeit_format(std::istream& input, const std::string& chrom) {
    // SHAPEIT format: pos rate cM (no header)
    std::string line;
    std::string norm_chrom = chrom.empty() ? "unknown" : normalize_chrom(chrom);

    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        uint64_t pos;
        double rate, cM;

        if (!(iss >> pos >> rate >> cM)) {
            continue;
        }

        GeneticMapEntry entry;
        entry.position = pos;
        entry.rate = rate;
        entry.cM = cM;

        chromosome_maps_[norm_chrom].chrom = norm_chrom;
        chromosome_maps_[norm_chrom].entries.push_back(entry);
    }
}

void GeneticMap::parse_beagle_format(std::istream& input, const std::string& filter_chrom) {
    // BEAGLE format: chrom position cM
    std::string line;
    std::string norm_filter = filter_chrom.empty() ? "" : normalize_chrom(filter_chrom);

    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string chrom;
        uint64_t pos;
        double cM;

        if (!(iss >> chrom >> pos >> cM)) {
            continue;
        }

        std::string norm_chrom = normalize_chrom(chrom);

        if (!norm_filter.empty() && norm_chrom != norm_filter) {
            continue;
        }

        GeneticMapEntry entry;
        entry.position = pos;
        entry.cM = cM;
        entry.rate = 0.0;

        chromosome_maps_[norm_chrom].chrom = norm_chrom;
        chromosome_maps_[norm_chrom].entries.push_back(entry);
    }
}

void GeneticMap::load_directory(
    const std::string& directory,
    GeneticMapFormat format,
    const std::string& pattern
) {
    LOG_INFO("Loading genetic maps from directory: " + directory);

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error("Directory does not exist: " + directory);
    }

    // Common chromosome names to look for
    std::vector<std::string> chroms = {
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "X", "Y", "MT"
    };

    int loaded = 0;

    for (const auto& chr : chroms) {
        // Try different filename patterns
        std::vector<std::string> patterns_to_try;

        // User-specified pattern
        std::string user_pattern = pattern;
        size_t pos = user_pattern.find("{chr}");
        if (pos != std::string::npos) {
            user_pattern.replace(pos, 5, chr);
        }
        patterns_to_try.push_back(user_pattern);

        // Common patterns
        patterns_to_try.push_back("genetic_map_chr" + chr + ".txt");
        patterns_to_try.push_back("genetic_map_chr" + chr + ".txt.gz");
        patterns_to_try.push_back("chr" + chr + ".map");
        patterns_to_try.push_back(chr + ".map");
        patterns_to_try.push_back("genetic_map_chr" + chr + "_b37.txt");
        patterns_to_try.push_back("genetic_map_chr" + chr + "_b38.txt");

        for (const auto& fname : patterns_to_try) {
            fs::path filepath = fs::path(directory) / fname;
            if (fs::exists(filepath)) {
                try {
                    load(filepath.string(), format, chr);
                    loaded++;
                    break;  // Found file for this chromosome
                } catch (const std::exception& e) {
                    LOG_WARNING("Failed to load " + filepath.string() + ": " + e.what());
                }
            }
        }
    }

    if (loaded == 0) {
        throw std::runtime_error("No genetic map files found in directory: " + directory);
    }

    LOG_INFO("Loaded genetic maps for " + std::to_string(loaded) + " chromosomes");
}

size_t GeneticMap::find_flanking_index(const ChromosomeMap& map, uint64_t position) const {
    // Binary search for the largest entry with position <= target
    const auto& entries = map.entries;

    if (entries.empty()) {
        return 0;
    }

    // Find first entry with position > target
    auto it = std::upper_bound(entries.begin(), entries.end(), position,
        [](uint64_t pos, const GeneticMapEntry& entry) {
            return pos < entry.position;
        });

    // Return index of entry before it (or 0 if at beginning)
    if (it == entries.begin()) {
        return 0;
    }

    return std::distance(entries.begin(), it) - 1;
}

double GeneticMap::interpolate(const std::string& chrom, uint64_t position) const {
    std::string norm_chrom = normalize_chrom(chrom);

    auto it = chromosome_maps_.find(norm_chrom);
    if (it == chromosome_maps_.end()) {
        // No map for this chromosome - use default rate of 1 cM/Mb
        return position / 1e6;
    }

    const ChromosomeMap& map = it->second;
    if (map.entries.empty()) {
        return position / 1e6;
    }

    // Handle positions outside map range
    if (position <= map.entries.front().position) {
        // Extrapolate before first entry
        if (map.entries.size() > 1) {
            double rate = (map.entries[1].cM - map.entries[0].cM) /
                         static_cast<double>(map.entries[1].position - map.entries[0].position);
            double delta = static_cast<double>(map.entries[0].position - position);
            return map.entries[0].cM - rate * delta;
        }
        return map.entries[0].cM;
    }

    if (position >= map.entries.back().position) {
        // Extrapolate after last entry
        size_t n = map.entries.size();
        if (n > 1) {
            double rate = (map.entries[n-1].cM - map.entries[n-2].cM) /
                         static_cast<double>(map.entries[n-1].position - map.entries[n-2].position);
            double delta = static_cast<double>(position - map.entries[n-1].position);
            return map.entries[n-1].cM + rate * delta;
        }
        return map.entries.back().cM;
    }

    // Linear interpolation between flanking entries
    size_t idx = find_flanking_index(map, position);
    const GeneticMapEntry& left = map.entries[idx];
    const GeneticMapEntry& right = map.entries[idx + 1];

    double frac = static_cast<double>(position - left.position) /
                  static_cast<double>(right.position - left.position);

    return left.cM + frac * (right.cM - left.cM);
}

void GeneticMap::apply_to_markers(std::vector<Marker>& markers) const {
    size_t updated = 0;

    for (auto& marker : markers) {
        double cM = interpolate(marker.chrom, marker.pos);
        marker.cM = cM;
        updated++;
    }

    LOG_INFO("Applied genetic map to " + std::to_string(updated) + " markers");
}

std::vector<double> GeneticMap::get_positions(
    const std::string& chrom,
    const std::vector<uint64_t>& positions
) const {
    std::vector<double> result;
    result.reserve(positions.size());

    for (uint64_t pos : positions) {
        result.push_back(interpolate(chrom, pos));
    }

    return result;
}

bool GeneticMap::has_chromosome(const std::string& chrom) const {
    return chromosome_maps_.find(normalize_chrom(chrom)) != chromosome_maps_.end();
}

std::vector<std::string> GeneticMap::chromosomes() const {
    std::vector<std::string> result;
    for (const auto& [chrom, _] : chromosome_maps_) {
        result.push_back(chrom);
    }
    return result;
}

const ChromosomeMap* GeneticMap::get_chromosome_map(const std::string& chrom) const {
    auto it = chromosome_maps_.find(normalize_chrom(chrom));
    if (it == chromosome_maps_.end()) {
        return nullptr;
    }
    return &it->second;
}

size_t GeneticMap::total_entries() const {
    size_t total = 0;
    for (const auto& [_, map] : chromosome_maps_) {
        total += map.entries.size();
    }
    return total;
}

double GeneticMap::estimate_avg_rate(const std::string& chrom) const {
    auto it = chromosome_maps_.find(normalize_chrom(chrom));
    if (it == chromosome_maps_.end() || it->second.entries.size() < 2) {
        return 1.0;  // Default 1 cM/Mb
    }

    const ChromosomeMap& map = it->second;
    double total_cM = map.max_cM() - map.min_cM();
    double total_Mb = static_cast<double>(map.max_pos() - map.min_pos()) / 1e6;

    if (total_Mb <= 0) {
        return 1.0;
    }

    return total_cM / total_Mb;
}

std::unique_ptr<GeneticMap> load_genetic_map(
    const std::string& filename,
    GeneticMapFormat format
) {
    auto map = std::make_unique<GeneticMap>();
    map->load(filename, format);
    return map;
}

} // namespace io
} // namespace swiftimpute
