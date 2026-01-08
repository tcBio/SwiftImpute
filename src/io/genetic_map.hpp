#pragma once

#include "core/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace swiftimpute {
namespace io {

/**
 * GeneticMap: Handles genetic map loading and interpolation
 *
 * Supports multiple file formats:
 * - PLINK .map format: chrom  id  cM  bp
 * - HapMap format: position(bp)  rate(cM/Mb)  genetic_position(cM)
 * - SHAPEIT/IMPUTE2 format: position  rate  genetic_position
 * - BEAGLE format: chrom  position(bp)  genetic_position(cM)
 *
 * Provides linear interpolation for positions not in the map.
 *
 * Usage:
 *   GeneticMap map;
 *   map.load("genetic_map_chr1.txt", GeneticMapFormat::HAPMAP);
 *   double cM = map.interpolate("chr1", 12345678);
 *   map.apply_to_markers(markers);
 */

enum class GeneticMapFormat {
    AUTO,       // Auto-detect from file content
    PLINK,      // 4 columns: chrom id cM bp
    HAPMAP,     // 3 columns: position rate(cM/Mb) cM (header line)
    SHAPEIT,    // 3 columns: pos rate cM (no header)
    BEAGLE      // 3 columns: chrom position cM
};

/**
 * Entry in the genetic map
 */
struct GeneticMapEntry {
    uint64_t position;      // Physical position (bp)
    double rate;            // Recombination rate (cM/Mb), optional
    double cM;              // Genetic position (cM)
};

/**
 * Chromosome-specific genetic map data
 */
struct ChromosomeMap {
    std::string chrom;
    std::vector<GeneticMapEntry> entries;

    // Check if map is valid
    bool is_valid() const { return !entries.empty(); }

    // Get range
    uint64_t min_pos() const { return entries.empty() ? 0 : entries.front().position; }
    uint64_t max_pos() const { return entries.empty() ? 0 : entries.back().position; }
    double min_cM() const { return entries.empty() ? 0 : entries.front().cM; }
    double max_cM() const { return entries.empty() ? 0 : entries.back().cM; }
};

class GeneticMap {
public:
    GeneticMap() = default;
    ~GeneticMap() = default;

    /**
     * Load a genetic map file
     *
     * @param filename Path to genetic map file
     * @param format File format (AUTO for auto-detection)
     * @param chromosome Optional: only load specific chromosome
     */
    void load(
        const std::string& filename,
        GeneticMapFormat format = GeneticMapFormat::AUTO,
        const std::string& chromosome = ""
    );

    /**
     * Load genetic maps from a directory
     *
     * Expects files named like: genetic_map_chr1.txt, genetic_map_chr2.txt, etc.
     * Or: chr1.map, chr2.map, etc.
     *
     * @param directory Path to directory containing map files
     * @param format File format
     * @param pattern Filename pattern (use {chr} for chromosome placeholder)
     */
    void load_directory(
        const std::string& directory,
        GeneticMapFormat format = GeneticMapFormat::AUTO,
        const std::string& pattern = "genetic_map_{chr}.txt"
    );

    /**
     * Interpolate genetic position for a physical position
     *
     * Uses linear interpolation between flanking map entries.
     * Extrapolates linearly beyond map boundaries.
     *
     * @param chrom Chromosome name
     * @param position Physical position (bp)
     * @return Genetic position in cM
     */
    double interpolate(const std::string& chrom, uint64_t position) const;

    /**
     * Apply genetic map to a vector of markers
     *
     * Updates the cM field of each marker using interpolation.
     *
     * @param markers Vector of markers to update
     */
    void apply_to_markers(std::vector<Marker>& markers) const;

    /**
     * Get genetic distances for a set of marker positions
     *
     * @param chrom Chromosome
     * @param positions Vector of physical positions
     * @return Vector of genetic positions (cM)
     */
    std::vector<double> get_positions(
        const std::string& chrom,
        const std::vector<uint64_t>& positions
    ) const;

    /**
     * Check if map is loaded for a chromosome
     */
    bool has_chromosome(const std::string& chrom) const;

    /**
     * Get list of loaded chromosomes
     */
    std::vector<std::string> chromosomes() const;

    /**
     * Get chromosome map data
     */
    const ChromosomeMap* get_chromosome_map(const std::string& chrom) const;

    /**
     * Get total number of map entries
     */
    size_t total_entries() const;

    /**
     * Check if any map is loaded
     */
    bool is_loaded() const { return !chromosome_maps_.empty(); }

    /**
     * Clear all loaded maps
     */
    void clear() { chromosome_maps_.clear(); }

    /**
     * Estimate uniform recombination rate from map
     *
     * Useful for regions without detailed map data.
     *
     * @param chrom Chromosome
     * @return Average cM per Mb
     */
    double estimate_avg_rate(const std::string& chrom) const;

private:
    std::map<std::string, ChromosomeMap> chromosome_maps_;

    // File format detection
    GeneticMapFormat detect_format(const std::string& filename) const;

    // Format-specific parsers
    void parse_plink_format(std::istream& input, const std::string& filter_chrom);
    void parse_hapmap_format(std::istream& input, const std::string& chrom);
    void parse_shapeit_format(std::istream& input, const std::string& chrom);
    void parse_beagle_format(std::istream& input, const std::string& filter_chrom);

    // Helper to normalize chromosome names
    static std::string normalize_chrom(const std::string& chrom);

    // Binary search for position
    size_t find_flanking_index(const ChromosomeMap& map, uint64_t position) const;
};

/**
 * Convenience function to load a genetic map
 */
std::unique_ptr<GeneticMap> load_genetic_map(
    const std::string& filename,
    GeneticMapFormat format = GeneticMapFormat::AUTO
);

} // namespace io
} // namespace swiftimpute
