#include "binary_format.hpp"
#include "vcf_reader.hpp"
#include <fstream>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace swiftimpute {
namespace io {

// =============================================================================
// MappedReferencePanel Implementation
// =============================================================================

MappedReferencePanel::MappedReferencePanel(
    const std::string& filename,
    const MMapConfig& config
) : filename_(filename), file_(filename, config) {
    if (file_.is_open()) {
        parse_header();
        build_chromosome_index();
    }
}

void MappedReferencePanel::parse_header() {
    if (file_.size() < sizeof(RefPanelHeader)) {
        throw std::runtime_error("Invalid binary reference file: too small");
    }

    std::memcpy(&header_, file_.data(), sizeof(RefPanelHeader));

    if (!header_.base.is_valid(REF_PANEL_MAGIC)) {
        throw std::runtime_error("Invalid binary reference file: bad magic or version");
    }
}

void MappedReferencePanel::build_chromosome_index() {
    if (header_.chrom_index_offset == 0) {
        return;  // No chromosome index
    }

    const char* data = file_.data() + header_.base.data_offset + header_.chrom_index_offset;

    // Read number of chromosomes
    uint32_t num_chroms;
    std::memcpy(&num_chroms, data, sizeof(num_chroms));
    data += sizeof(num_chroms);

    // Read chromosome names and ranges
    for (uint32_t i = 0; i < num_chroms; ++i) {
        uint32_t name_len;
        std::memcpy(&name_len, data, sizeof(name_len));
        data += sizeof(name_len);

        std::string name(data, name_len);
        data += name_len;

        uint32_t start_idx, end_idx;
        std::memcpy(&start_idx, data, sizeof(start_idx));
        data += sizeof(start_idx);
        std::memcpy(&end_idx, data, sizeof(end_idx));
        data += sizeof(end_idx);

        chromosomes_.push_back(name);
        chrom_ranges_.push_back({start_idx, end_idx});
    }
}

Marker MappedReferencePanel::get_marker(uint32_t idx) const {
    if (idx >= header_.num_markers) {
        throw std::out_of_range("Marker index out of range");
    }

    const char* markers_data = file_.data() + header_.base.data_offset + header_.markers_offset;

    // Find marker record (variable-length due to allele strings)
    // For simplicity, we use fixed-size records with max allele length
    // In production, this would use an index for variable-length records

    size_t record_size = sizeof(BinaryMarker) + 256;  // Max allele space
    const BinaryMarker* record = reinterpret_cast<const BinaryMarker*>(
        markers_data + idx * record_size);

    Marker marker;
    marker.pos = record->pos;
    marker.cM = record->cM;

    if (record->chrom_idx < chromosomes_.size()) {
        marker.chrom = chromosomes_[record->chrom_idx];
    }

    // Read alleles
    const char* allele_data = reinterpret_cast<const char*>(record + 1);
    marker.ref = std::string(allele_data, record->ref_len);
    marker.alt = std::string(allele_data + record->ref_len, record->alt_len);

    return marker;
}

std::vector<std::string> MappedReferencePanel::get_chromosomes() const {
    return chromosomes_;
}

void MappedReferencePanel::get_chromosome_range(
    const std::string& chrom,
    uint32_t& start_idx,
    uint32_t& end_idx
) const {
    for (size_t i = 0; i < chromosomes_.size(); ++i) {
        if (chromosomes_[i] == chrom) {
            start_idx = chrom_ranges_[i].first;
            end_idx = chrom_ranges_[i].second;
            return;
        }
    }

    // Chromosome not found
    start_idx = 0;
    end_idx = 0;
}

const allele_t* MappedReferencePanel::haplotypes(
    uint32_t start_marker,
    uint32_t end_marker
) const {
    if (start_marker >= header_.num_markers || end_marker > header_.num_markers) {
        throw std::out_of_range("Marker range out of bounds");
    }

    const char* hap_data = file_.data() + header_.base.data_offset + header_.haplotypes_offset;
    size_t offset = static_cast<size_t>(start_marker) * header_.num_haplotypes;

    return reinterpret_cast<const allele_t*>(hap_data + offset);
}

void MappedReferencePanel::copy_haplotypes(
    uint32_t start_marker,
    uint32_t end_marker,
    allele_t* buffer
) const {
    const allele_t* src = haplotypes(start_marker, end_marker);
    size_t size = static_cast<size_t>(end_marker - start_marker) * header_.num_haplotypes;
    std::memcpy(buffer, src, size * sizeof(allele_t));
}

allele_t MappedReferencePanel::get_allele(uint32_t marker, uint32_t haplotype) const {
    const char* hap_data = file_.data() + header_.base.data_offset + header_.haplotypes_offset;
    size_t offset = static_cast<size_t>(marker) * header_.num_haplotypes + haplotype;
    return static_cast<allele_t>(hap_data[offset]);
}

// =============================================================================
// MappedPBWTIndex Implementation
// =============================================================================

MappedPBWTIndex::MappedPBWTIndex(
    const std::string& filename,
    const MMapConfig& config
) : filename_(filename), file_(filename, config) {
    if (file_.is_open()) {
        parse_header();
    }
}

void MappedPBWTIndex::parse_header() {
    if (file_.size() < sizeof(PBWTHeader)) {
        throw std::runtime_error("Invalid PBWT index file: too small");
    }

    std::memcpy(&header_, file_.data(), sizeof(PBWTHeader));

    if (!header_.base.is_valid(PBWT_INDEX_MAGIC)) {
        throw std::runtime_error("Invalid PBWT index file: bad magic or version");
    }

    // Validate data type sizes match current compilation
    if (header_.haplotype_bytes != sizeof(haplotype_t) ||
        header_.marker_bytes != sizeof(marker_t)) {
        throw std::runtime_error("PBWT index compiled with different type sizes");
    }
}

haplotype_t MappedPBWTIndex::prefix_at(uint32_t marker, uint32_t pos) const {
    const char* data = file_.data() + header_.base.data_offset + header_.prefix_offset;
    size_t offset = static_cast<size_t>(marker) * header_.num_haplotypes + pos;
    return reinterpret_cast<const haplotype_t*>(data)[offset];
}

marker_t MappedPBWTIndex::divergence_at(uint32_t marker, uint32_t pos) const {
    const char* data = file_.data() + header_.base.data_offset + header_.divergence_offset;
    size_t offset = static_cast<size_t>(marker) * header_.num_haplotypes + pos;
    return reinterpret_cast<const marker_t*>(data)[offset];
}

const haplotype_t* MappedPBWTIndex::prefix_data(
    uint32_t start_marker,
    uint32_t end_marker
) const {
    (void)end_marker;  // Only used for validation
    const char* data = file_.data() + header_.base.data_offset + header_.prefix_offset;
    size_t offset = static_cast<size_t>(start_marker) * header_.num_haplotypes;
    return reinterpret_cast<const haplotype_t*>(data) + offset;
}

const marker_t* MappedPBWTIndex::divergence_data(
    uint32_t start_marker,
    uint32_t end_marker
) const {
    (void)end_marker;  // Only used for validation
    const char* data = file_.data() + header_.base.data_offset + header_.divergence_offset;
    size_t offset = static_cast<size_t>(start_marker) * header_.num_haplotypes;
    return reinterpret_cast<const marker_t*>(data) + offset;
}

void MappedPBWTIndex::copy_prefix(
    uint32_t start_marker,
    uint32_t end_marker,
    haplotype_t* buffer
) const {
    const haplotype_t* src = prefix_data(start_marker, end_marker);
    size_t count = static_cast<size_t>(end_marker - start_marker) * header_.num_haplotypes;
    std::memcpy(buffer, src, count * sizeof(haplotype_t));
}

void MappedPBWTIndex::copy_divergence(
    uint32_t start_marker,
    uint32_t end_marker,
    marker_t* buffer
) const {
    const marker_t* src = divergence_data(start_marker, end_marker);
    size_t count = static_cast<size_t>(end_marker - start_marker) * header_.num_haplotypes;
    std::memcpy(buffer, src, count * sizeof(marker_t));
}

size_t MappedPBWTIndex::memory_usage() const {
    // Only the header and OS bookkeeping, actual data is memory-mapped
    return sizeof(*this) + sizeof(PBWTHeader);
}

// =============================================================================
// Write Functions
// =============================================================================

void write_reference_panel_binary(
    const std::string& filename,
    const std::vector<Marker>& markers,
    const std::vector<Sample>& samples,
    const allele_t* haplotypes,
    uint32_t num_haplotypes
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Build chromosome index
    std::vector<std::string> chromosomes;
    std::vector<std::pair<uint32_t, uint32_t>> chrom_ranges;

    std::string current_chrom;
    uint32_t current_start = 0;

    for (uint32_t i = 0; i < markers.size(); ++i) {
        if (markers[i].chrom != current_chrom) {
            if (!current_chrom.empty()) {
                chrom_ranges.push_back({current_start, i});
            }
            current_chrom = markers[i].chrom;
            current_start = i;
            chromosomes.push_back(current_chrom);
        }
    }
    if (!current_chrom.empty()) {
        chrom_ranges.push_back({current_start, static_cast<uint32_t>(markers.size())});
    }

    // Prepare header
    RefPanelHeader header;
    header.num_markers = markers.size();
    header.num_samples = samples.size();
    header.num_haplotypes = num_haplotypes;
    header.base.created_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Calculate offsets
    uint64_t offset = 0;

    // Markers section
    header.markers_offset = offset;
    size_t record_size = sizeof(BinaryMarker) + 256;  // Fixed record size with allele space
    header.markers_size = markers.size() * record_size;
    offset += header.markers_size;

    // Samples section
    header.samples_offset = offset;
    header.samples_size = samples.size() * 256;  // Fixed sample record size
    offset += header.samples_size;

    // Haplotypes section
    header.haplotypes_offset = offset;
    header.haplotypes_size = static_cast<size_t>(markers.size()) * num_haplotypes;
    offset += header.haplotypes_size;

    // Chromosome index section
    header.chrom_index_offset = offset;
    // Calculate chromosome index size
    size_t chrom_index_size = sizeof(uint32_t);  // num_chroms
    for (const auto& chrom : chromosomes) {
        chrom_index_size += sizeof(uint32_t) + chrom.size() + 2 * sizeof(uint32_t);
    }
    header.chrom_index_size = chrom_index_size;
    offset += header.chrom_index_size;

    header.base.data_offset = sizeof(RefPanelHeader);
    header.base.data_size = offset;

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write markers
    std::vector<char> marker_buffer(record_size, 0);
    std::map<std::string, uint16_t> chrom_to_idx;
    for (size_t i = 0; i < chromosomes.size(); ++i) {
        chrom_to_idx[chromosomes[i]] = static_cast<uint16_t>(i);
    }

    for (const auto& marker : markers) {
        std::memset(marker_buffer.data(), 0, record_size);

        BinaryMarker* rec = reinterpret_cast<BinaryMarker*>(marker_buffer.data());
        rec->pos = marker.pos;
        rec->cM = marker.cM;
        rec->chrom_idx = chrom_to_idx.count(marker.chrom) ? chrom_to_idx[marker.chrom] : 0;
        rec->ref_len = std::min(static_cast<size_t>(128), marker.ref.size());
        rec->alt_len = std::min(static_cast<size_t>(128), marker.alt.size());
        rec->flags = 0;

        char* allele_ptr = marker_buffer.data() + sizeof(BinaryMarker);
        std::memcpy(allele_ptr, marker.ref.data(), rec->ref_len);
        std::memcpy(allele_ptr + rec->ref_len, marker.alt.data(), rec->alt_len);

        file.write(marker_buffer.data(), record_size);
    }

    // Write samples
    std::vector<char> sample_buffer(256, 0);
    for (const auto& sample : samples) {
        std::memset(sample_buffer.data(), 0, 256);
        size_t id_len = std::min(sample.id.size(), static_cast<size_t>(255));
        sample_buffer[0] = static_cast<char>(id_len);
        std::memcpy(sample_buffer.data() + 1, sample.id.data(), id_len);
        file.write(sample_buffer.data(), 256);
    }

    // Write haplotypes
    file.write(reinterpret_cast<const char*>(haplotypes),
               static_cast<size_t>(markers.size()) * num_haplotypes);

    // Write chromosome index
    uint32_t num_chroms = chromosomes.size();
    file.write(reinterpret_cast<const char*>(&num_chroms), sizeof(num_chroms));

    for (size_t i = 0; i < chromosomes.size(); ++i) {
        uint32_t name_len = chromosomes[i].size();
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(chromosomes[i].data(), name_len);
        file.write(reinterpret_cast<const char*>(&chrom_ranges[i].first), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&chrom_ranges[i].second), sizeof(uint32_t));
    }

    file.close();
    LOG_INFO("Wrote binary reference panel: " + filename +
             " (" + std::to_string(markers.size()) + " markers, " +
             std::to_string(num_haplotypes) + " haplotypes)");
}

void read_reference_panel_binary(
    const std::string& filename,
    std::vector<Marker>& markers,
    std::vector<Sample>& samples,
    std::unique_ptr<allele_t[]>& haplotypes,
    uint32_t& num_haplotypes
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open binary reference file: " + filename);
    }

    // Read header
    RefPanelHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!header.base.is_valid(REF_PANEL_MAGIC)) {
        throw std::runtime_error("Invalid binary reference file format");
    }

    num_haplotypes = header.num_haplotypes;

    // Read chromosome index first (needed for marker parsing)
    std::vector<std::string> chromosomes;
    if (header.chrom_index_offset > 0) {
        file.seekg(header.base.data_offset + header.chrom_index_offset);

        uint32_t num_chroms;
        file.read(reinterpret_cast<char*>(&num_chroms), sizeof(num_chroms));

        for (uint32_t i = 0; i < num_chroms; ++i) {
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

            std::string name(name_len, '\0');
            file.read(&name[0], name_len);

            uint32_t start_idx, end_idx;
            file.read(reinterpret_cast<char*>(&start_idx), sizeof(start_idx));
            file.read(reinterpret_cast<char*>(&end_idx), sizeof(end_idx));

            chromosomes.push_back(name);
        }
    }

    // Read markers
    file.seekg(header.base.data_offset + header.markers_offset);
    markers.clear();
    markers.reserve(header.num_markers);

    size_t record_size = sizeof(BinaryMarker) + 256;
    std::vector<char> buffer(record_size);

    for (uint32_t i = 0; i < header.num_markers; ++i) {
        file.read(buffer.data(), record_size);

        const BinaryMarker* rec = reinterpret_cast<const BinaryMarker*>(buffer.data());
        Marker marker;
        marker.pos = rec->pos;
        marker.cM = rec->cM;

        if (rec->chrom_idx < chromosomes.size()) {
            marker.chrom = chromosomes[rec->chrom_idx];
        }

        const char* allele_ptr = buffer.data() + sizeof(BinaryMarker);
        marker.ref = std::string(allele_ptr, rec->ref_len);
        marker.alt = std::string(allele_ptr + rec->ref_len, rec->alt_len);

        markers.push_back(marker);
    }

    // Read samples
    file.seekg(header.base.data_offset + header.samples_offset);
    samples.clear();
    samples.reserve(header.num_samples);

    std::vector<char> sample_buffer(256);
    for (uint32_t i = 0; i < header.num_samples; ++i) {
        file.read(sample_buffer.data(), 256);

        Sample sample;
        size_t id_len = static_cast<unsigned char>(sample_buffer[0]);
        sample.id = std::string(sample_buffer.data() + 1, id_len);
        samples.push_back(sample);
    }

    // Read haplotypes
    file.seekg(header.base.data_offset + header.haplotypes_offset);
    size_t hap_size = static_cast<size_t>(header.num_markers) * header.num_haplotypes;
    haplotypes = std::make_unique<allele_t[]>(hap_size);
    file.read(reinterpret_cast<char*>(haplotypes.get()), hap_size);

    LOG_INFO("Loaded binary reference panel: " + filename);
}

void write_pbwt_binary(
    const std::string& filename,
    uint32_t num_markers,
    uint32_t num_haplotypes,
    const haplotype_t* prefix_data,
    const marker_t* divergence_data
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Compute statistics
    double sum_div = 0;
    uint32_t max_div = 0;
    size_t total_entries = static_cast<size_t>(num_markers) * num_haplotypes;

    for (size_t i = 0; i < total_entries; ++i) {
        sum_div += divergence_data[i];
        if (divergence_data[i] > max_div) {
            max_div = divergence_data[i];
        }
    }

    // Prepare header
    PBWTHeader header;
    header.num_markers = num_markers;
    header.num_haplotypes = num_haplotypes;
    header.avg_divergence = sum_div / total_entries;
    header.max_divergence = max_div;
    header.base.created_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Calculate offsets
    header.prefix_offset = 0;
    header.prefix_size = total_entries * sizeof(haplotype_t);

    header.divergence_offset = header.prefix_size;
    header.divergence_size = total_entries * sizeof(marker_t);

    header.base.data_offset = sizeof(PBWTHeader);
    header.base.data_size = header.prefix_size + header.divergence_size;

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write prefix array
    file.write(reinterpret_cast<const char*>(prefix_data), header.prefix_size);

    // Write divergence array
    file.write(reinterpret_cast<const char*>(divergence_data), header.divergence_size);

    file.close();
    LOG_INFO("Wrote PBWT index: " + filename +
             " (" + std::to_string(num_markers) + " markers x " +
             std::to_string(num_haplotypes) + " haplotypes)");
}

void read_pbwt_binary(
    const std::string& filename,
    uint32_t& num_markers,
    uint32_t& num_haplotypes,
    std::vector<haplotype_t>& prefix_data,
    std::vector<marker_t>& divergence_data
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open PBWT index file: " + filename);
    }

    // Read header
    PBWTHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!header.base.is_valid(PBWT_INDEX_MAGIC)) {
        throw std::runtime_error("Invalid PBWT index file format");
    }

    if (header.haplotype_bytes != sizeof(haplotype_t) ||
        header.marker_bytes != sizeof(marker_t)) {
        throw std::runtime_error("PBWT index compiled with different type sizes");
    }

    num_markers = header.num_markers;
    num_haplotypes = header.num_haplotypes;

    size_t total_entries = static_cast<size_t>(num_markers) * num_haplotypes;

    // Read prefix array
    prefix_data.resize(total_entries);
    file.seekg(header.base.data_offset + header.prefix_offset);
    file.read(reinterpret_cast<char*>(prefix_data.data()), header.prefix_size);

    // Read divergence array
    divergence_data.resize(total_entries);
    file.seekg(header.base.data_offset + header.divergence_offset);
    file.read(reinterpret_cast<char*>(divergence_data.data()), header.divergence_size);

    LOG_INFO("Loaded PBWT index: " + filename);
}

void convert_vcf_to_binary_reference(
    const std::string& vcf_path,
    const std::string& binary_path,
    const std::string& region
) {
    LOG_INFO("Converting VCF to binary format: " + vcf_path);

    // Use existing VCF loader
    // This is a placeholder - in production, use ReferencePanel::load_vcf
    auto reference = ReferencePanel::load_vcf(vcf_path, region);

    if (!reference) {
        throw std::runtime_error("Failed to load VCF: " + vcf_path);
    }

    write_reference_panel_binary(
        binary_path,
        reference->markers(),
        reference->samples(),
        reference->haplotypes(),
        reference->num_haplotypes()
    );
}

bool is_binary_reference_valid(
    const std::string& binary_path,
    const std::string& vcf_path
) {
    // Check if binary file exists
    std::ifstream binary_file(binary_path, std::ios::binary);
    if (!binary_file) {
        return false;
    }

    // Check if binary is newer than VCF
    auto binary_time = std::filesystem::last_write_time(binary_path);
    auto vcf_time = std::filesystem::last_write_time(vcf_path);

    if (binary_time < vcf_time) {
        return false;  // VCF is newer, binary needs rebuild
    }

    // Validate binary header
    RefPanelHeader header;
    binary_file.read(reinterpret_cast<char*>(&header), sizeof(header));

    return header.base.is_valid(REF_PANEL_MAGIC);
}

bool is_pbwt_binary_valid(
    const std::string& pbwt_path,
    uint32_t expected_markers,
    uint32_t expected_haplotypes
) {
    std::ifstream file(pbwt_path, std::ios::binary);
    if (!file) {
        return false;
    }

    PBWTHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!header.base.is_valid(PBWT_INDEX_MAGIC)) {
        return false;
    }

    return header.num_markers == expected_markers &&
           header.num_haplotypes == expected_haplotypes;
}

} // namespace io
} // namespace swiftimpute
