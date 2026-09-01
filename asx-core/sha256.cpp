#include "sha256.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace asx {

namespace {

constexpr std::array<std::uint32_t, 64> kTable = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

inline std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
    return (value >> bits) | (value << (32U - bits));
}

std::string sha256_bytes(const std::vector<unsigned char>& input) {
    std::vector<unsigned char> data = input;
    const std::uint64_t bit_length = static_cast<std::uint64_t>(data.size()) * 8ULL;
    data.push_back(0x80U);
    while ((data.size() % 64U) != 56U) data.push_back(0x00U);
    for (int i = 7; i >= 0; --i) {
        data.push_back(static_cast<unsigned char>((bit_length >> (i * 8)) & 0xffU));
    }

    std::uint32_t h0 = 0x6a09e667U;
    std::uint32_t h1 = 0xbb67ae85U;
    std::uint32_t h2 = 0x3c6ef372U;
    std::uint32_t h3 = 0xa54ff53aU;
    std::uint32_t h4 = 0x510e527fU;
    std::uint32_t h5 = 0x9b05688cU;
    std::uint32_t h6 = 0x1f83d9abU;
    std::uint32_t h7 = 0x5be0cd19U;

    std::array<std::uint32_t, 64> w{};
    for (std::size_t chunk = 0; chunk < data.size(); chunk += 64U) {
        for (int i = 0; i < 16; ++i) {
            const std::size_t j = chunk + static_cast<std::size_t>(i) * 4U;
            w[i] = (static_cast<std::uint32_t>(data[j]) << 24U) |
                   (static_cast<std::uint32_t>(data[j + 1]) << 16U) |
                   (static_cast<std::uint32_t>(data[j + 2]) << 8U) |
                   (static_cast<std::uint32_t>(data[j + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7U) ^ rotr(w[i - 15], 18U) ^ (w[i - 15] >> 3U);
            const std::uint32_t s1 = rotr(w[i - 2], 17U) ^ rotr(w[i - 2], 19U) ^ (w[i - 2] >> 10U);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = h0;
        std::uint32_t b = h1;
        std::uint32_t c = h2;
        std::uint32_t d = h3;
        std::uint32_t e = h4;
        std::uint32_t f = h5;
        std::uint32_t g = h6;
        std::uint32_t h = h7;

        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + ch + kTable[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }

    std::ostringstream out;
    out << std::hex << std::setfill('0')
        << std::setw(8) << h0 << std::setw(8) << h1 << std::setw(8) << h2 << std::setw(8) << h3
        << std::setw(8) << h4 << std::setw(8) << h5 << std::setw(8) << h6 << std::setw(8) << h7;
    return out.str();
}

}  // namespace

std::string sha256_hex(const std::string& input) {
    return sha256_bytes(std::vector<unsigned char>(input.begin(), input.end()));
}

std::string sha256_file_hex(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read file for sha256: " + path);
    std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return sha256_bytes(buffer);
}

}  // namespace asx
