#ifndef KUHUL_VRAM_BUDGET_H
#define KUHUL_VRAM_BUDGET_H

#include <cstddef>
#include <string>

namespace kuhul {

class VramBudget {
public:
    explicit VramBudget(std::size_t limit_bytes = 128u * 1024u * 1024u);
    bool reserve(std::size_t bytes, std::string* error = nullptr);
    bool release(std::size_t bytes, std::string* error = nullptr);
    std::size_t used() const;
    std::size_t available() const;
    std::size_t limit() const;

private:
    std::size_t limit_bytes_;
    std::size_t used_bytes_;
};

}

#endif
