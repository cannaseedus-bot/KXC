#include "vram_budget.h"
#include <limits>

namespace kuhul {

VramBudget::VramBudget(std::size_t limit_bytes)
    : limit_bytes_(limit_bytes), used_bytes_(0) {}

bool VramBudget::reserve(std::size_t bytes, std::string* error) {
    if (bytes > limit_bytes_ || used_bytes_ > limit_bytes_ - bytes) {
        if (error) *error = "OpenCL allocation rejected by 128 MB admission budget";
        return false;
    }
    used_bytes_ += bytes;
    return true;
}

bool VramBudget::release(std::size_t bytes, std::string* error) {
    if (bytes > used_bytes_) {
        if (error) *error = "OpenCL release exceeds tracked allocation";
        return false;
    }
    used_bytes_ -= bytes;
    return true;
}

std::size_t VramBudget::used() const { return used_bytes_; }
std::size_t VramBudget::available() const { return limit_bytes_ - used_bytes_; }
std::size_t VramBudget::limit() const { return limit_bytes_; }

}
