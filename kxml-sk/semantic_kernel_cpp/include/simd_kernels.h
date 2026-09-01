#pragma once

#include <cstddef>
#include <cstdint>

void simd_add_i64(const int64_t* a, const int64_t* b, int64_t* out, size_t count);

#include "ir.h"
void executeAddLane(LaneBatch &batch, Buffers &buf);
