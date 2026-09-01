#include "matrix5x4.h"

int main() {
    Matrix5x4 a{}, b{};
    Matrix5x4 c = matrix5x4_add(a, b);
    return matrix5x4_at(c, 4, 3) == 0.0f ? 0 : 1;
}
