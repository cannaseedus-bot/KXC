#include <stdint.h>

static const uint8_t SOH = 1;
static const uint8_t STX = 2;
static const uint8_t ETX = 3;
static const uint8_t EOT = 4;
static const uint8_t SO = 14;
static const uint8_t SI = 15;
static const uint8_t RS = 30;

int verify(const uint8_t *data, uint32_t len) {
  if (!data || len < 4) {
    return 0;
  }

  if (data[0] != SOH || data[1] != STX) {
    return 0;
  }

  if (data[len - 2] != ETX || data[len - 1] != EOT) {
    return 0;
  }

  uint32_t i = 2;
  while (i < len - 2) {
    if (i + 2 >= len - 2) {
      return 0;
    }

    if (data[i] != SO || data[i + 1] != RS || data[i + 2] != SI) {
      return 0;
    }

    i += 3;
  }

  return i == len - 2;
}
