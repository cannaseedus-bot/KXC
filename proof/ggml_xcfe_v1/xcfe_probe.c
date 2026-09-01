// xcfe_probe.c — the discriminating check: does the KHANARY ggml-xcfe backend REGISTER?
// Walks the ggml backend registry and asserts an entry named "XCFE" (exit 0 = present).
#include "ggml-backend.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    size_t n = ggml_backend_reg_count();
    int found = 0;
    for (size_t i = 0; i < n; i++) {
        ggml_backend_reg_t r = ggml_backend_reg_get(i);
        const char * name = ggml_backend_reg_name(r);
        printf("backend[%zu] = %s\n", i, name);
        if (strcmp(name, "XCFE") == 0) found = 1;
    }
    printf("XCFE registered: %s\n", found ? "YES" : "NO");
    return found ? 0 : 2;
}
