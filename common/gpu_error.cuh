#pragma once

#define errchk(res) { gpu_assert((res), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (cudaSuccess != code) {
        fprintf(stderr, "gpu_assert: %s @ %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}
