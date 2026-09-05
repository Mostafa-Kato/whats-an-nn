//
// Created by mkato on 8/12/26.
//

#ifndef WHATS_AN_NN_CUDAERRORCHECK_CUH
#define WHATS_AN_NN_CUDAERRORCHECK_CUH

#include <cstdio>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#endif //WHATS_AN_NN_CUDAERRORCHECK_CUH
