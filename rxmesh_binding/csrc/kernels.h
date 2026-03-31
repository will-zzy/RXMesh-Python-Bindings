#pragma once
#include <torch/extension.h>
#include <cooperative_groups.h>
#include "rxmesh_bridge.cuh"

class RXMeshWrapper;
using EdgeStatus = uint8_t;
enum : EdgeStatus
{
    UNSEEN = 0,
    SKIP = 1,
    TO_SPLIT = 2,
    ADDED = 3,
    DELETED = 4,
    FLIP = 5,
    TO_FLIP = 6,
    UPDATE = 7
};

using FaceStatus = uint8_t;
enum : FaceStatus
{
    FUNSEEN = 0,
    FSKIP = 1,
    FTO_SPLIT = 2
};

void SmoothGrad(RXMeshWrapper* wrapper, bool debug);
void ComputeGrad(RXMeshWrapper* wrapper, const float ratioRigidityElasticity, const float weightRegularity, const float gstep);
void SplitEdges(RXMeshWrapper* wrapper);
void FlipEdges(RXMeshWrapper* wrapper);
// void Subdivide(RXMeshWrapper* wrapper);
void ComputeFaceNormals(RXMeshWrapper* wrapper);
void UpdateVertexPositions(RXMeshWrapper* wrapper);

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}