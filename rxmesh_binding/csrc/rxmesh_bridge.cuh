
#pragma once
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/attribute.h"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.h"
#include "rxmesh/util/timer.h"

class RXMeshDynamicBridge : public rxmesh::RXMeshDynamic {
   public:
    using rxmesh::RXMeshDynamic::RXMeshDynamic; 

    rxmesh::AttributeContainer* attr_container() { return m_attr_container.get(); }
    __host__ __device__ const rxmesh::PatchInfo* d_patches_info() const {
        return this->m_d_patches_info;
    }

    __host__ __device__ uint32_t* d_vertex_prefix() {
        return this->m_d_vertex_prefix;
    }

    __host__ __device__ uint32_t* d_edge_prefix() {
        return this->m_d_edge_prefix;
    }

    __host__ __device__ uint32_t* d_face_prefix() {
        return this->m_d_face_prefix;
    }

};