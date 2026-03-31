#pragma once

#include <torch/extension.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <regex>
#include <stdexcept>
#include <cassert>
#include <unordered_map>
namespace rxmesh{
    class AttributeBase;
}
class RXMeshDynamicBridge;

class RXMeshWrapper {
public:

    RXMeshWrapper(
        const std::string& mesh_path,
        int device = 0,
        const std::string  patcher_file             = "",
        const uint32_t     patch_size               = 256,
        const float        capacity_factor          = 3.5,
        const float        patch_alloc_factor       = 5.0,
        const float        lp_hashtable_load_factor = 0.5
    );
    
    ~RXMeshWrapper();
    
    RXMeshWrapper(const RXMeshWrapper&) = delete;
    RXMeshWrapper& operator=(const RXMeshWrapper&) = delete;
    RXMeshWrapper(RXMeshWrapper&&) = default;
    RXMeshWrapper& operator=(RXMeshWrapper&&) = default;
    


    uint32_t get_num_patches() const;
    uint32_t get_num_vertices() const;
    uint32_t get_num_faces() const;
    void synchronize() const;

    


    // attributes
    torch::Tensor get_attribute(std::string name);
    void set_attribute(std::string name, torch::Tensor value, bool force_add = false); 
    void remove_attribute(std::string name, bool log = false);
    void add_attribute(std::string name);

    // vertices position and face indices
    torch::Tensor copy_vertex_to_tensor();
    torch::Tensor copy_face_indices_to_tensor();
    // void write_vertex_positions_to_rxmesh(torch::Tensor value);

    torch::Tensor run_custom_kernel(
        const std::string& kernel_name,
        const std::vector<torch::Tensor>& inputs,
        const std::vector<int64_t>& params
    );
    std::string m_mesh_path;
    int m_device;
    uint32_t m_patch_size;
    float m_capacity_factor;          
    float m_patch_alloc_factor;
    float m_lp_hashtable_load_factor;




    RXMeshDynamicBridge* get_bridge();
    const RXMeshDynamicBridge* get_bridge() const;
    // some custom kernel
    void compute_smooth_gradients();
    void compute_gradients(const float ratioRigidityElasticity, const float weightRegularity, const float gstep);
    void split_edge();
    void flip_edge();
    // void subdivide();
    void compute_face_normal();
    void update_vertex_positions(); // need v:f3:gradient attribute to be set
    // void decimation();
    // void clean();
    // void manifold();
    

    struct Impl;  // PIMPL mode
    std::unique_ptr<Impl> m_impl;

    torch::Tensor m_vertex_positions;    // (N, 3) float32
    torch::Tensor m_face_indices;        // (F, 3) int32
    
    std::unordered_map<std::string, std::shared_ptr<rxmesh::AttributeBase>> m_attributes; // mapping from attribute names to attribute
};

