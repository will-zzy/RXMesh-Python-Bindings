#include "rxmesh_wrapper.h"
#include "kernels.h"
#include "rxmesh_bridge.cuh"
#include <cuda_runtime.h>
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/attribute.h"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
// 包含 RXMesh 头文件
// #include "rxmesh/patcher.h"
// 轻量包装，暴露 AttributeContainer（rxmesh 本身保持不变）


#define RXMESH_DISPATCH_SET_ATTRIBUTE_TYPES(ELEM_TYPE, DTYPE, MESH, ATTR_BASE, TENSOR) \
    do { \
        if (ELEM_TYPE == "v" && DTYPE == 'f') { \
            set_attribute_impl<rxmesh::VertexAttribute<float>, rxmesh::VertexHandle, float>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "v" && DTYPE == 'i') { \
            set_attribute_impl<rxmesh::VertexAttribute<int32_t>, rxmesh::VertexHandle, int32_t>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "v" && DTYPE == 'b') { \
            set_attribute_impl<rxmesh::VertexAttribute<bool>, rxmesh::VertexHandle, bool>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "v" && DTYPE == 'c') { \
            set_attribute_impl<rxmesh::VertexAttribute<int8_t>, rxmesh::VertexHandle, int8_t>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "e" && DTYPE == 'f') { \
            set_attribute_impl<rxmesh::EdgeAttribute<float>, rxmesh::EdgeHandle, float>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "e" && DTYPE == 'i') { \
            set_attribute_impl<rxmesh::EdgeAttribute<int32_t>, rxmesh::EdgeHandle, int32_t>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "e" && DTYPE == 'b') { \
            set_attribute_impl<rxmesh::EdgeAttribute<bool>, rxmesh::EdgeHandle, bool>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "e" && DTYPE == 'c') { \
            set_attribute_impl<rxmesh::EdgeAttribute<int8_t>, rxmesh::EdgeHandle, int8_t>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "f" && DTYPE == 'f') { \
            set_attribute_impl<rxmesh::FaceAttribute<float>, rxmesh::FaceHandle, float>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "f" && DTYPE == 'i') { \
            set_attribute_impl<rxmesh::FaceAttribute<int32_t>, rxmesh::FaceHandle, int32_t>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "f" && DTYPE == 'b') { \
            set_attribute_impl<rxmesh::FaceAttribute<bool>, rxmesh::FaceHandle, bool>(MESH, ATTR_BASE, TENSOR); \
        } else if (ELEM_TYPE == "f" && DTYPE == 'c') { \
            set_attribute_impl<rxmesh::FaceAttribute<int8_t>, rxmesh::FaceHandle, int8_t>(MESH, ATTR_BASE, TENSOR); \
        } else { \
            throw std::runtime_error("Unsupported element type or data type"); \
        } \
    } while (0)

#define RXMESH_DISPATCH_ADD_ATTRIBUTE(ELEM_TYPE, DTYPE, MESH, ATTRIBUTE_ARRAY, NAME, ELEM_NUM) \
    do { \
        if (ELEM_TYPE == "v" && DTYPE == 'f') { \
            auto ATTR = MESH->add_vertex_attribute<float>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "v" && DTYPE == 'i') { \
            auto ATTR = MESH->add_vertex_attribute<int32_t>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "v" && DTYPE == 'b') { \
            auto ATTR = MESH->add_vertex_attribute<bool>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "v" && DTYPE == 'c') { \
            auto ATTR = MESH->add_vertex_attribute<int8_t>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "e" && DTYPE == 'f') { \
            auto ATTR = MESH->add_edge_attribute<float>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "e" && DTYPE == 'i') { \
            auto ATTR = MESH->add_edge_attribute<int32_t>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "e" && DTYPE == 'b') { \
            auto ATTR = MESH->add_edge_attribute<bool>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "e" && DTYPE == 'c') { \
            auto ATTR = MESH->add_edge_attribute<int8_t>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "f" && DTYPE == 'f') { \
            auto ATTR = MESH->add_face_attribute<float>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "f" && DTYPE == 'i') { \
            auto ATTR = MESH->add_face_attribute<int32_t>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "f" && DTYPE == 'b') { \
            auto ATTR = MESH->add_face_attribute<bool>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else if (ELEM_TYPE == "f" && DTYPE == 'c') { \
            auto ATTR = MESH->add_face_attribute<int8_t>(NAME, ELEM_NUM); \
            ATTRIBUTE_ARRAY[NAME] = ATTR; \
        } else { \
            throw std::runtime_error("Unsupported element type or data type"); \
        } \
    } while (0)

#define RXMESH_DISPATCH_GET_TENSOR_FROM_ATTRIBUTE(ELEM_TYPE, DTYPE, MESH, ATTR_BASE, OUT, DEVICE) \
do { \
    if (ELEM_TYPE == "v" && DTYPE == 'f') { \
        OUT = get_tensor_impl<rxmesh::VertexAttribute<float>, rxmesh::VertexHandle, float>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "v" && DTYPE == 'i') { \
        OUT = get_tensor_impl<rxmesh::VertexAttribute<int32_t>, rxmesh::VertexHandle, int32_t>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "v" && DTYPE == 'b') { \
        OUT = get_tensor_impl<rxmesh::VertexAttribute<bool>, rxmesh::VertexHandle, bool>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "v" && DTYPE == 'c') { \
        OUT = get_tensor_impl<rxmesh::VertexAttribute<int8_t>, rxmesh::VertexHandle, int8_t>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "e" && DTYPE == 'f') { \
        OUT = get_tensor_impl<rxmesh::EdgeAttribute<float>, rxmesh::EdgeHandle, float>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "e" && DTYPE == 'i') { \
        OUT = get_tensor_impl<rxmesh::EdgeAttribute<int32_t>, rxmesh::EdgeHandle, int32_t>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "e" && DTYPE == 'b') { \
        OUT = get_tensor_impl<rxmesh::EdgeAttribute<bool>, rxmesh::EdgeHandle, bool>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "e" && DTYPE == 'c') { \
        OUT = get_tensor_impl<rxmesh::EdgeAttribute<EdgeStatus>, rxmesh::EdgeHandle, EdgeStatus>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "f" && DTYPE == 'f') { \
        OUT = get_tensor_impl<rxmesh::FaceAttribute<float>, rxmesh::FaceHandle, float>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "f" && DTYPE == 'i') { \
        OUT = get_tensor_impl<rxmesh::FaceAttribute<int32_t>, rxmesh::FaceHandle, int32_t>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "f" && DTYPE == 'b') { \
        OUT = get_tensor_impl<rxmesh::FaceAttribute<bool>, rxmesh::FaceHandle, bool>(MESH, ATTR_BASE, DEVICE); \
    } else if (ELEM_TYPE == "f" && DTYPE == 'c') { \
        OUT = get_tensor_impl<rxmesh::FaceAttribute<int8_t>, rxmesh::FaceHandle, int8_t>(MESH, ATTR_BASE, DEVICE); \
    } else { \
        throw std::runtime_error("Unsupported element type or data type"); \
    } \
} while (0)





template <typename T>
constexpr torch::ScalarType torch_scalar_type()
{
    return torch::CppTypeToScalarType<T>::value;
}

// 进程级 RXMesh 初始化标记，避免重复创建名为 "RXMesh" 的全局 logger
namespace {
bool g_rxmesh_initialized = false;
}

// PIMPL 实现，隐藏 RXMesh 细节
struct RXMeshWrapper::Impl {
    std::unique_ptr<RXMeshDynamicBridge> rxmesh;
    void*                                rxmesh_ptr;  // 暂时用 void*，实际使用时替换为具体类型

    Impl() : rxmesh_ptr(nullptr) {}
};

namespace {

__global__ void count_owned_active_per_patch_kernel(
    const uint32_t num_patches,
    const rxmesh::PatchInfo* patches_info,
    uint32_t* counts_v,
    uint32_t* counts_e,
    uint32_t* counts_f)
{
    const uint32_t p = blockIdx.x;
    if (p >= num_patches || threadIdx.x != 0) {
        return;
    }

    const rxmesh::PatchInfo& pi = patches_info[p];
    counts_v[p] = static_cast<uint32_t>(pi.get_num_owned<rxmesh::VertexHandle>());
    counts_e[p] = static_cast<uint32_t>(pi.get_num_owned<rxmesh::EdgeHandle>());
    counts_f[p] = static_cast<uint32_t>(pi.get_num_owned<rxmesh::FaceHandle>());
}

bool rebuild_prefix_on_device(RXMeshDynamicBridge* rx)
{
    const uint32_t num_patches = rx->get_num_patches(true);
    if (num_patches == 0) {
        return true;
    }

    const size_t n = static_cast<size_t>(num_patches) + 1;
    const size_t bytes = n * sizeof(uint32_t);

    uint32_t *counts_v = nullptr, *counts_e = nullptr, *counts_f = nullptr;
    if (cudaMalloc(&counts_v, bytes) != cudaSuccess ||
        cudaMalloc(&counts_e, bytes) != cudaSuccess ||
        cudaMalloc(&counts_f, bytes) != cudaSuccess) {
        if (counts_v) cudaFree(counts_v);
        if (counts_e) cudaFree(counts_e);
        if (counts_f) cudaFree(counts_f);
        return false;
    }

    cudaError_t err = cudaSuccess;
    err = cudaMemset(counts_v, 0, bytes);
    if (err == cudaSuccess) err = cudaMemset(counts_e, 0, bytes);
    if (err == cudaSuccess) err = cudaMemset(counts_f, 0, bytes);

    if (err == cudaSuccess) {
        count_owned_active_per_patch_kernel<<<num_patches, 1>>>(
            num_patches, rx->d_patches_info(), counts_v, counts_e, counts_f);
        err = cudaGetLastError();
    }

    if (err == cudaSuccess) {
        thrust::exclusive_scan(
            thrust::device, counts_v, counts_v + n, rx->d_vertex_prefix());
        thrust::exclusive_scan(
            thrust::device, counts_e, counts_e + n, rx->d_edge_prefix());
        thrust::exclusive_scan(
            thrust::device, counts_f, counts_f + n, rx->d_face_prefix());
        err = cudaDeviceSynchronize();
    }

    cudaFree(counts_v);
    cudaFree(counts_e);
    cudaFree(counts_f);

    return err == cudaSuccess;
}

void rebuild_prefix_or_throw(RXMeshWrapper* wrapper)
{
    RXMeshDynamicBridge* rx = wrapper->get_bridge();
    rebuild_prefix_on_device(rx);
    // if (!rebuild_prefix_on_device(rx)) {
    //     printf("Warning: GPU prefix rebuild failed, falling back to host update\n");
    //     rx->update_host();
    // }
    rx->update_host();
}   

} // namespace

// void RXMeshWrapper::write_vertex_positions_to_rxmesh(torch::Tensor value) { // 
//     // TORCH_CHECK(m_impl->rxmesh, "RXMesh has not been initialized");
//     auto attr = m_impl->rxmesh->get_input_vertex_coordinates();
//     if (!attr->is_device_allocated()) {
//         attr->move(rxmesh::HOST, rxmesh::DEVICE);
//     }
//     const uint32_t num_attr     = attr->get_num_attributes();
//     const uint32_t num_elements = attr->size();
//     TORCH_CHECK(value.size(0) == num_elements, "Mismatch in number of elements");
//     TORCH_CHECK(value.size(1) == num_attr, "Mismatch in number of attributes");
//     float* src_ptr = value.data_ptr<float>();
//     rxmesh::Context ctx = m_impl->rxmesh->get_context();
//     auto attr_val = *attr;  // capture by value so device code avoids shared_ptr
//     // note that attr is a shared_ptr, we cannot dereference it directly in device code, we need to capture the raw pointer or the underlying data structure that can be accessed in device code    
//     m_impl->rxmesh->for_each_vertex(rxmesh::DEVICE,
//                           [ctx, src_ptr, attr_val, num_attr]
//                           __device__(const rxmesh::VertexHandle vh) mutable {
//                               const uint32_t lid = ctx.linear_id(vh);
//                               for (uint32_t a = 0; a < num_attr; ++a) {
//                                   attr_val(vh, a) = src_ptr[lid * num_attr + a];
//                               }
//                           });
// }




torch::Tensor RXMeshWrapper::copy_vertex_to_tensor() {
    int device = m_device;
    m_impl->rxmesh->get_num_patches(true);
    uint32_t num = m_impl->rxmesh->get_num_vertices(true);
    // std::cout << "num vertices: " << num << std::endl;
    auto attr = m_impl->rxmesh->get_input_vertex_coordinates(); 
    const uint32_t num_attr     = attr->get_num_attributes();
    const uint32_t num_elements = attr->size();

    auto options = torch::TensorOptions()
                        .dtype(torch_scalar_type<float>())
                        .device(torch::kCUDA, device);  // global memory下的
    auto out = torch::empty({static_cast<long>(num_elements),
                             static_cast<long>(num_attr)},
                            options);             
    // std::cout << "tensor size: " << out.sizes() << std::endl;       
    float* out_ptr = out.data_ptr<float>();
    rxmesh::Context ctx = m_impl->rxmesh->get_context();
    auto attr_val = *attr;  // capture by value so device code can access m_d_attr
    TORCH_CHECK(attr_val.is_device_allocated(), "attribute not allocated on device");

    m_impl->rxmesh->for_each_vertex(rxmesh::DEVICE,
            [ctx, out_ptr, attr_val, num_attr]
            __device__(const rxmesh::VertexHandle vh) mutable {
                const uint32_t lid = ctx.linear_id(vh);
                for (uint32_t a = 0; a < num_attr; ++a) {
                    out_ptr[lid * num_attr + a] =
                        attr_val(vh, a);
                } 
            });

    return out;
}


// face indices are not attributes, we cannot use for_each, need a separate kernel using active_mask and owned_mask 
// to filter valid faces, and use linear_id to get the corresponding vertex indices in the output tensor
__global__ void copy_face_indices_to_tensor_kernel( 
    const uint32_t num_patches, 
    const rxmesh::Context ctx,
    const rxmesh::PatchInfo* patches_info,
    int* out_ptr
){
    const uint32_t p_id = blockIdx.x;
    if (p_id >= num_patches) return;
    const rxmesh::PatchInfo& patch_info = patches_info[p_id];
    const uint16_t num_faces = patch_info.num_faces[0];
    for (uint16_t f = threadIdx.x; f < num_faces; f += blockDim.x) {
        if (!rxmesh::detail::is_deleted(f, patch_info.active_mask_f)){
            if (!rxmesh::detail::is_owned(f, patch_info.owned_mask_f)) {
                continue;
            }
            rxmesh::FaceHandle f_handle(patch_info.patch_id, f);
            const uint32_t f_lid = ctx.linear_id(f_handle); 
            // linear_id need m_d_*_prefix result, so we need to make sure the prefix is up to date before calling this kernel
            for (uint32_t e = 0; e < 3; ++e) {
                uint16_t edge = patch_info.fe[3 * f + e].id;
                rxmesh::flag_t dir(0);
                ctx.unpack_edge_dir(edge, edge, dir);
                uint16_t e_id = (2 * edge) + dir;
                uint16_t v = patch_info.ev[e_id].id;
                rxmesh::VertexHandle vh(patch_info.patch_id, v);
                uint32_t v_lid = ctx.linear_id(vh);
                out_ptr[3 * f_lid + e] = v_lid;
            }
        }
    }
}

torch::Tensor RXMeshWrapper::copy_face_indices_to_tensor() { 
    // TORCH_CHECK(m_impl->rxmesh, "RXMesh has not been initialized");
    
    const uint32_t num_patches  = m_impl->rxmesh->get_num_patches(true);
    const uint32_t num_elements = m_impl->rxmesh->get_num_faces(true);

    auto options = torch::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(torch::kCUDA, m_device);
    if (num_elements == 0 || num_patches == 0) {
        return torch::empty({0, 3}, options);
    }

    cudaSetDevice(m_device);

    auto out = torch::full({static_cast<long>(num_elements),
                            3}, // only support triangle
                           -1,
                           options);
    int* out_ptr = out.data_ptr<int>();
    rxmesh::Context ctx = m_impl->rxmesh->get_context();

    const int threads = 256;

    copy_face_indices_to_tensor_kernel<<<num_patches, threads, 0>>>(
        num_patches, ctx, m_impl->rxmesh->d_patches_info(), out_ptr
    );
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    return out;
}



RXMeshWrapper::RXMeshWrapper(
    const std::string& mesh_path,
    int device,
    const std::string  patcher_file,
    const uint32_t     patch_size,
    const float        capacity_factor,
    const float        patch_alloc_factor,
    const float        lp_hashtable_load_factor)
    : m_impl(std::make_unique<Impl>())
    , m_device(device)
    , m_mesh_path(mesh_path)
    , m_patch_size(patch_size)
    , m_capacity_factor(capacity_factor)
    , m_patch_alloc_factor(patch_alloc_factor)
    , m_lp_hashtable_load_factor(lp_hashtable_load_factor)
{
    if (!g_rxmesh_initialized) {
        rxmesh::rx_init(m_device);
        g_rxmesh_initialized = true;
    }
    m_impl->rxmesh = std::make_unique<RXMeshDynamicBridge>(
        mesh_path, patcher_file, patch_size, capacity_factor, patch_alloc_factor
    );

    if(!m_impl->rxmesh->validate()){
        RXMESH_ERROR("RXMesh validation failed for mesh" );
    }
    
    rebuild_prefix_or_throw(this);
}

RXMeshWrapper::~RXMeshWrapper() = default;


// need to call after `rebuild_prefix_or_throw` if false?
uint32_t RXMeshWrapper::get_num_patches() const {
    return m_impl->rxmesh->get_num_patches(true);
}

uint32_t RXMeshWrapper::get_num_vertices() const {
    return m_impl->rxmesh->get_num_vertices(true);
}

uint32_t RXMeshWrapper::get_num_faces() const {
    return m_impl->rxmesh->get_num_faces(true);
}

void RXMeshWrapper::synchronize() const {
    cudaSetDevice(m_device);
    cudaDeviceSynchronize();
}
RXMeshDynamicBridge* RXMeshWrapper::get_bridge() {
    return m_impl->rxmesh.get();
}
const RXMeshDynamicBridge* RXMeshWrapper::get_bridge() const {
    return m_impl->rxmesh.get();
}
/*----------------------------write attribute to tensor----------------------------*/
template <typename AttrT, typename HandleT, typename ValueT>
torch::Tensor get_tensor_impl(RXMeshDynamicBridge* mesh,
                              std::shared_ptr<rxmesh::AttributeBase> attr_base,
                              int m_device
) {
    auto attr = std::dynamic_pointer_cast<AttrT>(attr_base);
    TORCH_CHECK(attr->is_device_allocated(), "attribute not allocated on device");
    if (!attr) {
        throw std::runtime_error("Attribute type mismatch");
    }

    const uint32_t num_attr = attr->get_num_attributes();
    const uint32_t num_elements = attr->size();
    auto options = torch::TensorOptions()
                        .dtype(torch_scalar_type<ValueT>())
                        .device(torch::kCUDA, m_device);
    auto value = torch::empty({static_cast<long>(num_elements),
                              static_cast<long>(num_attr)},
                             options);

    ValueT* value_ptr = value.template data_ptr<ValueT>();
    rxmesh::Context ctx = mesh->get_context();
    auto attr_val = *attr; // capture by value so device code can access m_d_attr

    if constexpr (std::is_same_v<HandleT, rxmesh::VertexHandle>) {
        mesh->for_each_vertex(rxmesh::DEVICE,
            [ctx, value_ptr, attr_val, num_attr] __device__ (rxmesh::VertexHandle vh) mutable {
                const uint32_t lid = ctx.linear_id(vh);
                for (uint32_t a = 0; a < num_attr; ++a) {
                    value_ptr[lid * num_attr + a] = attr_val(vh, a);
                }
            });
    } else if constexpr (std::is_same_v<HandleT, rxmesh::EdgeHandle>) {
        mesh->for_each_edge(rxmesh::DEVICE,
            [ctx, value_ptr, attr_val, num_attr] __device__ (rxmesh::EdgeHandle eh) mutable {
                const uint32_t lid = ctx.linear_id(eh);
                for (uint32_t a = 0; a < num_attr; ++a) {
                    value_ptr[lid * num_attr + a] = attr_val(eh, a);
                }
            });
    } else if constexpr (std::is_same_v<HandleT, rxmesh::FaceHandle>) {
        mesh->for_each_face(rxmesh::DEVICE,
            [ctx, value_ptr, attr_val, num_attr] __device__ (rxmesh::FaceHandle fh) mutable {
                const uint32_t lid = ctx.linear_id(fh);
                for (uint32_t a = 0; a < num_attr; ++a) {
                    value_ptr[lid * num_attr + a] = attr_val(fh, a);
                }
            });
    }
    return value;

}

// We use name format "[element type]:[dtype][num_attr]:[attribute name]" to identify different attributes, for example:
// "v:f3:gradient" means vertex attribute named "gradient" with float3 type
// all added attributes will be managed in m_attributes map, and we can get the corresponding attribute base pointer through the name, 
// then we can dispatch to the correct get_tensor_impl to create tensor from attribute
torch::Tensor RXMeshWrapper::get_attribute(std::string name) {
    // auto out = get_tensor_from_attribute(this, name, m_device);

    static const std::regex re("(v|e|f):([fibc])([1-4]):(.+)");
    std::smatch match;
    if (!std::regex_match(name, match, re)) {
        throw std::invalid_argument("Invalid attribute name format: " + name);
    }
    std::string type = match[1]; // v/e/f
    char dtype = match[2].str()[0]; // f/i/b/c
    int num_attr = std::stoi(match[3]); // 1-4

    bool is_exist = m_attributes.find(name) != m_attributes.end();
    if (!is_exist) {
        throw std::runtime_error("Attribute not found: " + name);
    }

    torch::Tensor out;
    auto base = m_attributes.at(name);
    RXMESH_DISPATCH_GET_TENSOR_FROM_ATTRIBUTE(type, dtype, m_impl->rxmesh.get(), base, out, m_device);
    return out;
}

/*----------------------------write attribute to tensor----------------------------*/




/*----------------------------write tensor to attribute----------------------------*/
template <typename AttrT, typename HandleT, typename ValueT>
void set_attribute_impl(RXMeshDynamicBridge* mesh,
                        std::shared_ptr<rxmesh::AttributeBase> attr_base,
                        const torch::Tensor& value) 
{
    if (!attr_base) {
        throw std::runtime_error("Attribute not found");
    }

    auto attr = std::dynamic_pointer_cast<AttrT>(attr_base);
    TORCH_CHECK(attr, "Attribute type mismatch for this name");
    // check tensor properties
    TORCH_CHECK(value.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(value.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(value.scalar_type() == torch_scalar_type<ValueT>(),
                "Input tensor dtype mismatch: expected ", torch_scalar_type<ValueT>(),
                " got ", value.scalar_type());
    const uint32_t num_attr = attr->get_num_attributes();
    const uint32_t num_elements = attr->size();
    TORCH_CHECK(num_attr == value.size(1), "Mismatch in number of attributes");
    TORCH_CHECK(num_elements == value.size(0), "Mismatch in number of elements");
    TORCH_CHECK(attr->is_device_allocated(), "attribute not allocated on device");
    
    ValueT* src_ptr = (ValueT*)value.data_ptr<ValueT>();
    rxmesh::Context ctx = mesh->get_context();
    auto attr_val = *attr; 
    
    if constexpr (std::is_same_v<HandleT, rxmesh::VertexHandle>) {
        mesh->for_each_vertex(rxmesh::DEVICE,
            [ctx, src_ptr, attr_val, num_attr] __device__ (rxmesh::VertexHandle vh) mutable {
                const uint32_t lid = ctx.linear_id(vh);
                for (uint32_t a = 0; a < num_attr; ++a) {
                    attr_val(vh, a) = src_ptr[lid * num_attr + a]; // note difference between SoA and AoS
                }
            });
    } else if constexpr (std::is_same_v<HandleT, rxmesh::EdgeHandle>) {
        mesh->for_each_edge(rxmesh::DEVICE,
            [ctx, src_ptr, attr_val, num_attr] __device__ (rxmesh::EdgeHandle eh) mutable {
                const uint32_t lid = ctx.linear_id(eh);
                for (uint32_t a = 0; a < num_attr; ++a) {
                    attr_val(eh, a) = src_ptr[lid * num_attr + a];
                }
            });
    } else if constexpr (std::is_same_v<HandleT, rxmesh::FaceHandle>) {
        mesh->for_each_face(rxmesh::DEVICE,
            [ctx, src_ptr, attr_val, num_attr] __device__ (rxmesh::FaceHandle fh) mutable {
                const uint32_t lid = ctx.linear_id(fh);
                for (uint32_t a = 0; a < num_attr; ++a) {
                    attr_val(fh, a) = src_ptr[lid * num_attr + a];
                }
            });
    }
}

// input tensor and name, write to rxmesh attribute, if attribute not exist, create it first, if exist, check shape and dtype, then write
void RXMeshWrapper::set_attribute(std::string name, torch::Tensor value, bool force_add) {

    TORCH_CHECK(m_device == value.device().index(), "Input tensor device mismatch: expected CUDA:", m_device, " got ", value.device());
    // value一定是[N, C]的
    static const std::regex re("(v|e|f):([fib])([1-4]):(.+)");
    std::smatch match;
    if (!std::regex_match(name, match, re)) {
        throw std::invalid_argument("Invalid attribute name format: " + name);
    }
    std::string type = match[1]; // v/e/f
    char dtype = match[2].str()[0]; // f/i
    int num_attr = std::stoi(match[3]); // 1-4
    // std::string attr_name = match[4]; // gradient/color/temperature etc

    if (force_add) {
        remove_attribute(name, false); 
        // force remove as maybe the topology changed and the existing attribute is not compatible with the new tensor
        RXMESH_DISPATCH_ADD_ATTRIBUTE(type, dtype, m_impl->rxmesh.get(), m_attributes, name, num_attr);
    }
    else {
        auto it = m_attributes.find(name);
        if (it == m_attributes.end()) {
            RXMESH_DISPATCH_ADD_ATTRIBUTE(type, dtype, m_impl->rxmesh.get(), m_attributes, name, num_attr);
        } 
    }
    auto base_attr = m_attributes[name]; 
    
    RXMESH_DISPATCH_SET_ATTRIBUTE_TYPES(type, dtype, m_impl->rxmesh.get(), base_attr, value); 
}

/*----------------------------write tensor to attribute----------------------------*/


/*--------------------------------remove attribute----------------------------------*/
void RXMeshWrapper::remove_attribute(std::string name, bool log) {
    auto it = m_attributes.find(name);
    if (it == m_attributes.end()) {
        if (log)
            std::cerr << "Warning: Attempting to remove non-existent attribute: " << name << std::endl;
        return;
    }
    m_impl->rxmesh->remove_attribute(name);
    m_attributes.erase(it);
}
/*--------------------------------remove attribute----------------------------------*/


/*--------------------------------add attribute----------------------------------*/

void RXMeshWrapper::add_attribute(std::string name){
static const std::regex re("(v|e|f):([fib])([1-4]):(.+)");
    std::smatch match;
    if (!std::regex_match(name, match, re)) {
        throw std::invalid_argument("Invalid attribute name format: " + name);
    }
    std::string type = match[1]; // v/e/f
    char dtype = match[2].str()[0]; // f/i
    int num_attr = std::stoi(match[3]); // 1-4
    this->remove_attribute(name, false);
    
    RXMESH_DISPATCH_ADD_ATTRIBUTE(type, dtype, m_impl->rxmesh.get(), m_attributes, name, num_attr);
}
/*--------------------------------add attribute----------------------------------*/


/*-----------------------------kernel: compute_smmoth_gradients---------------------*/
void RXMeshWrapper::compute_smooth_gradients() {
    SmoothGrad(this, false);
}
/*-----------------------------kernel: compute_smmoth_gradients---------------------*/

/*----------------------------apply_gradients--------------------------------------------------*/
void RXMeshWrapper::compute_gradients(
    const float ratioRigidityElasticity,
    const float weightRegularity,
    const float gstep
) {
    // 首先确保python计算过了smmoth_gradient1/2以及photo_grad
    auto it_smooth1 = m_attributes.find("v:f3:smooth_grad1");
    auto it_smooth2 = m_attributes.find("v:f3:smooth_grad2");
    auto it_photo_grad = m_attributes.find("v:f3:photo_grad");
    if (it_smooth1 == m_attributes.end() || it_smooth2 == m_attributes.end() || it_photo_grad == m_attributes.end()) {
        throw std::runtime_error("Required attributes for applying gradients not found");
    }

    ComputeGrad(this, ratioRigidityElasticity, weightRegularity, gstep);
    
    // remove_attribute("v:f3:smooth_grad1");
    // remove_attribute("v:f3:smooth_grad2");
    remove_attribute("v:f3:photo_grad");
    remove_attribute("v:i1:grad_norm");
    
}

// void RXMeshWrapper::subdivide(){
//     // need to know "f:b1:need_subdivide" attributes which is computed in python
//     auto need_subdivide = m_attributes.find("f:b1:need_subdivide");
//     if (need_subdivide == m_attributes.end()) {
//         throw std::runtime_error("Attribute not found: f:b1:need_subdivide");
//     }
//     Subdivide(this);
// }

void RXMeshWrapper::split_edge()
{
    auto need_subdivide = m_attributes.find("f:b1:need_subdivide");
    if (need_subdivide == m_attributes.end()) {
        throw std::runtime_error("Attribute not found: f:b1:need_subdivide");
    }

    SplitEdges(this);
    rebuild_prefix_or_throw(this);
}

void RXMeshWrapper::flip_edge()
{

    FlipEdges(this);
    rebuild_prefix_or_throw(this);
}

void RXMeshWrapper::compute_face_normal() {
    ComputeFaceNormals(this);
}

void RXMeshWrapper::update_vertex_positions(){
    auto has_gradients = m_attributes.find("v:f3:gradient");
    if (has_gradients == m_attributes.end()) {
        throw std::runtime_error("Attribute not found: v:f3:gradient");
    }
    UpdateVertexPositions(this);
}