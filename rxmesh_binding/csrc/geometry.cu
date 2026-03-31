#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include "kernels.h"
#include "rxmesh_bridge.cuh"
#include "rxmesh_wrapper.h"
// kernel:输入vertex position，计算每个vertex的smoothgrad1


template <uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads) compute_smooth_grad1_kernel(
    rxmesh::Context context,
    const rxmesh::VertexAttribute<float> vertex_position,
    rxmesh::VertexAttribute<float> smooth_grad,
    rxmesh::VertexAttribute<float> smooth_score,
    rxmesh::VertexAttribute<bool> v_boundary

) {
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    // 计算one ring smooth grad1
    auto smooth = [&](VertexHandle v_id, VertexIterator& iter) {
        if (iter.size() == 0){
            smooth_score(v_id) = 0.0f;
            smooth_grad(v_id, 0) = 0.0f;
            smooth_grad(v_id, 1) = 0.0f;
            smooth_grad(v_id, 2) = 0.0f;
            return;
        }
        
        if (v_boundary(v_id) || iter.size() == 0){
            smooth_score(v_id) = 0.0f;
            smooth_grad(v_id, 0) = 0.0f;
            smooth_grad(v_id, 1) = 0.0f;
            smooth_grad(v_id, 2) = 0.0f;
            return;
        }

        const vec3<float> v_pos = vertex_position.to_glm<3>(v_id); //

        vec3<float> grad(0.0, 0.0, 0.0);
        for (uint32_t i = 0; i < iter.size(); i++) {
            const VertexHandle r_id = iter[i];
            const vec3<float> r_pos = vertex_position.to_glm<3>(r_id);
            grad += r_pos;
        }
        grad = grad / (float)iter.size() - v_pos;
        const float regularityScore = glm::length(grad);
        smooth_score(v_id) = regularityScore;
        smooth_grad(v_id, 0) = grad.x;
        smooth_grad(v_id, 1) = grad.y;
        smooth_grad(v_id, 2) = grad.z;
    };

    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::VV>(block, shrd_alloc, smooth, false);
}


template <uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    compute_valence(rxmesh::Context                  context,
                    rxmesh::VertexAttribute<uint8_t> v_valence)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    Query<blockThreads> query(context);
    query.compute_vertex_valence(block, shrd_alloc);
    block.sync();

    for_each_vertex(query.get_patch_info(), [&](VertexHandle vh) {
        v_valence(vh) = query.vertex_valence(vh);
    });
}

template <uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads) compute_smooth_grad2_kernel(
    rxmesh::Context context,
    const rxmesh::VertexAttribute<float> smooth_grad1,
    rxmesh::VertexAttribute<float> smooth_grad2,
    rxmesh::VertexAttribute<bool> v_boundary,
    rxmesh::VertexAttribute<uint8_t> v_valence
) {
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    // 计算one ring smooth grad1
    
    auto smooth = [&](VertexHandle v_id, VertexIterator& iter) {
        
        if (iter.size() == 0){
            smooth_grad2(v_id, 0) = 0.0f;
            smooth_grad2(v_id, 1) = 0.0f;
            smooth_grad2(v_id, 2) = 0.0f;
            return;
        }
        
        if (v_boundary(v_id) || iter.size() == 0){ 
            smooth_grad2(v_id, 0) = 0.0f;
            smooth_grad2(v_id, 1) = 0.0f;
            smooth_grad2(v_id, 2) = 0.0f;
            return;
        }
        float w = 0.0f;
        const vec3<float> grad_1 = smooth_grad1.to_glm<3>(v_id); //

        vec3<float> grad(0.0, 0.0, 0.0);
        for (uint32_t i = 0; i < iter.size(); i++) {
            const VertexHandle r_id = iter[i];
            const vec3<float> r_grad_1 = smooth_grad1.to_glm<3>(r_id);
            grad += r_grad_1;
            // 需要知道neighbor的valence
            if (v_valence(r_id) > 0)
                w += 1.0f / (float)v_valence(r_id);
        }
        const float numVert = (float)iter.size();
        const float nrm = 1 / (1 + w / numVert);
        grad = grad * (nrm / numVert) - grad_1 * nrm;
        smooth_grad2(v_id, 0) = grad.x;
        smooth_grad2(v_id, 1) = grad.y;
        smooth_grad2(v_id, 2) = grad.z;
    };

    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::VV>(block, shrd_alloc, smooth, false);
}

void SmoothGrad(RXMeshWrapper* wrapper, bool debug){
    RXMeshDynamicBridge* rx = wrapper->get_bridge();
    // 预先需要先remove相关的attribute
    // printf("Start computing smooth grad\n");

    auto vertex_position = rx->get_input_vertex_coordinates();

    wrapper->remove_attribute("v:f3:smooth_grad1", false); // force remove attribute
    wrapper->remove_attribute("v:f3:smooth_grad2", false);
    wrapper->remove_attribute("v:f1:smooth_score", false);
    auto smooth_grad1 = rx->add_vertex_attribute<float>("v:f3:smooth_grad1", 3);
    auto smooth_grad2 = rx->add_vertex_attribute<float>("v:f3:smooth_grad2", 3);
    auto smooth_score = rx->add_vertex_attribute<float>("v:f1:smooth_score", 1);
    wrapper->m_attributes["v:f3:smooth_grad1"] = smooth_grad1;
    wrapper->m_attributes["v:f3:smooth_grad2"] = smooth_grad2;
    wrapper->m_attributes["v:f1:smooth_score"] = smooth_score;

    auto v_boundary = rx->add_vertex_attribute<bool>("v:b1:boundary", 1);
    auto v_valence = rx->add_vertex_attribute<uint8_t>("v:i1:valence", 1);
    // printf("Finish adding attributes for smooth grad kernel\n");
    // rxmesh::Timers<rxmesh::GPUTimer> timers;
    // timers.add("smooth_grad1");
    // timers.add("smooth_grad2");
    // timers.add("clean_up");
    rx->get_boundary_vertices(*v_boundary); 
    // printf("Finish identifying boundary vertices\n");
    constexpr uint32_t blockThreads = 256;

    rxmesh::LaunchBox<blockThreads> lb_smooth_grad_1;
    rx->update_launch_box(
        {},
        lb_smooth_grad_1,
        (void*)compute_smooth_grad1_kernel<blockThreads>,
        false,  // is_dyn, if there will be dynamic updates
        false,  // oriented, if the query is oriented. Valid only for Op::VV queries
        false,   // with_vertex_valence, if vertex valence is requested to be pre-computed and stored in shared memory
        false   // is_concurrent, in case of multiple queries (i.e. op.size() > 1), indicates if queries needs to be access at the same time
                // user_shmem, a lambda function that takes the number of vertices, edges and faces as input, and returns additional uer-desired shared memory in bytes
    );

    rxmesh::LaunchBox<blockThreads> lb_valence;

    rx->update_launch_box({},
                         lb_valence,
                         (void*)compute_valence<blockThreads>,
                         false,
                         false,
                         true);


    rxmesh::LaunchBox<blockThreads> lb_smooth_grad_2;
    rx->update_launch_box(
        {},
        lb_smooth_grad_2,
        (void*)compute_smooth_grad2_kernel<blockThreads>,
        false, 
        false, 
        false,  
        false  
            
    );
    // printf("Finish updating launch boxes\n");
    // timers.start("smooth_grad1");
    compute_smooth_grad1_kernel<blockThreads>
        <<<lb_smooth_grad_1.blocks,
           lb_smooth_grad_1.num_threads,
           lb_smooth_grad_1.smem_bytes_dyn>>>(rx->get_context(), 
                                        *vertex_position, 
                                        *smooth_grad1,
                                        *smooth_score,
                                        *v_boundary
                                        );

    compute_valence<blockThreads>
        <<<lb_valence.blocks,
           lb_valence.num_threads,
           lb_valence.smem_bytes_dyn>>>(rx->get_context(), *v_valence);
    // timers.stop("smooth_grad1");
    // timers.start("clean_up");           
    // timers.stop("clean_up");
    // printf("Finish computing smooth grad1 and valence for one iteration\n");

    // printf("Finish computing smooth grad1 and valence\n");
    CHECK_CUDA(cudaDeviceSynchronize(), debug);
    // after complishing smooth grad1, free the buffer, and load smooth grad2 kernel
    // timers.start("smooth_grad2");
    compute_smooth_grad2_kernel<blockThreads>
        <<<lb_smooth_grad_2.blocks,
           lb_smooth_grad_2.num_threads,
           lb_smooth_grad_2.smem_bytes_dyn>>>(rx->get_context(), 
                                        *smooth_grad1,
                                        *smooth_grad2,
                                        *v_boundary,
                                        *v_valence
                                        );
    // timers.stop("smooth_grad2");
    // timers.start("clean_up");
    // timers.stop("clean_up");
    // printf("Finish computing smooth grad2 for one iteration\n");
        

    CHECK_CUDA(cudaDeviceSynchronize(), debug);
    rx->remove_attribute("v:i1:valence");
    rx->remove_attribute("v:b1:boundary");
    //
}


template <uint32_t blockThreads>
__global__ void compute_gradient(
    const float ratioRigidityElasticity, const float weightRegularity, const float gstep, 
    rxmesh::Context context,
    rxmesh::VertexAttribute<float> photo_grad,
    rxmesh::VertexAttribute<int> grad_norm,
    rxmesh::VertexAttribute<float> smooth_grad1,
    rxmesh::VertexAttribute<float> smooth_grad2,
    rxmesh::VertexAttribute<float> gv, // debug only, to visualize the gradient magnitude on vertex color
    rxmesh::VertexAttribute<float> gradient 
){
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto apply_grad = [&](VertexHandle v_id, VertexIterator& iter) {
        // const vec3<float> v_pos = vertex_position.to_glm<3>(v_id); 
        const float grad_n = (float)grad_norm(v_id);
        const vec3<float> grad_p = photo_grad.to_glm<3>(v_id);
        const vec3<float> grad_1 = smooth_grad1.to_glm<3>(v_id); 
        const vec3<float> grad_2 = smooth_grad2.to_glm<3>(v_id); 
        vec3<float> grad(0.0f, 0.0f, 0.0f); 
        if (ratioRigidityElasticity >= 1.0f) {
            grad = grad_n > 0 ? 
            grad_p / grad_n + grad_2 * weightRegularity : 
            grad_2 * weightRegularity;
        } else {
            const float rigidity = (1.0f - ratioRigidityElasticity) * weightRegularity;
            const float elasticity = ratioRigidityElasticity * weightRegularity;
            grad = grad_n > 0 ? 
            grad_p / grad_n + grad_2 * elasticity - grad_1 * rigidity :
            grad_2 * elasticity - grad_1 * rigidity;
        }
        // vec3<float> updated_pos = v_pos - grad * gstep;
        // vertex_position(v_id, 0) = updated_pos.x;
        // vertex_position(v_id, 1) = updated_pos.y;
        // vertex_position(v_id, 2) = updated_pos.z;
        gv(v_id) = glm::length(grad); // debug only
        gradient(v_id, 0) = gstep * grad.x;
        gradient(v_id, 1) = gstep * grad.y;
        gradient(v_id, 2) = gstep * grad.z;
    };

    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::VV>(block, shrd_alloc, apply_grad, false);
}


void ComputeGrad(RXMeshWrapper* wrapper, const float ratioRigidityElasticity, const float weightRegularity, const float gstep){

    RXMeshDynamicBridge* rx = wrapper->get_bridge();
    // auto vertex_position = rx->get_input_vertex_coordinates();
    auto photo_grad = std::dynamic_pointer_cast<rxmesh::VertexAttribute<float>>(wrapper->m_attributes["v:f3:photo_grad"]);
    auto grad_norm = std::dynamic_pointer_cast<rxmesh::VertexAttribute<int>>(wrapper->m_attributes["v:i1:grad_norm"]);
    auto smooth_grad1 = std::dynamic_pointer_cast<rxmesh::VertexAttribute<float>>(wrapper->m_attributes["v:f3:smooth_grad1"]);
    auto smooth_grad2 = std::dynamic_pointer_cast<rxmesh::VertexAttribute<float>>(wrapper->m_attributes["v:f3:smooth_grad2"]);
    auto gv = rx->add_vertex_attribute<float>("v:f1:gv_debug", 1); // debug only, to visualize the gradient magnitude on vertex color
    auto gradient = rx->add_vertex_attribute<float>("v:f3:gradient", 3); 
    wrapper->m_attributes["v:f3:gradient"] = gradient;
    wrapper->m_attributes["v:f1:gv_debug"] = gv;
    constexpr uint32_t blockThreads = 256;
    rxmesh::LaunchBox<blockThreads> lb_apply_grad;
    rx->update_launch_box(
        {},
        lb_apply_grad,
        (void*)compute_gradient<blockThreads>,
        false,  
        false, 
        false, 
        false  
               
    );
    compute_gradient<blockThreads>
        <<<lb_apply_grad.blocks,
           lb_apply_grad.num_threads,
           lb_apply_grad.smem_bytes_dyn>>>(ratioRigidityElasticity, weightRegularity, gstep, 
                                        rx->get_context(), 
                                        *photo_grad,
                                        *grad_norm,
                                        *smooth_grad1,
                                        *smooth_grad2,
                                        *gv,
                                        *gradient
                                        );

}












template <typename T, uint32_t blockThreads>
__global__ static void compute_face_normal(const rxmesh::Context      context,
                                             rxmesh::VertexAttribute<T> coords,
                                             rxmesh::FaceAttribute<T> normals)
{
    using namespace rxmesh;
    auto vn_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        // get the face's three vertices coordinates
        vec3<T> c0 = coords.to_glm<3>(fv[0]);
        vec3<T> c1 = coords.to_glm<3>(fv[1]);
        vec3<T> c2 = coords.to_glm<3>(fv[2]);       

        // compute the face normal
        vec3<T> n = cross(c1 - c0, c2 - c0);

        // the three edges length
        // vec3<T> l(glm::distance2(c0, c1),
        //           glm::distance2(c1, c2),
        //           glm::distance2(c2, c0));
        n /= glm::length(n) + 1e-10f; // normalize the normal, add small value to avoid divide-by-zero
        
        normals(face_id, 0) = n.x;
        normals(face_id, 1) = n.y;
        normals(face_id, 2) = n.z;

        // add the face's normal to its vertices
        // for (uint32_t v = 0; v < 3; ++v) {      // for every vertex in this face
        //     for (uint32_t i = 0; i < 3; ++i) {  // for the vertex 3 coordinates
        //         atomicAdd(&normals(fv[v], i), n[i] / (l[v] + l[(v + 2) % 3]));
        //     }
        // }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, vn_lambda);
}

void ComputeFaceNormals(RXMeshWrapper* wrapper) {
    using namespace rxmesh;
    RXMeshDynamicBridge* rx = wrapper->get_bridge();
    auto coords = rx->get_input_vertex_coordinates();

    wrapper->remove_attribute("f:f3:normal", false);
    auto normals = rx->add_face_attribute<float>("f:f3:normal", 3);
    wrapper->m_attributes["f:f3:normal"] = normals;
    normals->reset(0.0f, rxmesh::DEVICE);

    constexpr uint32_t blockThreads = 256;
    LaunchBox<blockThreads> lb_fn;

    rx->update_launch_box(
        {Op::FV},
        lb_fn,
        (void*)compute_face_normal<float, blockThreads>,
        false,
        false,
        false,
        false);

    compute_face_normal<float, blockThreads>
        <<<lb_fn.blocks,
           lb_fn.num_threads,
           lb_fn.smem_bytes_dyn>>>(
            rx->get_context(), *coords, *normals);
}



template <uint32_t blockThreads>
__global__ void updated_positions(rxmesh::Context context, 
    rxmesh::VertexAttribute<float> coords, 
    rxmesh::VertexAttribute<float> gradient
) {
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto update_lambda = [&](VertexHandle v_id, VertexIterator& iter) {
        vec3<float> pos = coords.to_glm<3>(v_id);
        vec3<float> grad = gradient.to_glm<3>(v_id);
        vec3<float> updated_pos = pos - grad; // here we directly apply the gradient step, so gstep is already multiplied in the gradient attribute
        coords(v_id, 0) = updated_pos.x;
        coords(v_id, 1) = updated_pos.y;
        coords(v_id, 2) = updated_pos.z;
    };

    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, update_lambda);
}

void UpdateVertexPositions(RXMeshWrapper* wrapper) {
    using namespace rxmesh;
    RXMeshDynamicBridge* rx = wrapper->get_bridge();
    auto coords = rx->get_input_vertex_coordinates();

    auto gradient = std::dynamic_pointer_cast<VertexAttribute<float>>(wrapper->m_attributes["v:f3:gradient"]);

    // auto v_boundary = rx->add_vertex_attribute<bool>("v:b1:boundary", 1);
    // wrapper->m_attributes["v:b1:boundary"] = v_boundary;
    // v_boundary->reset(false, DEVICE);
    // rx->get_boundary_vertices(*v_boundary);

    constexpr uint32_t blockThreads = 256;
    LaunchBox<blockThreads> lb_update;

    rx->update_launch_box(
        {Op::VV},
        lb_update,
        (void*)updated_positions<blockThreads>,
        false,
        false,
        false,
        false);

    updated_positions<blockThreads>
        <<<lb_update.blocks,
           lb_update.num_threads,
           lb_update.smem_bytes_dyn>>>(
            rx->get_context(), 
            *coords, 
            *gradient
    );
}