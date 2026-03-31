#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include "kernels.h"
#include "rxmesh_bridge.cuh"
#include "rxmesh_wrapper.h"


int is_done(const rxmesh::RXMeshDynamic*             rx,
            const rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
            int*                                     d_buffer)
{
    using namespace rxmesh;

    // if there is at least one edge that is ADDED or SKIP, then we are not done yet
    CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));

    rx->for_each_edge(
        rxmesh::DEVICE,
        [edge_status = *edge_status, d_buffer] __device__(const EdgeHandle eh) {
            if (edge_status(eh) == TO_SPLIT) {
                ::atomicAdd(d_buffer, 1);
            }
        });

    CUDA_ERROR(cudaDeviceSynchronize());
    return d_buffer[0];
}

template <uint32_t blockThreads>
__global__ void printf_edge_flip(
    rxmesh::Context                context,
    rxmesh::VertexAttribute<float>     coords,
    rxmesh::EdgeAttribute<bool>         edge_flip
) {
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Query<blockThreads> query(context);
    auto print_flip = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        if (edge_flip(eh)) {
            printf("flip edge: %f, %f, %f\n"
                "%f, %f, %f\n"
                , coords(iter[0], 0), coords(iter[0], 1), coords(iter[0], 2)
                , coords(iter[2], 0), coords(iter[2], 1), coords(iter[2], 2));
        }

    };
    query.dispatch<Op::EVDiamond>(
        block,
        shrd_alloc,
        print_flip);


}
template <uint32_t blockThreads>
__global__ void printf_all_edges(
    rxmesh::Context                context,
    rxmesh::VertexAttribute<float>     coords,
    rxmesh::EdgeAttribute<bool>         edge_flip
) {
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Query<blockThreads> query(context);
    auto print_flip = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        printf("flip edge: %f, %f, %f     %f, %f, %f\n"
            , coords(iter[0], 0), coords(iter[0], 1), coords(iter[0], 2)
            , coords(iter[2], 0), coords(iter[2], 1), coords(iter[2], 2));
    };
    query.dispatch<Op::EVDiamond>(
        block,
        shrd_alloc,
        print_flip);


}
template <typename T, uint32_t blockThreads>
__global__ void face_subdivide_split(
    rxmesh::Context                     context,
    rxmesh::VertexAttribute<T>          coords,
    rxmesh::EdgeAttribute<EdgeStatus>   edge_status,
    rxmesh::EdgeAttribute<bool>         edge_flip,
    const int iteration)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true);

    if (cavity.patch_id() == INVALID32) {
        return;
    }

    Bitmask is_updated(cavity.patch_info().edges_capacity, shrd_alloc);
    is_updated.reset(block); 
    // 用来记录新增的边并更新status，避免在迭代之间触发旧边的status造成新增边重复split

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    Query<blockThreads> query(context, cavity.patch_id());

    query.dispatch<Op::EVDiamond>(
        block,
        shrd_alloc,
        [&](const EdgeHandle& eh, const VertexIterator& iter) {
            assert(iter.size() == 4);
            if (edge_status(eh) == TO_SPLIT){ 
                // 如果遇上了标记的边，则判断
                // 为了避免单个 patch 的 cavity 数量超过 CavityManager
                // 的假设上限，这里按 pass(iteration) 对边做简单子采样，
                // 只在当前 pass 处理一半左右的候选边，其余留到后续 pass。
                // if (((eh.local_id() + iteration) & 1) != 0) {
                //     return;
                // }
                // printf("pass: %d\n", iteration);
                cavity.create(eh);
            }
            
        });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status, edge_flip)) {
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            if (size != 4) {
                return;
            }
            // E-cavity :
            // 0 = seed edge one end (v0)
            // 1 = one side opposite vertex (may be invalid)
            // 2 = seed edge other end (v1)
            // 3 = other side opposite vertex (may be invalid)
            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);
            // printf("v0_pos: (%f, %f, %f), v1_pos: (%f, %f, %f)\n"
            //        , coords(v0, 0), coords(v0, 1), coords(v0, 2)
            //        , coords(v1, 0), coords(v1, 1), coords(v1, 2));
            const VertexHandle new_v = cavity.add_vertex();
            if (!new_v.is_valid()) {
                return;
            }

            coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
            coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
            coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * T(0.5);

            DEdgeHandle e0 = cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
            const DEdgeHandle e_init = e0;

            if (!e0.is_valid()) {
                // printf("a!\n");
                return;
            }

            is_updated.set(e0.local_id(), true);
            // edge_status(e0.get_edge_handle()) = ADDED;

            for (uint16_t i = 0; i < size; ++i) {
                const DEdgeHandle e = cavity.get_cavity_edge(c, i);
                const EdgeHandle eh = e.get_edge_handle();
                const EdgeHandle eh_next = cavity.get_cavity_edge(c, (i + 1) % size).get_edge_handle();

                const DEdgeHandle e1 =
                    (i == size - 1) ?
                        e_init.get_flip_dedge() :
                        cavity.add_edge(cavity.get_cavity_vertex(c, i + 1), new_v);

                if (!e1.is_valid()) {
                    // printf("b!\n");
                    break;
                }
                // edge_status(e1.get_edge_handle()) = ADDED;
                if (i != size - 1) {
                    is_updated.set(e1.local_id(), true);
                }
                
                
                const FaceHandle f = cavity.add_face(e0, e, e1);
                if (!f.is_valid()) {
                    
                    const VertexHandle v00 = cavity.get_cavity_vertex(c, (i + 1) % size);
                    const VertexHandle v11 = new_v;
                    const VertexHandle v22 = cavity.get_cavity_vertex(c, i);
                    printf("c! face vertex pos: %f, %f, %f\n"
                    "%f, %f, %f\n"
                    "%f, %f, %f\n"
                    ,
                    coords(v00, 0), coords(v00, 1), coords(v00, 2),
                    coords(v11, 0), coords(v11, 1), coords(v11, 2),
                    coords(v22, 0), coords(v22, 1), coords(v22, 2)
                    );
                    break;
                }

                // 有没有可能如果我新增边还没来及更新status，其他cavity就访问到这个边的status从记录了错误的flip edge?
                // 没可能，因为相邻cavity是互斥的
                // 如果两条边都是TO_SPLIT，且iteration对齐，那么新边就是diamond的对角线，需要flip
                if (edge_status(eh) == TO_SPLIT && edge_status(eh_next) == TO_SPLIT && i % 2 == 0) {
                    // flip = true; // If both adjacent edges of the diamond are original edges (TO_SPLIT), the new edge is a bad diagonal and needs to be flipped
                    EdgeHandle eh_flip = e1.get_edge_handle();
                    edge_flip(eh_flip) = true; 
                    const VertexHandle v00 = cavity.get_cavity_vertex(c, i + 1);
                    const VertexHandle v11 = new_v;
                    if ((abs(coords(v00, 0)-3.0f) < 1e-2 && abs(coords(v00, 1)-4.0f) < 1e-2 && abs(coords(v11, 2)-2.0f) < 1e-2 && abs(coords(v11, 2)-3.5f) < 1e-2) || 
                        (abs(coords(v11, 0)-3.0f) < 1e-2 && abs(coords(v11, 2)-4.0f) < 1e-2 && abs(coords(v00, 0)-2.0f) < 1e-2 && abs(coords(v00, 2)-3.5f) < 1e-2)
                        ) {
                        printf("flip edge vertex: %f, %f, %f\n"
                            "%f, %f, %f\n"
                            , coords(v00, 0), coords(v00, 1), coords(v00, 2)
                            , coords(v11, 0), coords(v11, 1), coords(v11, 2)
                        );
                    }
                }
                
                e0 = e1.get_flip_dedge();
            }
            // const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);
            // edge_status(src) = SKIP; // mark the original edge as ADDED to avoid it being processed in later passes
            // is_updated.set(src.local_id(), true);
        });
    }

    cavity.epilogue(block);
    block.sync();
    if (cavity.is_successful()) { // 如果所有的cavities都成功了，才更新
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) { // 记得原先的也要设为false
                edge_status(eh) = ADDED;
            } 
        });

    } 
}



template <uint32_t blockThreads>
__global__ void mark_bdry_edges(
    rxmesh::Context             context,
    rxmesh::EdgeAttribute<bool> edge_bdry)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Query<blockThreads> query(context);
    
    query.dispatch<Op::EF>(
        block,
        shrd_alloc,
        [&](const EdgeHandle& eh, const FaceIterator& iter) {
            if (iter.size() < 2) {
                edge_bdry(eh) = true; // boundary edge won't be split
            }
        });
    block.sync();
}



template <uint32_t blockThreads>
__global__ void mark_split_edges_from_faces(
    rxmesh::Context                   context,
    rxmesh::FaceAttribute<bool>       face_to_split,
    rxmesh::EdgeAttribute<EdgeStatus> edge_status,
    rxmesh::EdgeAttribute<bool>       edge_bdry
)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Query<blockThreads> query(context);
    // 先mark edge是否是边界边，再根据边界边决定哪些face需要subdivide，需要subdivide的face三条边都标记为TO_SPLIT
    
    query.dispatch<Op::FE>(
        block,
        shrd_alloc,
        [&](const FaceHandle& fh, const EdgeIterator& eiter) {
            if (face_to_split(fh)){
                bool has_bdry_edge = false;
                for (uint16_t i = 0; i < eiter.size(); ++i) {
                    const EdgeHandle eh = eiter[i];
                    if (edge_bdry(eh)) {
                        has_bdry_edge = true;
                        break;
                    }
                }

                if (!has_bdry_edge) {
                    for (uint16_t i = 0; i < eiter.size(); ++i) {
                        const EdgeHandle eh = eiter[i];
                        if (edge_status(eh) != TO_SPLIT) { // in case one edge is shared by two faces that are both marked to split
                            edge_status(eh) = TO_SPLIT;
                        }
                    }
                } 
            }
        });
}


template <typename T, uint32_t blockThreads>
__global__ void face_subdivide_flip(
    rxmesh::Context                context,
    rxmesh::VertexAttribute<T>     coords,
    rxmesh::EdgeAttribute<EdgeStatus> edge_status,
    rxmesh::EdgeAttribute<bool>  edge_flip,
    int* d_buffer
)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false, false);

    if (cavity.patch_id() == INVALID32) {
        return;
    }

    Bitmask is_updated(cavity.patch_info().edges_capacity, shrd_alloc);
    is_updated.reset(block);
    block.sync();
    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // uint16_t* v_info =
    //     shrd_alloc.alloc<uint16_t>(2 * cavity.patch_info().num_vertices[0]);
    // fill_n<blockThreads>(
    //     v_info, 2 * cavity.patch_info().num_vertices[0], uint16_t(INVALID16));

    // // a bitmask that indicates which edge we want to flip
    // Bitmask e_flip(cavity.patch_info().num_edges[0], shrd_alloc);
    // e_flip.reset(block);


    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(
        block,
        shrd_alloc,
        [&](const EdgeHandle& eh, const VertexIterator& iter) {
            const bool is_bdry = (!iter[1].is_valid() || !iter[3].is_valid());
            // edge_boundary(eh) = is_bdry;

            if (!is_bdry && edge_status(eh) == UNSEEN && edge_flip(eh) == true) {
                cavity.create(eh);
            }
        });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status, edge_flip)) {
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            if (size != 4) {
                return;
            }
            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));


            if (new_edge.is_valid()) {
                is_updated.set(new_edge.local_id(), true);
                edge_flip(new_edge.get_edge_handle()) = false; // don't flip the new edge
                cavity.add_face(cavity.get_cavity_edge(c, 0),
                                new_edge,
                                cavity.get_cavity_edge(c, 3));


                cavity.add_face(cavity.get_cavity_edge(c, 1),
                                cavity.get_cavity_edge(c, 2),
                                new_edge.get_flip_dedge());
            }

        });
    }
    cavity.epilogue(block);
    block.sync();

    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (is_updated(eh.local_id())) {
            edge_status(eh) = ADDED;
        } 
        if (edge_flip(eh) == true) {
            atomicAdd(d_buffer, 1);
            //  printf("flip edge %u: %f, %f, %f -- %f, %f, %f\n", eh.local_id(),
            //     coords(cavity.get_cavity_vertex(c, 0), 0), coords(cavity.get_cavity_vertex(c, 0), 1), coords(cavity.get_cavity_vertex(c, 0), 2),
            //     coords(cavity.get_cavity_vertex(c, 2), 0), coords(cavity.get_cavity_vertex(c, 2), 1), coords(cavity.get_cavity_vertex(c, 2), 2));
        }
    });
}



void SplitEdges(RXMeshWrapper* wrapper)
{
    using namespace rxmesh;

    RXMeshDynamicBridge* rx = wrapper->get_bridge();
    auto coords = rx->get_input_vertex_coordinates();

    auto need_subdivide =
        std::dynamic_pointer_cast<FaceAttribute<bool>>(
            wrapper->m_attributes["f:b1:need_subdivide"]);

    // wrapper->remove_attribute("v:b1:boundary", false);
    wrapper->remove_attribute("e:b1:boundary", false);
    wrapper->remove_attribute("e:c1:status", false);
    wrapper->remove_attribute("e:b1:flip", false);

    // auto v_boundary    = rx->add_vertex_attribute<bool>("v:b1:boundary", 1);
    auto edge_boundary = rx->add_edge_attribute<bool>("e:b1:boundary", 1);
    auto edge_status   = rx->add_edge_attribute<EdgeStatus>("e:c1:status", 1);
    auto edge_flip     = rx->add_edge_attribute<bool>("e:b1:flip", 1);

    // wrapper->m_attributes["v:b1:boundary"]      = v_boundary;
    wrapper->m_attributes["e:b1:flip"]      = edge_flip;
    wrapper->m_attributes["e:c1:status"]      = edge_status;
    // wrapper->m_attributes["e:c1:status"]      = edge_status;
    // v_boundary->reset(false, DEVICE);
    edge_boundary->reset(false, DEVICE);
    edge_status->reset(SKIP, DEVICE);
    edge_flip->reset(false, DEVICE);
    // rx->get_boundary_vertices(*v_boundary);
    // get_boundary_vertices(wrapper, *v_boundary);
    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));



    constexpr uint32_t blockThreads = 256;

    // Pass 1: split all edges touched by red faces
    LaunchBox<blockThreads> lb_mark_bdry_edges;
    rx->update_launch_box(
        {Op::EF},
        lb_mark_bdry_edges,
        (void*)mark_bdry_edges<blockThreads>,
        false,
        false,        
        false,
        false,
        [&](uint32_t, uint32_t e, uint32_t) {
            return rxmesh::detail::mask_num_bytes(e) +
                   ShmemAllocator::default_alignment;
        });


    mark_bdry_edges<blockThreads>
        <<<lb_mark_bdry_edges.blocks,
           lb_mark_bdry_edges.num_threads,
           lb_mark_bdry_edges.smem_bytes_dyn>>>(
            rx->get_context(),
            *edge_boundary);



    LaunchBox<blockThreads> lb_mark_edges;
    rx->update_launch_box(
        {Op::FE},
        lb_mark_edges,
        (void*)mark_split_edges_from_faces<blockThreads>,
        false,
        false,
        false,
        false,
        [&](uint32_t v, uint32_t e, uint32_t f) {
            return rxmesh::detail::mask_num_bytes(e) +
                   ShmemAllocator::default_alignment;
        });
        
    mark_split_edges_from_faces<blockThreads>
        <<<lb_mark_edges.blocks,
           lb_mark_edges.num_threads,
           lb_mark_edges.smem_bytes_dyn>>>(
            rx->get_context(),
            *need_subdivide,
            *edge_status,
            *edge_boundary
        );

    LaunchBox<blockThreads> lb_split;
    rx->update_launch_box(
        {Op::EVDiamond},
        lb_split,
        (void*)face_subdivide_split<float, blockThreads>,
        true,
        false,
        false,
        false,
        [&](uint32_t, uint32_t e, uint32_t) {
            return 4 * rxmesh::detail::mask_num_bytes(e) +
                ShmemAllocator::default_alignment;
        });
    
    // Debug: validate right after marking split edges, before any split
    // rx->update_host();
    // if (!rx->validate()) {
    //     RXMESH_ERROR("Validation failed after mark_split_edges_from_faces");
    // }
    // printf("%d, %d, %d\n", lb_mark_bdry_edges.smem_bytes_dyn, lb_mark_edges.smem_bytes_dyn, lb_split.smem_bytes_dyn);
    constexpr int kMaxSplitPasses = 12;
    // printf("blocks, %d, num_threads: %d\n", lb_split.blocks, lb_split.num_threads);
    for (int pass = 0; pass < kMaxSplitPasses; ++pass) {
        const uint32_t vertices_before = rx->get_num_vertices(true);
        rx->reset_scheduler();
        int inner_iter = 0;
        constexpr int kMaxInnerIter = 12;

        // printf("[Subdivide] pass %d: vertices = %u\n", pass, vertices_before);
        while (!rx->is_queue_empty()) {
            if (++inner_iter > kMaxInnerIter) {
                // printf("[Subdivide] stop inner loop at pass %d due to kMaxInnerIter=%d\n", pass, kMaxInnerIter);
                break;
            }
            // printf("[Subdivide] pass %d, inner iter %d\n", pass, inner_iter);
            face_subdivide_split<float, blockThreads>
                <<<lb_split.blocks,
                   lb_split.num_threads,
                   lb_split.smem_bytes_dyn>>>(
                    rx->get_context(),
                    *coords,
                    *edge_status,
                    *edge_flip,
                    pass);
            cudaDeviceSynchronize();
            // printf("launch kernel done!\n");
            rx->cleanup();
            cudaDeviceSynchronize();
            // printf("launch first clean done!\n");
            rx->slice_patches(*coords, *edge_status, *edge_flip);
            cudaDeviceSynchronize();
            // printf("launch slice_patch done!\n");
            rx->cleanup();
            cudaDeviceSynchronize();
            // printf("launch second clean done!\n");
        }

        // Slice once per pass (instead of each inner iteration) to avoid
        // queue feedback loops that can prevent convergence.
        int done = is_done(rx, edge_status.get(), d_buffer);
        // const uint32_t vertices_after = rx->get_num_vertices(true);
        if (done == 0) {
            // pass++;
            // continue;
            break;
        }
        
    }

    CHECK_CUDA(cudaDeviceSynchronize(), false);


    LaunchBox<blockThreads> lb_print_flip;
    rx->update_launch_box(
        {Op::EVDiamond},
        lb_print_flip,
        (void*)printf_all_edges<blockThreads>,
        false,
        false,
        false,
        false);
    printf_all_edges<blockThreads> <<< lb_print_flip.blocks,
                       lb_print_flip.num_threads,
                       lb_print_flip.smem_bytes_dyn>>>(
        rx->get_context(),
        *coords,
        *edge_flip);
    CHECK_CUDA(cudaDeviceSynchronize(), false);



    rx->update_host();
    if (!rx->validate()) { // 一旦需要validate，一定要先update_host
        RXMESH_ERROR("Validation failed after SplitEdges");
    }
    rx->remove_attribute("e:b1:boundary");
    // rx->remove_attribute("v:b1:boundary");
    // rx->remove_attribute("e:c1:status");
}

void FlipEdges(RXMeshWrapper* wrapper)
{
    using namespace rxmesh;

    RXMeshDynamicBridge* rx = wrapper->get_bridge();
    auto coords = rx->get_input_vertex_coordinates();


    // The attributes might have been cleaned up by SplitEdges.
    // Instead of forcing removal, allow add_attribute to overwrite if existing.

    auto v_boundary  = rx->add_vertex_attribute<bool>("v:b1:boundary", 1);
    auto edge_status = rx->add_edge_attribute<EdgeStatus>("e:c1:status", 1);
    auto edge_flip =
        std::dynamic_pointer_cast<EdgeAttribute<bool>>(
            wrapper->m_attributes["e:b1:flip"]);

    // wrapper->m_attributes["v:b1:boundary"] = v_boundary;
    // wrapper->m_attributes["e:c1:status"] = edge_status;

    edge_status->reset(UNSEEN, DEVICE);
    
    // rx->update_host();

    rx->get_boundary_vertices(*v_boundary);
    // get_boundary_vertices(wrapper, *v_boundary);

    constexpr uint32_t blockThreads = 256;

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));
    int prv_remaining_work = rx->get_num_edges();

    LaunchBox<blockThreads> lb_flip;
    rx->update_launch_box(
        {Op::EVDiamond, Op::VV},
        lb_flip,
        (void*)face_subdivide_flip<float, blockThreads>,
        true,
        false,
        false,
        false,
        [&](uint32_t v, uint32_t e, uint32_t) {
               return 3 * rxmesh::detail::mask_num_bytes(e) +
                 sizeof(uint16_t) * 2 * v +
                 4 * ShmemAllocator::default_alignment;
        });

    constexpr int kMaxFlipPasses = 8;
    for (int pass = 0; pass < kMaxFlipPasses; ++pass) {
        rx->reset_scheduler();

        int inner_iter = 0;
        constexpr int kMaxFlipInnerIter = 10;
        while (!rx->is_queue_empty()) {
            if (++inner_iter > kMaxFlipInnerIter) {
                // printf("[Flip] stop inner loop at pass %d due to kMaxFlipInnerIter=%d\n", pass, kMaxFlipInnerIter);
                break;
            }

            face_subdivide_flip<float, blockThreads>
                <<<lb_flip.blocks,
                   lb_flip.num_threads,
                   lb_flip.smem_bytes_dyn>>>(
                    rx->get_context(),
                    *coords,
                    *edge_status,
                    *edge_flip,
                    d_buffer);

            rx->cleanup();
            rx->slice_patches(*coords, *edge_status, *edge_flip);
            rx->cleanup();
        }

        // int remaining_work = is_done(rx, edge_status.get(), d_buffer);
        int d = 0;
        cudaMemcpy(&d, d_buffer, sizeof(int), cudaMemcpyDeviceToHost);
        int remaining_work = d;
        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            // printf("[Flip] stop at pass %d as remaining work = %d\n", pass, remaining_work);
            break;
        }
        prv_remaining_work = remaining_work;
    }

    CHECK_CUDA(cudaDeviceSynchronize(), false);

    rx->update_host();

    wrapper->remove_attribute("e:c1:status", false);
    // wrapper->remove_attribute("v:b1:boundary", false);
    CUDA_ERROR(cudaFree(d_buffer));
}


// // DONT TOUCH FACES WHICH INCIDENT TO BOUNDARY EDGE!!!
// // UNSTABLE!!!
// template <uint32_t blockThreads>
// __global__ static void subdivide_faces(
//     rxmesh::Context context,
//     rxmesh::VertexAttribute<float> coords,
//     rxmesh::FaceAttribute<bool> need_subdivide,
//     rxmesh::FaceAttribute<FaceStatus> face_status,
//     rxmesh::EdgeAttribute<bool> edge_boundary 
// )
// {
//     using namespace rxmesh;
//     auto block = cooperative_groups::this_thread_block();

//     ShmemAllocator shrd_alloc;

//     CavityManager<blockThreads, CavityOp::FE> cavity(block, context, shrd_alloc, true);

//     if (cavity.patch_id() == INVALID32) {
//         return;
//     }
//     Bitmask is_updated(cavity.patch_info().faces_capacity, shrd_alloc);
//     auto should_refine = [&](const FaceHandle& fh, const VertexIterator& iter) {
//         if (need_subdivide(fh)) {
//             printf("cavity vertex positions:\n");
//              for (uint16_t i = 0; i < iter.size(); ++i) {
//                 const VertexHandle vh = iter[i];
//                 printf("vertex %u: %f, %f, %f\n", vh.local_id(),
//                     coords(vh, 0), coords(vh, 1), coords(vh, 2));
//             }
//             if (!edge_boundary(cavity.get_cavity_edge(fh, 0).get_edge_handle()) &&
//                 !edge_boundary(cavity.get_cavity_edge(fh, 1).get_edge_handle()) &&
//                 !edge_boundary(cavity.get_cavity_edge(fh, 2).get_edge_handle())) {
                
//                 cavity.create(fh);
//             } else{
//                 face_status(fh) = FSKIP; // 有没有可能某个面subdivide之后，它的邻面(需要被subdivide的)就不是bdry面了？
//                 // 有可能，所以提前标记为SKIP
//                 // 但是不管，
//             }
//         } else {
//             face_status(fh) = FSKIP; // 不需要subdivide的面无论邻面怎么细分，这个面都不会被subdivide
//         }
//     };


//     Query<blockThreads> query(context, cavity.patch_id());
//     query.dispatch<Op::FV>(block, shrd_alloc, should_refine);
//     block.sync();


//     if (cavity.prologue(block, shrd_alloc, coords, need_subdivide, face_status, edge_boundary)){
//         cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
//             printf("cavity size = %u\n", size);
//             for (int i = 0; i < size; ++i) {
//                 printf("cavity bdry vertex %u: %f, %f, %f\n", i,
//                     coords(cavity.get_cavity_vertex(c, i), 0),
//                     coords(cavity.get_cavity_vertex(c, i), 1),
//                     coords(cavity.get_cavity_vertex(c, i), 2));
//             }

//             // if (size != 3) {
//             //     return;
//             // }
//             // const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
//             // const VertexHandle v1 = cavity.get_cavity_vertex(c, 1);
//             // const VertexHandle v2 = cavity.get_cavity_vertex(c, 2);

//             // const VertexHandle new_v01 = cavity.add_vertex();
//             // const VertexHandle new_v12 = cavity.add_vertex();
//             // const VertexHandle new_v20 = cavity.add_vertex();
//             // if (!new_v01.is_valid() || !new_v12.is_valid() || !new_v20.is_valid()) {
//             //     return;
//             // }

//             // coords(new_v01, 0) = (coords(v0, 0) + coords(v1, 0)) / 2.0f;
//             // coords(new_v01, 1) = (coords(v0, 1) + coords(v1, 1)) / 2.0f;
//             // coords(new_v01, 2) = (coords(v0, 2) + coords(v1, 2)) / 2.0f;

//             // DEdgeHandle e01 = cavity.add_edge(new_v01, v0);
//             // DEdgeHandle e12 = cavity.add_edge(new_v12, v1);
//             // DEdgeHandle e20 = cavity.add_edge(new_v20, v2);

//             // if (!e01.is_valid() || !e12.is_valid() || !e20.is_valid()) {
//             //     return;
//             // }

//             // cavity.add_face(e01, e12.get_flip_dedge(), e20.get_flip_dedge());
//             // cavity.add_face(e12, e20.get_flip_dedge(), e01.get_flip_dedge());
//             // cavity.add_face(e20, e01.get_flip_dedge(), e12.get_flip_dedge());
//         });
//     }




// }


// template <uint32_t blockThreads>
// __global__ static void mark_bdry_edges(
//     rxmesh::Context context,
//     rxmesh::EdgeAttribute<bool> edge_bdry)
// {
//     using namespace rxmesh;
//     auto block = cooperative_groups::this_thread_block();
//     ShmemAllocator shrd_alloc;
//     Query<blockThreads> query(context);
//     query.dispatch<Op::EF>(block, shrd_alloc, [&](const EdgeHandle& eh, FaceIterator& iter) {
//         if (iter.size() < 2) {
//             edge_bdry(eh) = true;
//         }
//     });
//     block.sync();
// }


// void Subdivide(RXMeshWrapper* wrapper)
// {
//     using namespace rxmesh;
//     // 
//     RXMeshDynamicBridge* rx = wrapper->get_bridge();
//     auto need_subdivide = std::dynamic_pointer_cast<FaceAttribute<bool>>(
//         wrapper->m_attributes["f:b1:need_subdivide"]);
//     auto face_status = rx->add_face_attribute<FaceStatus>("f:c1:status", 1);
//     auto edge_bdry = rx->add_edge_attribute<bool>("e:b1:boundary", 1);
//     auto coords = rx->get_input_vertex_coordinates();
//     constexpr uint32_t blockThreads = 256;

//     edge_bdry.reset(false, DEVICE);
//     face_status.reset(FUNSEEN, DEVICE);
//     LaunchBox<blockThreads> lb_mark_bdry_edges;
//     rx->prepare_launch_box({Op::E}, lb_mark_bdry_edges, (void*)mark_bdry_edges<bool, blockThreads>);
//     mark_bdry_edges<blockThreads><<<lb_mark_bdry_edges.blocks,
//                           lb_mark_bdry_edges.num_threads,
//                           lb_mark_bdry_edges.smem_bytes_dyn>>>(
//                         rx->get_context(), *edge_bdry);




//     int iter = 0;
    
//     while (!rx->is_queue_empty()) {
//         RXMESH_INFO("iter = {}", ++iter);
//         LaunchBox<blockThreads> launch_box;
//         rx->prepare_launch_box({}, launch_box, (void*)subdivide_faces<blockThreads>);

//         subdivide_faces<blockThreads>
//             <<<launch_box.blocks,
//                launch_box.num_threads,
//                launch_box.smem_bytes_dyn>>>(
//                     rx->get_context(), 
//                     *coords, 
//                     *need_subdivide,
//                     *face_status,
//                     *edge_bdry
//                 );

//         rx->slice_patches(*coords, *need_subdivide, *face_status, *edge_bdry);
//         rx->cleanup();
//     }

//     CUDA_ERROR(cudaDeviceSynchronize());
//     rx->update_host();
//     // SplitEdges(wrapper);
//     // FlipEdges(wrapper);
// }














// void SplitLongEdges(RXMeshWrapper* wrapper) {
    
//     using namespace rxmesh;

//     RXMeshDynamicBridge* rx = wrapper->get_bridge();
//     auto coords = rx->get_input_vertex_coordinates();

//     auto need_subdivide =
//         std::dynamic_pointer_cast<FaceAttribute<bool>>(
//             wrapper->m_attributes["f:b1:need_subdivide"]);

//     wrapper->remove_attribute("v:b1:boundary", false);
//     wrapper->remove_attribute("e:b1:boundary", false);
//     wrapper->remove_attribute("e:c1:status", false);
//     wrapper->remove_attribute("e:b1:flip", false);

//     auto v_boundary    = rx->add_vertex_attribute<bool>("v:b1:boundary", 1);
//     auto edge_boundary = rx->add_edge_attribute<bool>("e:b1:boundary", 1);
//     auto edge_status   = rx->add_edge_attribute<EdgeStatus>("e:c1:status", 1);
//     auto edge_flip     = rx->add_edge_attribute<bool>("e:b1:flip", 1);

//     // wrapper->m_attributes["v:b1:boundary"]      = v_boundary;
//     wrapper->m_attributes["e:b1:flip"]      = edge_flip;
//     // wrapper->m_attributes["e:c1:status"]      = edge_status;
//     v_boundary->reset(false, DEVICE);
//     edge_boundary->reset(false, DEVICE);
//     edge_status->reset(UNSEEN, DEVICE);
// }



