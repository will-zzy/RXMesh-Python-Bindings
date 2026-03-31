#include <iostream>


#include <string>

#include <torch/extension.h>

#include "rxmesh_wrapper.h"

int main()
{
    const int    device    = 0;
    const std::string mesh = "../rxmesh/input/sphere3.obj";
    std::cout << "Loading mesh: " << mesh << std::endl;
    RXMeshWrapper wrapper(mesh, device);
    // Optional: ensure device sync before querying
    wrapper.synchronize();
    auto verts = wrapper.copy_vertex_to_tensor();
    std::cout << "rx:vertices shape = [" << verts.size(0) << ", "
              << verts.size(1) << "]" << std::endl;
    std::cout << "dtype = " << verts.dtype() << ", device = "
              << verts.device() << std::endl;
    // Print the first vertex for quick inspection
    if (verts.numel() >= 3) {
        auto v0 = verts[0];
        std::cout << "v0 = [" << v0[0].item<float>() << ", "
                  << v0[1].item<float>() << ", "
                  << v0[2].item<float>() << "]" << std::endl;
    }
    verts[0][0] = -100.0f;
    verts[0][1] = -100.0f;
    verts[0][2] = -100.0f;
    auto verts2 = wrapper.copy_vertex_to_tensor();
    std::cout << "After modification, v0 = [" << verts2[0][0].item<float>() << ", "
              << verts2[0][1].item<float>() << ", "
              << verts2[0][2].item<float>() << "]" << std::endl;
    

    return 0;
}




