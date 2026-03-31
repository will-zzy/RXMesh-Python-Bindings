#include <torch/extension.h>
#include "rxmesh_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RXMesh PyTorch Extension";
    
    py::class_<RXMeshWrapper>(m, "RXMeshWrapper")
        .def(py::init<const std::string&, int, const std::string&, uint32_t, float, float, float>(),
             py::arg("mesh_path"),
             py::arg("device") = 0,
             py::arg("patcher_file") = "",
             py::arg("patch_size") = 256,
             py::arg("capacity_factor") = 3.5,
             py::arg("patch_alloc_factor") = 5.0,
             py::arg("lp_hashtable_load_factor") = 0.5,
             "Create RXMeshWrapper from mesh file")
        .def("copy_vertex_to_tensor", &RXMeshWrapper::copy_vertex_to_tensor, "Copy vertex positions to a new tensor")
        .def("copy_face_indices_to_tensor", &RXMeshWrapper::copy_face_indices_to_tensor, "Copy face indices to a new tensor")
        .def("get_attribute", &RXMeshWrapper::get_attribute, "Get contiguous CUDA tensor for a named RXMesh attribute")
        .def("set_attribute", &RXMeshWrapper::set_attribute, "Set RXMesh attribute from a contiguous CUDA tensor")
        .def("update_vertex_positions", &RXMeshWrapper::update_vertex_positions, "Update vertex positions in RXMesh from the tensor")
        .def("get_num_patches", &RXMeshWrapper::get_num_patches, "Get number of patches")
        .def("get_num_vertices", &RXMeshWrapper::get_num_vertices, "Get number of vertices")
        .def("get_num_faces", &RXMeshWrapper::get_num_faces, "Get number of faces")
        .def("synchronize", &RXMeshWrapper::synchronize, "Synchronize CUDA device")
        .def("add_attribute", &RXMeshWrapper::add_attribute, "Add a new RXMesh attribute with given name and type")
        .def("remove_attribute", &RXMeshWrapper::remove_attribute, "Remove a named RXMesh attribute")
        // customed kernels
        .def("compute_smooth_gradients", &RXMeshWrapper::compute_smooth_gradients, "Compute smooth gradients for each vertex")
        .def("compute_gradients", &RXMeshWrapper::compute_gradients, "Apply gradients to vertex positions with given parameters")
        .def("split_edge", &RXMeshWrapper::split_edge, "Split marked edges using current 'f:b1:need_subdivide' attribute")
        .def("flip_edge", &RXMeshWrapper::flip_edge, "Flip bad diagonals after edge split")
        // .def("subdivide", &RXMeshWrapper::subdivide, "Subdivide the mesh based on current 'f:b1:need_subdivide' attribute")
        .def("compute_face_normal", &RXMeshWrapper::compute_face_normal, "Compute face normals for the mesh");

    m.attr("__version__") = "0.1.0";
}
