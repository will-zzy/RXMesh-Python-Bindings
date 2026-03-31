# RXMesh PyTorch Binding

A lightweight PyTorch extension that wraps **[RXMesh](https://github.com/owensgroup/RXMesh)** and exposes mesh topology, attributes, and several custom CUDA kernels to Python.

Thanks [Ahdhn](https://github.com/Ahdhn) for your help!

This project is intended for workflows where:

- mesh topology is managed by **RXMesh**,
- high-level logic is written in **Python/PyTorch**,
- per-vertex / per-edge / per-face data is exchanged through CUDA tensors,
- topology-changing operations such as **edge split** and **edge flip** are executed inside RXMesh and then synchronized back to PyTorch.

---

## Features

- Load a mesh into **RXMeshDynamic**
- Export vertex positions and face indices as CUDA tensors
- Add / remove / read / write RXMesh attributes from PyTorch
- Run custom CUDA kernels from Python:
  - smooth gradient computation
  - gradient aggregation
  - edge split
  - edge flip
  - face normal computation
  - vertex position update
- Keep RXMesh implementation details hidden using a **PIMPL** wrapper

---

## Project Layout

Because **RXMesh** is an **INTERFACE** library, this binding is built from the **top-level CMake project** rather than as a completely standalone package.

A typical layout is:

```text
Main-Proj
├── rxmesh
│   ├── CMakeLists.txt
│   ├── apps
│   ├── include
│   │   └── rxmesh/...
│   └── ...
│
├── rxmesh_binding
│   ├── csrc
│   ├── CMakeLists.txt
│   └── setup.py
│
└── CMakeLists.txt
```

After running:

```bash
pip install ./rxmesh_binding -v
```

`setup.py` invokes the **top-level `CMakeLists.txt`**, and both `rxmesh` and `rxmesh_binding` are compiled together as subdirectories.

This approach is used because RXMesh is not linked as a traditional prebuilt shared library. Instead, the wrapper is compiled in the same build graph as RXMesh.

---

## CMake Design

Since RXMesh is an interface-style library, `rxmesh_binding` does not try to build RXMesh independently. Instead, the top-level project owns the full build process.

In practice:

- the top-level `CMakeLists.txt` includes both `rxmesh` and `rxmesh_binding`,
- `setup.py` delegates to that top-level build,
- the binding can directly use RXMesh headers, templates, and CUDA kernels,
- no modifications to the RXMesh source tree are required.

This is especially useful because RXMesh is heavily template-based and tightly integrated with its CMake build graph.

---

## RXMeshWrapper

### PIMPL Pattern

`RXMeshWrapper` uses a PIMPL-style internal implementation so that RXMesh internals stay out of the public header and the original RXMesh source code does not need to be modified.

Conceptually:

```cpp
struct RXMeshWrapper::Impl {
    std::unique_ptr<RXMeshDynamicBridge> rxmesh;
    void* rxmesh_ptr;
};
```

Goals of this design:

1. hide RXMesh implementation details from the Python-facing wrapper,
2. reduce compile-time coupling in the public header,
3. keep the wrapper maintainable without changing RXMesh itself.

---

## Python Binding

The extension exposes a Python class:

```python
rx.RXMeshWrapper(...)
```

through `pybind11`.

Currently exposed methods include:

- `copy_vertex_to_tensor`
- `copy_face_indices_to_tensor`
- `get_attribute`
- `set_attribute`
- `update_vertex_positions`
- `get_num_patches`
- `get_num_vertices`
- `get_num_faces`
- `synchronize`
- `add_attribute`
- `remove_attribute`
- `compute_smooth_gradients`
- `compute_gradients`
- `split_edge`
- `flip_edge`
- `compute_face_normal`

---

## Initialization

A wrapper is created from a mesh file:

```python
import rxmesh_torch_ops as rx

wrapper = rx.RXMeshWrapper(
    mesh_path="input.obj",
    device=0,
    patcher_file="",
    patch_size=256,
    capacity_factor=3.5,
    patch_alloc_factor=5.0,
    lp_hashtable_load_factor=0.5,
)
```

### Constructor arguments

- `mesh_path`: input mesh path
- `device`: CUDA device id
- `patcher_file`: optional RXMesh patcher file
- `patch_size`: target patch size
- `capacity_factor`: patch capacity multiplier
- `patch_alloc_factor`: patch allocation multiplier
- `lp_hashtable_load_factor`: load factor used by RXMesh local hash tables

### What happens during initialization

1. `rxmesh::rx_init(device)` is called once per process
2. `RXMeshDynamicBridge` is created
3. `validate()` is called on the initial mesh
4. prefix arrays are rebuilt and synchronized to host

---

## Tensor / Attribute / Face Transformation

RXMesh stores topology and attributes **per vertex / per edge / per face**, distributed over patches.  
PyTorch, on the other hand, expects flat dense tensors.

The wrapper is responsible for transforming between these two representations.

### Vertex Positions → Tensor

`copy_vertex_to_tensor()` exports the RXMesh input vertex coordinates to a CUDA tensor of shape:

```python
[V, C]
```

For standard positions, this is usually:

```python
[V, 3]
```

Internally, the wrapper iterates over all active owned vertices and writes them into a contiguous CUDA tensor using RXMesh linear ids.

---

### Face Indices → Tensor

`copy_face_indices_to_tensor()` exports faces as a tensor of shape:

```python
[F, 3]
```

Each face is converted from patch-local topology into **global linear vertex indices**. This is why valid prefix arrays must exist before exporting face tensors.

---

### Why Prefix Rebuild Is Necessary

RXMesh stores vertices / edges / faces patch-locally. The wrapper uses **global linear ids** when exporting tensors into PyTorch. Therefore, after any topology-changing operation such as `split_edge()` or `flip_edge()`, the prefix arrays must be rebuilt.

This is handled by:

```cpp
rebuild_prefix_or_throw(this);
```

Without this step, exported tensors—especially face indices—would become inconsistent.

---

## Attribute System

RXMesh stores attributes separately by mesh element type:

- per-vertex attributes
- per-edge attributes
- per-face attributes

The wrapper exposes these attributes to Python as CUDA tensors.

### Naming Convention

The wrapper identifies attributes using the following format:

```text
[element_type]:[dtype][num_attr]:[attribute_name]
```

Examples:

- `v:f3:gradient` → vertex attribute, float, 3 channels
- `f:b1:need_subdivide` → face attribute, bool, 1 channel
- `e:c1:status` → edge attribute, int8 / enum-like scalar, 1 channel

### Meaning of each field

- `element_type`
  - `v` = vertex
  - `e` = edge
  - `f` = face

- `dtype`
  - `f` = float
  - `i` = int32
  - `b` = bool
  - `c` = int8 / char-like storage

- `num_attr`
  - number of channels, usually `1` to `4`

This convention is parsed by regex inside the wrapper and used to dispatch to the correct RXMesh attribute type.

---

## Supported Attribute Types

The wrapper currently supports:

- `VertexAttribute<float>`
- `VertexAttribute<int32_t>`
- `VertexAttribute<bool>`
- `VertexAttribute<int8_t>`

- `EdgeAttribute<float>`
- `EdgeAttribute<int32_t>`
- `EdgeAttribute<bool>`
- `EdgeAttribute<int8_t>`

- `FaceAttribute<float>`
- `FaceAttribute<int32_t>`
- `FaceAttribute<bool>`
- `FaceAttribute<int8_t>`

Dispatch is implemented through macro-based type selection in `wrapper.cu`.

---

## Attribute API

### Add an Attribute

```python
wrapper.add_attribute("v:f3:gradient")
```

If needed, the wrapper creates the corresponding RXMesh attribute using the parsed element type, scalar type, and channel count.

---

### Remove an Attribute

```python
wrapper.remove_attribute("v:f3:gradient")
```

This removes the attribute from both:

- RXMesh internal storage
- the wrapper's `m_attributes` map

---

### Set an Attribute from a Tensor

```python
wrapper.set_attribute("f:b1:need_subdivide", tensor, force_add=True)
```

Requirements:

- tensor must be on CUDA
- tensor must be contiguous
- tensor dtype must match the declared attribute type
- tensor shape must match `[num_elements, num_channels]`

If `force_add=True`, the wrapper removes any old attribute with the same name and recreates it. This is especially useful after topology changes, where the old attribute size may no longer match the current mesh.

---

### Read an Attribute as a Tensor

```python
tensor = wrapper.get_attribute("e:c1:status")
```

This exports the RXMesh attribute into a new contiguous CUDA tensor.

---

## Custom CUDA Kernels

The wrapper currently exposes several topology-aware or geometry-aware CUDA routines.

### `compute_smooth_gradients()`

Computes smooth gradient quantities and stores them in RXMesh-managed attributes.

---

### `compute_gradients(ratioRigidityElasticity, weightRegularity, gstep)`

Combines gradient terms and updates the RXMesh gradient pipeline.

This method expects the following attributes to already exist:

- `v:f3:smooth_grad1`
- `v:f3:smooth_grad2`
- `v:f3:photo_grad`

After running, some temporary attributes are removed.

---

### `split_edge()`

Runs the custom RXMesh edge-splitting routine.

This requires the face attribute:

```text
f:b1:need_subdivide
```

to exist beforehand.

After the split, prefix arrays are rebuilt automatically.

---

### `flip_edge()`

Runs the custom edge-flip routine and then rebuilds prefix arrays.

---

### `compute_face_normal()`

Runs a custom kernel to compute face normals and store them in a face attribute.

---

### `update_vertex_positions()`

Updates vertex positions using the currently stored gradient-related attribute(s).

This method checks that:

```text
v:f3:gradient
```

exists before running.

---

## Example Usage

### Load a Mesh

```python
import rxmesh_torch_ops as rx

wrapper = rx.RXMeshWrapper(
    "input.obj",
    device=0,
    patch_size=256,
    capacity_factor=3.5,
    patch_alloc_factor=5.0,
    lp_hashtable_load_factor=0.5,
)
```

### Export Geometry

```python
vertices = wrapper.copy_vertex_to_tensor()      # [V, 3]
faces = wrapper.copy_face_indices_to_tensor()   # [F, 3]
```

### Create a Face Flag Attribute

```python
import torch

need_subdivide = torch.zeros(
    (wrapper.get_num_faces(), 1),
    dtype=torch.bool,
    device="cuda"
)
need_subdivide[8] = True

wrapper.set_attribute("f:b1:need_subdivide", need_subdivide, force_add=True)
```

### Run Topology Editing

```python
wrapper.split_edge()
wrapper.flip_edge()
```

### Read Back Updated Mesh

```python
vertices = wrapper.copy_vertex_to_tensor()
faces = wrapper.copy_face_indices_to_tensor()
```

---

## Internal Notes

### RXMesh Is Patch-Local, PyTorch Is Flat

RXMesh stores mesh elements in patches and manages ownership / duplication internally. PyTorch works best with flat tensors. The wrapper bridges these two worlds by mapping patch-local handles to global linear ids.

---

### Topology-Changing Operations Require Host Synchronization

After split / flip operations, the wrapper rebuilds prefix arrays on device and synchronizes host-side patch information. This ensures that:

- exported tensors remain valid,
- host-side validation stays consistent,
- later topology queries see the updated mesh state.

---

### Face Indices Are Not Stored as Attributes

Unlike vertex / edge / face attributes, face indices are part of RXMesh topology rather than the generic attribute system. Therefore, they are exported through a dedicated CUDA kernel rather than through the generic attribute dispatch path.

---

## Build Notes

This binding is intended to be built inside the same top-level CMake project as RXMesh.

Typical workflow:

```bash
pip install ./rxmesh_binding -v
```

where `setup.py` triggers the top-level CMake configuration and compiles:

- RXMesh
- RXMesh binding
- custom CUDA kernels

together.

---

## Limitations

Current scope:

- mesh I/O into RXMesh
- tensor export / import
- attribute management
- a set of custom topology and optimization kernels

Current non-goals:

- exposing all RXMesh internals to Python
- replacing RXMesh native APIs
- providing a fully general-purpose mesh processing library independent of RXMesh

This wrapper is intended as a practical bridge between:

- RXMesh topology processing
- PyTorch tensor workflows
- custom CUDA kernels for research and development

---

## Notes for Development

When extending the wrapper:

1. follow the attribute naming convention strictly,
2. rebuild prefix arrays after any topology change,
3. ensure tensors passed into `set_attribute()` are CUDA + contiguous,
4. use `force_add=True` after topology edits if attribute sizes may have changed,
5. keep RXMesh internals hidden behind the wrapper whenever possible.

---
