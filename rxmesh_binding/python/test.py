import time

import torch
import rxmesh_torch_ops as rx
import trimesh
import numpy as np
import os
def save_ply_trimesh(path: str, vertices: torch.Tensor, faces: torch.Tensor,
                     vertex_colors: torch.Tensor | None = None,
                     binary: bool = True):
    v = vertices.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    kwargs = {}
    if vertex_colors is not None:
        c = vertex_colors.detach().cpu().numpy()
        if c.dtype != np.uint8:
            if c.max() <= 1.0:
                c = (c * 255.0).clip(0, 255).astype(np.uint8)
            else:
                c = c.clip(0, 255).astype(np.uint8)
        kwargs["vertex_colors"] = c
    m = trimesh.Trimesh(vertices=v, faces=f, process=False, **kwargs)
    m.export(path, file_type="ply", encoding="binary_little_endian" if binary else "ascii")
name="curved_20x20"
# name="curved_8x8"
# name="curved_5x5"
# name="curved_4x4"
# name="torus"
# name = "fuse_post_2dgs_dec_clean_cut"
# name = "scan_24"
# root_dir = "/data1/zzy/public_data/TNT/TNT_GOF/TrainingSet/Truck/exp/meshSplats/train/ours_10000"
root_dir = "/home/zzy/engineer/rxmesh_wsl/rxmesh/input"


# mesh_path = os.path.join(root_dir, f"{name}.ply")
# m = trimesh.load(mesh_path, process=True)
# m.update_faces(m.nondegenerate_faces())
# m.update_faces(m.unique_faces())
# m.remove_unreferenced_vertices()
# m.remove_infinite_values()
# m.merge_vertices()
# clean_name = f"{name}_clean"
# mesh_path = os.path.join(root_dir, f"{clean_name}.obj")
# a=m.export(mesh_path)



# wrapper.remesh()
# wrapper = rx.RXMeshWrapper(f"/home/zzy/engineer/rxmesh_wsl/rxmesh/input/{name}.obj")
# mesh_path = os.path.join(root_dir, "fuse_post_2dgs_dec_clean_clean.obj")
mesh_path = os.path.join(root_dir, name+".obj")
wrapper = rx.RXMeshWrapper(mesh_path, 0, "", patch_size=256, capacity_factor=4.0, patch_alloc_factor=3.0, lp_hashtable_load_factor=0.5)
vertices = wrapper.copy_vertex_to_tensor()
faces = wrapper.copy_face_indices_to_tensor()
need_subdivide = torch.zeros((faces.shape[0], 1), dtype=torch.bool,device=vertices.device)
# mask = torch.rand(faces.shape[0], device=vertices.device) < 0.5
fid=8
need_subdivide[:,] = True 
# print(vertices[faces[fid][0]], vertices[faces[fid][1]], vertices[faces[fid][2]])
wrapper.set_attribute("f:b1:need_subdivide", need_subdivide, True)
wrapper.split_edge()
edge_status = wrapper.get_attribute("e:c1:status")
# print(edge_status)
wrapper.remove_attribute("f:b1:need_subdivide", False)
print("split done")

e_flip = wrapper.get_attribute("e:b1:flip")
print(e_flip.sum())
# edge_status = wrapper.get_attribute("e:c1:status")
vertices = wrapper.copy_vertex_to_tensor()
faces = wrapper.copy_face_indices_to_tensor()
save_ply_trimesh(f"/home/zzy/engineer/rxmesh_wsl/rxmesh/output/{name}_split.ply", vertices, faces, binary=False)
print(faces.shape)

# wrapper.flip_edge()
# print("flip done")
# vertices = wrapper.copy_vertex_to_tensor()
# faces = wrapper.copy_face_indices_to_tensor()
# save_ply_trimesh(f"/home/zzy/engineer/rxmesh_wsl/rxmesh/output/{name}_split_flip.ply", vertices, faces)

faces = wrapper.copy_face_indices_to_tensor()
edge_set = set()
for f in faces:
    edge_set.add(tuple(sorted((f[0].item(), f[1].item()))))
    edge_set.add(tuple(sorted((f[1].item(), f[2].item()))))
    edge_set.add(tuple(sorted((f[2].item(), f[0].item()))))
print(f"number of edges: {len(edge_set)}")



