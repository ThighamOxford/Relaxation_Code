# make_tokamak_mesh.py
import numpy as np
import pygmsh
import meshio

# ---- geometry parameters ----
R0 = 3.0       # major radius (m)
a  = 1.25       # minor radius (m)
kappa = 2    # elongation
delta = 0.3    # triangularity
n_boundary = 200   # number of boundary sample points (smooth curve)
mesh_size = 0.006*R0*kappa   # target element size (tune as needed)

# ---- parametric boundary points ----
theta = np.linspace(0.0, 2*np.pi, n_boundary, endpoint=False)
R = R0 + a * np.cos(theta + np.arcsin(delta)*np.sin(theta))
Z = kappa * a * np.sin(theta)

# create closed list of points in (R,Z) plane for 2D mesh
boundary_pts = np.column_stack([R, Z])

# ---- build geometry with pygmsh ----
with pygmsh.geo.Geometry() as geom:
    # create points in the R-Z plane (z=0)
    pts = [geom.add_point([float(r), float(z), 0.0], mesh_size=mesh_size) 
           for r, z in boundary_pts]
    # create lines between successive points and close loop
    lines = []
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i+1) % len(pts)]
        lines.append(geom.add_line(p1, p2))
    loop = geom.add_curve_loop(lines)
    surface = geom.add_plane_surface(loop)

    # optional: add a refinement field closer to the inner/outboard midplane etc.
    # generate mesh and write as .msh
    msh = geom.generate_mesh(dim=2)
    # msh is the meshio.Mesh returned by pygmsh
    try:
        # prefer gmsh v4 format (modern)
        meshio.write("tokamakfine.msh", msh, file_format="gmsh4")
    except Exception:
        # fallback to legacy Gmsh v2 which Firedrake often wants
        meshio.write("tokamakfine.msh", msh, file_format="gmsh22")
print("Wrote tokamakfine.msh")

