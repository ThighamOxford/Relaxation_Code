# make_tokamak_mesh.py
import numpy as np
import pygmsh
import meshio

# ---- geometry parameters ----
R0 = 3.0       # major radius (m)
a  = 1.25      # minor radius (m)
kappa = 2      # elongation
delta = 0.3    # triangularity
n_boundary = 100   # number of boundary sample points
mesh_size = 0.01 * R0 * kappa   # target element size

# ---- parametric boundary points ----
theta = np.linspace(0.0, 2*np.pi, n_boundary, endpoint=False)
R = R0 + a * np.cos(theta + np.arcsin(delta)*np.sin(theta))
Z = kappa * a * np.sin(theta)
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

    # generate mesh (meshio.Mesh)
    msh = geom.generate_mesh(dim=2)

# ----------------------------
# Post-process meshio.Mesh to add physical tags
# ----------------------------

# Choose integer IDs for physical groups (non-zero)
PHYS_BOUNDARY = 1  # for line facets
PHYS_PLASMA = 2    # for 2D domain (triangles)

# Ensure msh.field_data exists and assign names -> id mapping
# meshio expects arrays (commonly np.array([id])) for field_data values.
msh.field_data = {
    "boundary": np.array([PHYS_BOUNDARY]),
    "plasma": np.array([PHYS_PLASMA])
}

# Build gmsh:physical arrays for each cell block in the same order as msh.cells.
phys_lists = []
for cell_block in msh.cells:
    ctype = cell_block.type.lower()
    n_elems = len(cell_block.data)
    if ctype in ("line", "line3"):   # 1D facets
        phys_lists.append(np.full(n_elems, PHYS_BOUNDARY, dtype=int))
    elif ctype in ("triangle", "tri", "triangle6"):
        phys_lists.append(np.full(n_elems, PHYS_PLASMA, dtype=int))
    else:
        # For any other cell types (vertices, quads, etc), set to 0 or a chosen id.
        # Vertex blocks sometimes appear — we set them to PHYS_BOUNDARY to be safe for point markers,
        # or choose zero if you prefer no tag:
        if ctype == "vertex":
            phys_lists.append(np.full(n_elems, PHYS_BOUNDARY, dtype=int))
        else:
            phys_lists.append(np.zeros(n_elems, dtype=int))

# Attach to msh.cell_data using the gmsh key expected by meshio
msh.cell_data = {"gmsh:physical": phys_lists}

# Optionally attach geometrical tags too (meshio sometimes expects gmsh:geometrical)
# We'll set them to zero (or you can supply meaningful ids similarly).
geom_lists = [np.zeros(len(cb.data), dtype=int) for cb in msh.cells]
msh.cell_data["gmsh:geometrical"] = geom_lists

# Write out gmsh v2 format (safer compatibility with many Firedrake installs)
meshio.write("tokamak_coarse.msh", msh, file_format="gmsh22")
print("Wrote tokamak.msh (with physical tags)")

# ----------------------------
# Quick verification
# ----------------------------
m = meshio.read("tokamak_coarse.msh")
print("field_data:", m.field_data)
if "gmsh:physical" in m.cell_data:
    for cblock, phys in zip(m.cells, m.cell_data["gmsh:physical"]):
        ctype = cblock.type
        u = np.unique(phys)
        print(f"cell type: {ctype:8s}  n elems: {len(cblock.data):5d}  unique physical ids: {u}")
else:
    print("No 'gmsh:physical' found in cell_data.")
