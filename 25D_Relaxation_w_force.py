# avfet method for relaxation 
from firedrake import *
import csv
import os
import sys

# parameters 
output = True # whether we save to file
Lx, Ly = 8, 8
Nx, Ny = 16, 16
order = 2  # polynomial degree
t = Constant(0)
dt = Constant(1)
T = 5e3 # final time

mesh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=False) # quad argument is for the mesh type
mesh.coordinates.dat.data[:, 0] -= Lx/2
mesh.coordinates.dat.data[:, 1] -= Ly/2

# Function spaces - underscore when space is not a composition
Vg_ = FunctionSpace(mesh, "CG", order)
Vr_ = FunctionSpace(mesh, "N1curl", order)
Vd_ = FunctionSpace(mesh, "RT", order)
Vn_ = FunctionSpace(mesh, "DG", order-1)

# Set up composition of function spaces for middle of De Rham complex.
Vc = [Vg_, Vr_]
Vd = [Vn_, Vd_]

# Mixed unknowns: [B, j, H, u, E, p]
Z = MixedFunctionSpace([*Vd, *Vc, *Vc, *Vc, *Vc, Vg_]) # *unpacks the vectors
z = Function(Z)

# Split into perpendicular and parallel components
(Bper, Bpar, jper, jpar, Hper, Hpar, uper, upar, Eper, Epar, p) = split(z)
B = (Bper, Bpar)
j = (jper, jpar)
H = (Hper, Hpar)
u = (uper, upar)
E = (Eper, Epar)

# subfunctions let us access the per and par components separately. We use at end of timestepping. 
(Bper_sub, Bpar_sub) = z.subfunctions[0:2]
p_sub = z.subfunctions[-1]
p_sub.rename("Pressure")

# Test functions
(Bper_t, Bpar_t, jper_t, jpar_t, Hper_t, Hpar_t, uper_t, upar_t, Eper_t, Epar_t, p_t) = split(TestFunction(Z))
B_t = (Bper_t, Bpar_t)
j_t = (jper_t, jpar_t)
H_t = (Hper_t, Hpar_t)
u_t = (uper_t, upar_t)
E_t = (Eper_t, Epar_t)

# Tracking function
V_prev = MixedFunctionSpace([*Vd])
z_prev = Function(V_prev)
(Bper_prev, Bpar_prev) = split(z_prev)
B_prev = (Bper_prev, Bpar_prev)

def inner_(X_, Y_):
    return inner(X_[0], Y_[0]) + inner(X_[1], Y_[1])
def curl_(X_):
    """
    Function that takes in a list of scalar + vec corresponding to perp component + parallel component.
    Returns the curl of the perp component, the rot of the parallel component.
    Output: tuple of scalar + 2D vector
    """
    return (curl(X_[1]), rot(X_[0]))
def cross_(X_, Y_):
    """
    Cross product of objects that are lists of scalar + vec. Follows from 3D cross product.
    Output: tuple of scalar + 2D vector.
    """
    return (
        X_[1][0]*Y_[1][1] - X_[1][1]*Y_[1][0],
        as_vector([
            X_[1][1]*Y_[0] - X_[0]*Y_[1][1],
            Y_[1][0]*X_[0] - Y_[0]*X_[1][0],
        ])
        )

# Value of B at midpoint needed for terms with other variables as they have to be taken
# at midpoint (crucially not the point of discontinuity at integer time!)
B_avg = ((Bper+Bper_prev)/2,  (Bpar+Bpar_prev)/2)
# Implicit midpoint scheme
dB_dt = ((Bper-Bper_prev)/dt, (Bpar-Bpar_prev)/dt)


# Residual
F = (
      inner_(dB_dt, B_t)
    + inner_(curl_(E), B_t)
    + inner_(j, j_t)
    - inner_(B_avg, curl_(j_t))
    + inner_(H, H_t)
    - inner_(B_avg, H_t)
    + inner_(u, u_t)
    - inner_(cross_(j, H), u_t)
    + inner(grad(p), upar_t)
    + inner_(E, E_t)
    + inner_(cross_(u, H), E_t)
    + inner(upar, grad(p_t))
    ) * dx

# Boundary conditions
bcs = [DirichletBC(Z.sub(index), 0, "on_boundary") for index in range(len(Z))]

# Might want to use later as currently sp is first called below in timestepper
lu = {
	# "mat_type": "aij",
	# "snes_type": "newtonls",
        "snes_monitor": None,
        "ksp_monitor": None,
    #     "ksp_type": "preonly",
	# "pc_type": "lu",
    #     "pc_factor_mat_solver_type": "mumps"
}
sp = lu

# Initial conditions
(X0, Y0) = SpatialCoordinate(mesh)

# Boris & Tom's fun ICs
Bper_ic = (Lx**2 - 4*X0**2) * (Ly**2 - 4*Y0**2) / 1e3
# Take the rot for the parallel part
Bpar_ic = as_vector([
    - X0*(Lx**2 - 4*X0**2) * 8*Y0 / 1e3,
    (Lx - 12*X0**2) * (Ly**2 - 4*Y0**2) / 1e3
])

(Bper_prev_sub, Bpar_prev_sub) = z_prev.subfunctions
Bper_prev_sub.rename("MagneticFieldPerpendicular")
Bpar_prev_sub.rename("MagneticFieldParallel")

def project_initial_conditions(Bpar_ic_):
    # Need to project the initial conditions
    # such that div(B) = 0 and B·n = 0
    # perpendicular component is automatically div free.

    Z_init = MixedFunctionSpace([Vd_, Vn_])
    z_init = Function(Z_init)
    (Bpar_init, p_init) = split(z_init)
    (Bpar_init_t, p_init_t) = split(TestFunction(Z_init))

    F_init = (
          inner(Bpar_init - Bpar_ic_, Bpar_init_t)
        - inner(p_init, div(Bpar_init_t))
        - inner(div(Bpar_init), p_init_t)
        ) * dx

    bcs_init = [DirichletBC(Z_init.sub(0), 0, "on_boundary")]

    spp = {
        # "mat_type": "nest",
        # "snes_type": "ksponly",
        # "snes_monitor": None,
        # "ksp_monitor": None,
        # "ksp_max_it": 1000,
        # "ksp_norm_type": "preconditioned",
        # "ksp_type": "minres",
        # "pc_type": "fieldsplit",
        # "pc_fieldsplit_type": "additive",
        # "fieldsplit_pc_type": "cholesky",
        # "fieldsplit_pc_factor_mat_solver_type": "mumps",
        # "ksp_atol": 1.0e-5,
        # "ksp_rtol": 1.0e-5,
        # "ksp_minres_nutol": 1E-8,
        # "ksp_convergence_test": "skip",
    }
    solve(F_init == 0, z_init, bcs_init, solver_parameters=spp)
    return z_init.subfunctions[0]

# In above function we project into divergence free space (RT) for the parallel component. Here we do explicitly for perp.
Bper_prev_sub.project(Bper_ic)
Bpar_prev_sub.assign(project_initial_conditions(Bpar_ic))

def build_linear_solver(a, L, u_sol, bcs, aP=None, solver_parameters = None, options_prefix=None):
    problem = LinearVariationalProblem(a, L, u_sol, bcs=bcs, aP=aP)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=solver_parameters,
                                     options_prefix=options_prefix)
    return solver

def build_nonlinear_solver(F, z, bcs, Jp=None, solver_parameters = None, options_prefix=None):
    problem = NonlinearVariationalProblem(F, z, bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(problem,
                solver_parameters=solver_parameters,
                options_prefix=options_prefix)
    return solver

# def helicity_solver():
#     # Spaces for magnetic potential computation
#     # If using periodic boundary conditions, we need to modify
#     # this to account for the harmonic form [0, 0, 1]^T
#     # using Yang's solver

#     u = TrialFunction(Vc)
#     v = TestFunction(Vc)
#     u_sol = Function(Vc)

#     # weak form of curl-curl problem 
#     a = inner(curl(u), curl(v)) * dx
#     L = inner(B_, curl(v)) * dx
#     beta = Constant(0.1)
#     Jp_curl = a + inner(beta * u, v) * dx
#     bcs_curl = [DirichletBC(Vc, 0, subdomain) for subdomain in dirichlet_ids]
#     rtol = 1E-8
#     preconditioner = True
#     if preconditioner:
#         pc_type = "lu"
#     else:
#         pc_type = "none"
#     sparams = {
#         "snes_type": "ksponly",
#         # "ksp_type": "lsqr",
#         "ksp_type": "minres",
#         "ksp_max_it": 1000,
#         "ksp_convergence_test": "skip",
#         #"ksp_monitor": None,
#         "pc_type": pc_type,
#         "ksp_norm_type": "preconditioned",
#         "ksp_minres_nutol": 1E-8,
#         }

#     solver = build_linear_solver(a, L, u_sol, bcs_curl, Jp_curl, sparams, options_prefix="helicity")
#     return solver

# helicity_solver = helicity_solver()

# def riesz_map(functional):
#     function = Function(functional.function_space().dual())
#     with functional.dat.vec as x, function.dat.vec as y:
#         helicity_solver.snes.ksp.pc.apply(x, y)
#     return function

# def compute_helicity_energy(B):
#     helicity_solver.solve()
#     problem = helicity_solver._problem
#     if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
#         # lifting strategy
#         r = assemble(problem.F, bcs=problem.bcs)
#         rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
#         rstar.rename("RHS")
#         # lft = uh - inner(r, uh)/inner(r, rstar) * rstar
#         c = assemble(action(r, problem.u)) / assemble(action(r, rstar))
#         ulft = Function(Vc, name="u_lifted")
#         ulft.assign(problem.u - c * rstar)
#         A = ulft
#     else:
#         A = problem.u
#     diff = norm(curl(A) - B, "L2")
#     if mesh.comm.rank == 0:
#         print(f"magnetic potential: ||curl(A) - B||_L2 = {diff:.8e}", flush=True)
#     A_ = Function(Vc, name="MagneticPotential")
#     A_.project(A)
#     curlA = Function(Vd, name="CurlA")
#     curlA.project(curl(A))
#     diff_ = Function(Vd, name="CurlAMinusB")
#     diff_.project(B-curlA)
#     #VTKFile("output/magnetic_potential.pvd").write(curlA, diff_, A_)
#     if bc=="closed":
#         return assemble(inner(A, B)*dx), diff, diff_, assemble(inner(B, B) * dx)
#     else: 
#         return assemble(inner(A, B + diff_)*dx), diff, diff_, assemble(inner(B - diff_, B-diff_) * dx)
       
# def compute_Bn(B):
#     n = FacetNormal(mesh)
#     return assemble(inner(dot(B, n), dot(B, n))*ds_v)

# def compute_divB(B):
#     return norm(div(B), "L2")

# # monitor of (non)linear force-free field
# def compute_lamb(j, B):
#     eps = 1e-10
#     lamb = Function(Vg_).interpolate(dot(j, B)/(dot(B, B) + eps))
#     with lamb.dat.vec_ro as v:
#         _, max_val = v.max()
#         _, min_val = v.min()
#     if abs(min_val) < eps:
#         return eps
#     else:
#         return max_val/min_val

# # monitor of force-free
# def compute_xi_max(j, B):
#     eps = 1e-10
#     xi = Function(Vg).interpolate(cross(j, B)/(dot(B, B) + eps))
#     with xi.dat.vec_ro as v:
#         _, max_val = v.max()
#         _, min_val = v.min()
    
#     if abs(min_val) < eps:
#         return eps
#     else:
#         return max_val/min_val

# # define files
# data_filename = "output/data.csv"
# fieldnames = ["t", "helicity", "energy", "divB", "lamb", "xi"]
# if mesh.comm.rank == 0:
#     with open(data_filename, "w") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

# # store the initial value
# helicity, diff, diff_, energy = compute_helicity_energy(z.sub(0))
# divB = compute_divB(z.sub(0))
# lamb = compute_lamb(z.sub(1), z.sub(0))
# xi = compute_xi_max(z.sub(1), z.sub(0))

# if mesh.comm.rank == 0:
#     row = {
#         "t": float(t),
#         "helicity": float(helicity),
#         "energy": float(energy),
#         "divB": float(divB),
#         "xi": float(xi),
#     }
#     with open(data_filename, "a", newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writerow(row)
#         print(f"{row}")

def Bdotnorm(Bdot):
    """
    Define a norm on Bdot - used to check when full relaxation has occured.
    per is in L2 space, par in H(div)
    Output: a sensible norm on these two spaces combined
    """
    return sqrt(assemble(inner_(Bdot, Bdot) * dx))

# solver
time_stepper = build_nonlinear_solver(F, z, bcs, solver_parameters= sp, options_prefix="time_stepper")

if output:
    pvd = VTKFile("output/two_helicies/continuous_output.pvd")
    pvd_2 = VTKFile("output/two_helicies/discontinuous_output.pvd")
    pvd.write(Bper_prev_sub, Bpar_prev_sub)


timestep = 0
#E_old = compute_energy(z_prev.sub(0), diff_)

while (float(t) < float(T) + 1.0e-10):
    if float(t) + float(dt) > float(T):
        dt.assign(T - float(t))
    if float(dt) <=1e-14:
        break
    t.assign(t + dt)
    if mesh.comm.rank == 0:
        print(RED % f"Solving for t = {float(t)}, dt={float(dt)}, T={T}", flush=True)
    
    time_stepper.solve()
    
    print(GREEN % f"Energy: {assemble(1/2 * inner_(B, B) * dx)}")
    print(BLUE % f"Bdot norm: {Bdotnorm(dB_dt)}")

    # # monitor
    # helicity, diff, diff_, energy= compute_helicity_energy(z.sub(0))
    # divB = compute_divB(z.sub(0))
    # lamb = compute_lamb(z.sub(1), z.sub(0))
    # xi = compute_xi_max(z.sub(1), z.sub(0))
    
    # if mesh.comm.rank == 0:
    #     row = {
    #         "t": float(t),
    #         "helicity": float(helicity),
    #         "energy": float(energy),
    #         "divB": float(divB),
    #         "lamb": float(lamb),
    #         "xi": float(xi),
    #     }
    #     with open(data_filename, "a", newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writerow(row)
    #         print(f"{row}")

    Bper_prev_sub.assign(Bper_sub)
    Bpar_prev_sub.assign(Bpar_sub)

    if output:
        #if timestep % 10 == 0:
        pvd.write(Bper_prev_sub, Bpar_prev_sub)
        pvd_2.write(p_sub)
    timestep += 1
