from firedrake import *
from firedrake import inner as fd_inner, cross as fd_cross, div as fd_div, grad as fd_grad, curl as fd_curl, rot as fd_rot
from solvers import build_linear_solver
import numpy as np



# --- Operators ---

def inner(X, Y):
    """
    Inner product for the mixed domain (scalar + vector).
    X and Y are tuples/lists of (scalar, vector).
    """
    if isinstance(X, tuple) and isinstance(Y, tuple):
        return sum([fd_inner(X_, Y_) if (Y_ != None) and (X_ != None) else 0 for (X_, Y_) in zip(X, Y)])
    else:
        return fd_inner(X, Y)

def cross(X, Y):
    """
    Cross product for 2.5D vectors represented as (scalar, vector).
    X = (Xper, Xpar)
    Y = (Yper, Ypar)
    """
    if isinstance(X, tuple) and isinstance(Y, tuple):
        return (
            X[1][0]*Y[1][1] - X[1][1]*Y[1][0],
            as_vector([
                X[1][1]*Y[0] - X[0]*Y[1][1],
                Y[1][0]*X[0] - Y[0]*X[1][0],
            ])
        )
    else:
        return fd_cross(X, Y)

def grad(X):
    """
    Gradient operator for the potential phi.
    Returns (None, grad(X)).
    """
    return (None, fd_grad(X))

def curl(X, R = None , cylindrical = False):
    """
    Curl operator for the 2.5D formulation.
    Takes in (Perpendicular, Parallel) components.
    Returns (rot(Parallel), curl(Perpendicular)).
    """
    if isinstance(X, tuple) and cylindrical == True:
        return (fd_rot(X[1]), fd_curl(R*X[0]) / R)
    elif isinstance(X, tuple):
        return (fd_rot(X[1]), fd_curl(X[0]))
    else:
        return fd_curl(X)

def div(X, R = None, cylindrical = False):
    """
    Divergence operator for the 2.5D formulation.
    Takes in (Perpendicular, Parallel) components.
    Perp part must be div-free so only use parallel components. 
    Returns div(Parallel).
    """
    if isinstance(X, tuple) and cylindrical == True:
        return fd_div(R*X[1]) / R
    elif isinstance(X, tuple):
        return fd_div(X[1])
    else:
        return fd_div(X)



# --- Spaces ---

def get_spaces(mesh, order):
    """
    Returns the function spaces for the 2.5D formulation.
    Returns: Vg, Vr, Vd, Vn, Vc, Vd_mixed
    """
    Vg_ = FunctionSpace(mesh, "CG", order)
    Vr_ = FunctionSpace(mesh, "N1curl", order)
    Vd_ = FunctionSpace(mesh, "RT", order)
    Vn_ = FunctionSpace(mesh, "DG", order-1)
    
    Vc = (Vg_, Vr_)
    Vd = (Vn_, Vd_)
    
    return Vg_, Vr_, Vd_, Vn_, Vc, Vd



# --- Projection ---

def project_div_free(u_target, mesh, order, cylindrical = False, solver_parameters=None):
    _, _, _, Vn_, _, Vd = get_spaces(mesh, order)

    R = SpatialCoordinate(mesh)[0]
    Z = MixedFunctionSpace([*Vd, Vn_])
    z = Function(Z)
    (uper, upar, p) = split(z)
    (uper_sub, upar_sub, _) = z.subfunctions
    
    (vper, vpar, q) = split(TestFunction(Z))
    u = (uper, upar)
    v = (vper, vpar)

    if cylindrical == True:
        F = (
            # Note: div(...) returns fd_div(R*X)/R. Multiply whole weak form by R for measure.
            inner(u, v) 
        - inner(u_target, v)
        - inner(p, div(v, R, True))
        - inner(div(u, R, True), q)
        ) * R * dx
    else:
        F = (
            inner(u, v) 
        - inner(u_target, v)
        + inner(p, div(v))
        + inner(div(u), q)
        ) * dx


    # enforce zero normal on parallel component RT elements 
    bc_rt = DirichletBC(Z.sub(1), Constant((0.0, 0.0)), "on_boundary")
    bcs = [bc_rt]

    # n = FacetNormal(mesh)

    # # g_expr = Constant((0.0, 0.0))   # or whatever you prescribe on the boundary

    # # flux = assemble(dot(R * g_expr, n) * ds)
    # # print("Weighted boundary flux =", float(flux))
    
    #     # weighted boundary flux (no 2*pi)
    # flux = assemble(dot(R * upar, n) * ds)
    # print("weighted net flux (∫ r B·n ds):", float(flux))

   
        
    solve(F == 0, z, bcs=bcs, solver_parameters=solver_parameters)
    
    test = sqrt(assemble(inner(div(u, R, True), div(u, R, True)) * 2 * np.pi * R * dx))
    print("divergence is ", test)
    Z_out = MixedFunctionSpace([*Vd])
    z_out = Function(Z_out)
    (uper_out_sub, upar_out_sub) = z_out.subfunctions
    uper_out_sub.assign(uper_sub); upar_out_sub.assign(upar_sub)

    return z_out


class HelicitySolver:
    """
    Computes the magnetic helicity H = \ int A . B dx.
    Solves for divergence-free A such that curl(A) = B (up to cohomology in perpendicular component).
    """
    def __init__(self, mesh, order, cylindrical = False, solver_parameters=None):
        Vg_, _, _, _, Vc, _ = get_spaces(mesh, order)
        Z = MixedFunctionSpace([*Vc, Vg_])
        R = SpatialCoordinate(mesh)[0]
        self.z = Function(Z)
        self.cylindrical = cylindrical
        self.R = R
        
        u = TrialFunction(Z)
        (Aper, Apar, phi) = split(u)
        A = (Aper, Apar)

        (Aper_t, Apar_t, phi_t) = split(TestFunction(Z))
        self.A_t = (Aper_t, Apar_t)
        if cylindrical == True:
            self.a = (
                inner(curl(A, R, True), curl(self.A_t, R, True))
            - inner(grad(phi), self.A_t)
            - inner(A, grad(phi_t))
            ) * 2 * np.pi* R * dx
        else:
            self.a = (
                inner(curl(A), curl(self.A_t))
            - inner(grad(phi), self.A_t)
            - inner(A, grad(phi_t))
            ) * dx
    
        self.bcs = [DirichletBC(Z.sub(i), 0, "on_boundary") for i in range(len(Z))]
        if solver_parameters is None:
            self.solver_parameters = {}
        else:
            self.solver_parameters = solver_parameters
    
    def solve(self, B):
        """
        If cylindrical system, this function assumes B has parallel component multiplied by R.
        """
        if self.cylindrical == True:
            L = (
                inner(B, curl(self.A_t, self.R, True ))
            ) * 2 * np.pi * self.R * dx
        else:
            L = (
                inner(B, curl(self.A_t))
            ) * dx

    
        solver = build_linear_solver(self.a, L, self.z, bcs=self.bcs, solver_parameters=self.solver_parameters)
        solver.solve()
        
        (Aper, Apar, _) = split(self.z)
        A = (Aper, Apar)
        
        if self.cylindrical == True:
            return assemble((inner(A, B) - 0.5 * inner(A, curl(A, self.R, True))) * 2 * np.pi * self.R * dx)
        else:
            return assemble((inner(A, B) - 0.5 * inner(A, curl(A))) * dx)
