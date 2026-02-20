# 2.5D magnetic relaxation in cylindrical coordinates (r, phi, z)

"""
This codes uses a 2.5D De Rham complex in cylindrical coordinates that requires scaling
some of the componenents of functions by R. This is done at the start and convertion back happens for 
outputs.
"""

from firedrake import *
from firedrake import as_vector, dot
import os
from firedrake import inner as fd_inner; from firedrake import div as fd_div
from tools import inner, curl, cross, grad, div, get_spaces, project_div_free, HelicitySolver
from solvers import build_nonlinear_solver
from netgen.geom2d import CSG2d, Circle
from ufl import asin, sin, cos, pi, conditional
import csv


def relaxation_pressure(
    order       = 2,
    T           = 1e3 ,
    dt_val      = 0.05,  # If using the golden stepsize, this should be set sufficiently low that a uniform stepsize does not see oscillations
    golden_dt   = False,  # Whether to use the golden stepsize schedule
    dt_danger   = None, #0.1,  # Value above which we use a stronger solver (linesearch)
    dt_max      = None,  # Maximum stepsize
    output_dir  = "output_cylindrical",
    output_freq = 5,
):
    """
    Main simulation function.
    """
    # Create output directory
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        

    # parameters for target ellipse
    R0 = 1.25          # target disk radius (before y-stretch)
    cx, cy = 3.0, 0.0
    dy_stretch = 2.0  # stretch factor in y-direction to make an ellipse   

    mesh = Mesh("tokamak.msh")
    (R, Z) = SpatialCoordinate(mesh)
    # Function spaces
    (Vg_, _, _, _, Vc, Vd) = get_spaces(mesh, order)
    V = MixedFunctionSpace([*Vd, *Vc, *Vc, *Vc, *Vc, Vg_])
  
    # Trial functions
    v = Function(V)
    (Bper, Bpar, jper, jpar, Hper, Hpar, uper, upar, Eper, Epar, p) = split(v)
    B = (Bper, Bpar/R); j = (jper / R, jpar); H = (Hper / R, Hpar); u = (uper / R, upar); E = (Eper / R, Epar)


    (Bper_sub, Bpar_sub, jper_sub, jpar_sub, Hper_sub, Hpar_sub, uper_sub, upar_sub, Eper_sub, Epar_sub, p_sub) = v.subfunctions
    Bper_sub.rename("Magnetic field (perpendicular)")
    Bpar_sub.rename("Magnetic field (parallel) * R")
    jper_sub.rename("Current (perpendicular) * R")
    jpar_sub.rename("Current (parallel)")
    Hper_sub.rename("Auxiliary magnetic field (perpendicular) * R")
    Hpar_sub.rename("Auxiliary magnetic field (parallel)")
    uper_sub.rename("Velocity (perpendicular) * R")
    upar_sub.rename("Velocity (parallel)")
    Eper_sub.rename("Electric field (perpendicular) * R")
    Epar_sub.rename("Electric field (parallel)")
    p_sub.rename("Pressure")

    # Test functions
    (Bper_t, Bpar_t, jper_t, jpar_t, Hper_t, Hpar_t, uper_t, upar_t, Eper_t, Epar_t, p_t) = split(TestFunction(V))
    B_t = (Bper_t, Bpar_t/R); j_t = (jper_t/R, jpar_t); H_t = (Hper_t / R, Hpar_t); u_t = (uper_t / R, upar_t); E_t = (Eper_t / R, Epar_t)

    # Boris & Tom's fun ICs
    helices = (  # (Position, Radius, Strength, Circulation)
    ( (cx-R0/3, cy-R0/3), R0/2, 1, 1),
    ((cx+R0/3, cy + R0/3), R0/2, 1, 1)

    )

    def helix(position, radius, strength, circulation):
        r = sqrt((R - position[0])**2 + (Z - position[1])**2)
        return (
            strength    * conditional(le(r, radius), 1 - (r/radius)**2, 0),
            circulation * conditional(le(r, radius), (1 - (r/radius)**2)**2 * radius / 2, 0)
        )
    Bper_ic = sum([helix(*helices_)[0] for helices_ in helices])
    phi_ic = sum([helix(*helices_)[1] for helices_ in helices])

    Bpar_ic = as_vector([-phi_ic.dx(1) / R, phi_ic.dx(0) / R])

    
    solver_parameters = {
        # SNES (nonlinear) opts
        "snes_type": "newtonls",
        "snes_rtol": 8e-13,
        "snes_atol": 1e-14,
        "snes_max_it": 10,
        # "snes_monitor": None,               # turns on SNES iteration monitoring
        # "snes_converged_reason": None,      # print why SNES stopped
        # preconditioner / direct solver (MUMPS)
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    # Setting up divergence free B_0
    B_prev = project_div_free((Bper_ic, Bpar_ic), mesh, order, cylindrical = True, solver_parameters = solver_parameters)
    (Bper_prev, Bpar_prev) = split(B_prev)
    (Bper_prev_sub, Bpar_prev_sub) = B_prev.subfunctions
    Bper_sub.assign(Bper_prev_sub); Bpar_sub.assign(Bpar_prev_sub)

    # Time variable
    t = Constant(0)
    dt = Constant(dt_val)
    tau = Constant(1)

    # Midpoint/change variables
    B_avg = ((Bper+Bper_prev)/2,  (Bpar + Bpar_prev)/2 / R)
    dB_dt = ((Bper-Bper_prev)/dt, (Bpar-Bpar_prev)/dt /R)

    # Form
    F = (  # dB/dt = - curl E
        inner(dB_dt, B_t)
      + inner(curl(E, R, cylindrical = True), B_t)
    )* R * dx
    F += (  # j = curl(B)
        inner(j, j_t)
      - inner(B_avg, curl(j_t, R, cylindrical = True))
    ) * R * dx
    F += (  # H = B
        inner(H, H_t)
      - inner(B_avg, H_t)
    ) * R * dx
    F += (  # u = j x H - grad p
        inner(u, u_t)
      - tau*inner(cross(j, H), u_t)  #/ scale
      + tau*inner(grad(p), u_t) #/ scale
    ) * R * dx
    F += (  # E = H x u
        inner(E, E_t)
      - inner(cross(H, u), E_t)
    ) * R * dx
    F += (  # div u = 0
        -inner(u, grad(p_t))* R * dx
            )

    # Boundary conditions
    bcs = [DirichletBC(V.sub(i), 0, "on_boundary") for i in range(len(V))]

    # Solver parameters
    sp = {
        # "snes_monitor": None,
        # "snes_converged_reason": None,
        # "ksp_converged_reason": None,
        # "ksp_monitor": None,
    }
    if golden_dt:
        sp_danger = {
            "snes_linesearch_type": "bt",
            "snes_max_it": 200,
            # "snes_atol": 5e-3,
            # "snes_monitor": None,
            # "snes_converged_reason": None,
            # "snes_linesearch_monitor": None,
        }

    # Build solver
    time_stepper = build_nonlinear_solver(F, v, bcs, solver_parameters=sp, options_prefix="time_stepper")
    if golden_dt:
        time_stepper_golden = build_nonlinear_solver(F, v, bcs, solver_parameters=sp|sp_danger, options_prefix="time_stepper")

    # Output files
    Bpar_scaled = Function(Bpar_sub.function_space(), name="Bpar_over_R")
    Bpar_scaled.interpolate(Bpar_sub / R)
    pvd = VTKFile(f"{output_dir}/continuous_output.pvd")
    pvd_2 = VTKFile(f"{output_dir}/discontinuous_output.pvd")
    pvd.write(Bper_sub, Bpar_scaled)

    
    with open("data.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # write header once
        writer.writerow(["iteration","time", "Energy", "Helicity", "force_residual"])

        # Solve loop
        helicity_solver = HelicitySolver(mesh, order, cylindrical = True)
        iteration = 0
        if golden_dt:
            phi_pi = (sqrt(5) - 1) / 2 * pi
            if dt_max is not None:
                dt_val_golden = lambda i : dt_val * dt_max / (dt_val + (dt_max - dt_val) * sin(phi_pi * i)**2)
            else:
                dt_val_golden = lambda i : dt_val / sin(phi_pi * i)**2
        while (float(t) < float(T) + 1.0e-10):
            iteration += 1

            if golden_dt:
                dt.assign(dt_val_golden(iteration))
            if float(t) + float(dt) > float(T):
                dt.assign(T - float(t))
            if float(dt) <= 1e-14:
                break
                
            t.assign(t + dt)

            if mesh.comm.rank == 0:
                print(GREEN % f"Solving for t = {float(t):.4f}, dt = {float(dt):.4f}, iteration = {iteration}...", flush=True)

            if dt_danger is not None and float(dt) >= dt_danger:
                print(RED % "Large dt! Using strict solver preferences...")
                time_stepper_golden.solve()
            else:
                time_stepper.solve()
            
            # Compute metrics
            energy = assemble(0.5 * inner(B, B) * 2 * np.pi * R * dx)
            helicity = helicity_solver.solve(B) # helicity requires 25D De Rham in cylindrical coords
            u_norm = assemble(inner(u, u) * 2 * np.pi * R * dx)
            div_B_norm = sqrt(assemble(inner(div(B, R, True), div(B, R, True)) * 2 * np.pi * R * dx)) 

            # constructing force residual
            temp = cross(j,B)[0]
            temp2 = cross(j,B)[1] - grad(p)[1]
            num = assemble((temp*temp + dot(temp2, temp2))*R*dx)
            den = assemble(inner(grad(p)[1],grad(p)[1]) * R * dx)
            if den < 1e-14:
                raise RuntimeError("Denominator (||grad p||^2) is (near) zero.")
            force_residual = np.sqrt(num/den)   # relative L2 norm

            if mesh.comm.rank == 0:
                print(BLUE % f"  Energy:    {energy:.6e}")
                print(BLUE % f"  Helicity:  {helicity:.6e}")
                print(BLUE % f"  |u|^2:     {u_norm:.6e}")
                print(BLUE % f"  |div B|: {div_B_norm:.6e}")
                print(BLUE % f" ||jxB - grad(p)||/||grad p||: {force_residual:.6e}")


            # Update previous step
            Bper_prev_sub.assign(Bper_sub)
            Bpar_prev_sub.assign(Bpar_sub)

            # Output
            if iteration % output_freq == 0:
                Bpar_scaled.interpolate(Bpar_sub / R)
                pvd.write(Bper_sub, Bpar_scaled)
                pvd_2.write(jper_sub, jpar_sub, Hper_sub, Hpar_sub, uper_sub, upar_sub, Eper_sub, Epar_sub, p_sub)
                writer.writerow([iteration, float(t), energy, helicity, force_residual])



if __name__ == "__main__":
        relaxation_pressure()
