from firedrake import LinearVariationalProblem, LinearVariationalSolver, NonlinearVariationalProblem, NonlinearVariationalSolver

def build_linear_solver(a, L, u_sol, bcs, aP=None, solver_parameters=None, options_prefix=None):
    """
    Helper to build a LinearVariationalSolver.
    """
    return LinearVariationalSolver(
        LinearVariationalProblem(a, L, u_sol, bcs=bcs, aP=aP),
        solver_parameters=solver_parameters,
        options_prefix=options_prefix
    )

def build_nonlinear_solver(F, z, bcs, Jp=None, solver_parameters=None, options_prefix=None):
    """
    Helper to build a NonlinearVariationalSolver.
    """
    return NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, z, bcs, Jp=Jp),
        solver_parameters=solver_parameters,
        options_prefix=options_prefix
    )
