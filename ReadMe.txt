22/01/26 - Thomas Higham and Boris Andrews

This code implements a 2.5D modified magneto-frictional system in Firedrake which provides non-force-free 
equilibria of the system. relaxation.py implements in Cartesian coordinates (x,y,z), assuming independence of the z coordinate. 
relaxation_cylindrical.py implements in cylindrical coordinates (R,phi,Z) assuming independence in phi coordinate.

These schemes assume axisymmetry and break the vector fields into perpendicular and parallel components. We then choose finite element spaces based on a modified 2.5D De Rham complex so that we can preserve helicty and divergence of B (magnetic field), as well as ensure energy dissipates as we relax.

This uses concepts from the paper "Helicity-preserving finite element discretization for magnetic
relaxation" by Mingdong He, Patrick E. Farrell, Kaibo Hu, and Boris D. Andrews, 2025.

-----------------------------------------------------------------------------------------------------
The main code is found within code/two_five_d

relaxation.py is the Cartesian implementation 
relaxation_cylindrical.py is the cylindrical implementation [Fixed and working]
solvers.py contains linear and nonlinear solver functions
tools.py contains definitions of the differential operators, the spaces we want, projection to divergence free, and a helicity calculation.
