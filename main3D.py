import time
start = time.perf_counter()

import csv
input = "parameters.csv"
params = []
with open(input,newline="") as file:
  csvFile = csv.reader(file)
  for row in csvFile:
    if not row:
      continue
    params.append(row)
def RemoveAfterHash(s: str) -> str:
  idx = s.find("#")
  if idx != -1:
    return s[:idx]
  return s
for i in range(len(params)):
  params[i][0] = RemoveAfterHash("".join(params[i][0].split()))
  params[i][1] = RemoveAfterHash("".join(params[i][1].split()))
def GetValue(arr: list[str], key: str, vtype: str) -> (int | float | str):
  value = None
  for row in arr:
    if row[0] == key:
      value = row[1]
      break
  if value is None:
    print(f"The keyword \"{key}\" is NOT set!")
    exit()
  if vtype == "int":
    return int(value)
  elif vtype == "float":
    return float(value)
  elif vtype == "str":
    return str(value)
  else:
    print("Undefined variable type!")
    exit()

from mpi4py import MPI
import dolfinx.mesh
import numpy as np

width, height = 1, 15 # [m]
nx, ny = 5, 75
mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD,
                               np.array([[0.0, 0.0, 0.0], [width, width, height]]),
                               [nx, nx, ny],
                               cell_type=dolfinx.mesh.CellType.hexahedron)
fdim = mesh.topology.dim - 1
tdim = mesh.topology.dim
mesh.topology.create_connectivity(fdim,tdim) # estsablish surface (fdim) -> cell (tdim) connection

bd = {
  "bottom" : 1,
  "front"  : 2,
  "right"  : 3,
  "back"   : 4,
  "left"   : 5,
  "top"    : 6
}
boundaries = [(bd["bottom"], lambda x : np.isclose(x[2], 0.0)),
              (bd["front"],  lambda x : np.isclose(x[1], 0.0)),
              (bd["right"],  lambda x : np.isclose(x[0], width)),
              (bd["back"],   lambda x : np.isclose(x[1], width)),
              (bd["left"],   lambda x : np.isclose(x[0], 0.0)),
              (bd["top"],    lambda x : np.isclose(x[2], height))]
def GetFacets() -> dolfinx.mesh.MeshTags:
  indices, markers = [], []
  for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities(mesh,fdim,locator)
    indices.append(facets)
    markers.append(np.full_like(facets, marker))
  indices = np.hstack(indices).astype(np.int32)
  markers = np.hstack(markers).astype(np.int32)
  indices_sorted = np.argsort(indices)
  return dolfinx.mesh.meshtags(mesh,fdim,indices[indices_sorted],markers[indices_sorted])
facets = GetFacets()

import basix.ufl
import dolfinx.fem
from petsc4py import PETSc

# Taylor-Hood element
Q2 = basix.ufl.element("CG", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), dtype=dolfinx.default_real_type)
Q1 = basix.ufl.element("CG", mesh.basix_cell(), 1, dtype=dolfinx.default_real_type)
MixedSpace = dolfinx.fem.functionspace(mesh,basix.ufl.mixed_element([Q2,Q1]))

X = dolfinx.fem.Function(MixedSpace)
X_prev = dolfinx.fem.Function(MixedSpace)

# functionspace for u
V, dofV = MixedSpace.sub(0).collapse()
U = dolfinx.fem.Function(V, name="displacement")
with U.vector.localForm() as local:
  local.set(PETSc.ScalarType(0.0))
X_prev.x.array[dofV] = U.x.array
X_prev.x.scatter_forward()

# functionspace for p
Q, dofQ = MixedSpace.sub(1).collapse()
P = dolfinx.fem.Function(Q, name="pressure")
with P.vector.localForm() as local:
  local.set(PETSc.ScalarType(GetValue(params,"P_init","float")))
X_prev.x.array[dofQ] = P.x.array
X_prev.x.scatter_forward()

import ufl
from ufl import dx, diff, inner, dot, grad, Identity, tr, det, ln, nabla_div, derivative, sym

def CauchyStress_LinearElasticBody(u,Mu,Lambda) :
  return 2*Mu*sym(grad(u)) + Lambda*tr(sym(grad(u)))*Identity(len(u))
def DilationalStrain(u) :
  F = ufl.variable(Identity(len(u)) + grad(u))
  J = det(F)
  return J - 1

u, p = ufl.split(X)
u_prev, p_prev = ufl.split(X_prev)
wu, wp = ufl.TestFunctions(MixedSpace)

theta = 1.0 # backward Euler
# theta = 0.5 # Crank-Nicolson
p_theta = p*theta + p_prev*(1-theta)

metadata = {} # {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facets)
# dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
# ds = ufl.Measure("ds", domain=mesh, subdomain_data=facets, metadata=metadata)
nVec = ufl.FacetNormal(mesh)

dt     = dolfinx.fem.Constant(mesh, PETSc.ScalarType(GetValue(params,"dt","float")))
k      = dolfinx.fem.Constant(mesh, PETSc.ScalarType(GetValue(params,"Permeability","float")))
E      = dolfinx.fem.Constant(mesh, PETSc.ScalarType(GetValue(params,"YoungModulus","float")))
nu     = dolfinx.fem.Constant(mesh, PETSc.ScalarType(GetValue(params,"PoissonRatio","float")))
alpha  = dolfinx.fem.Constant(mesh, PETSc.ScalarType(GetValue(params,"BiotWillisCoeff","float")))
S      = dolfinx.fem.Constant(mesh, PETSc.ScalarType(GetValue(params,"SpecificStorage","float")))

Mu     = dolfinx.fem.Constant(mesh, PETSc.ScalarType(E.value/(2*(1+nu.value))))
Lambda = dolfinx.fem.Constant(mesh, PETSc.ScalarType(E.value*nu.value/((1+nu.value)*(1-2*nu.value))))

sigma = CauchyStress_LinearElasticBody(u,Mu,Lambda)
epsilon = DilationalStrain(u)
epsilon_dot = DilationalStrain((u-u_prev)/dt)
traction = dolfinx.fem.Constant(mesh,PETSc.ScalarType(GetValue(params,"traction","float")))
flux = dolfinx.fem.Constant(mesh,PETSc.ScalarType(GetValue(params,"NeumannFlux","float")))
# source = dolfinx.fem.Constant(mesh,PETSc.ScalarType(0.0)) # 湧き出しなし．

# Solid equilibrium
WeakForm  =  inner(grad(wu),sigma)*dx              \
           + (epsilon - p)*nabla_div(wu)*dx        \
           - traction*inner(wu,nVec)*ds(bd["top"])

# Mass conservation
WeakForm +=  S*wp*((p-p_prev)/dt)*dx          \
           + k*dot(grad(p_theta),grad(wp))*dx \
           + alpha*wp*epsilon_dot*dx          \
           - wp*flux*ds(bd["top"])
          #  - wp*source*dx               \

bcs = []
# uz = 0 on bottom
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(2),fdim,facets.find(bd["bottom"]))
bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(2)))
# ux = uy = 0 on side
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(1),fdim,facets.find(bd["front"]))
bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(1)))
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(0),fdim,facets.find(bd["right"]))
bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(0)))
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(1),fdim,facets.find(bd["back"]))
bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(1)))
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(0),fdim,facets.find(bd["left"]))
bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(0)))
# p = 0 (permeable) on top
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(1),fdim,facets.find(bd["top"]))
bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(1)))

from dolfinx.io import VTKFile
import dolfinx.fem.petsc
import dolfinx.nls.petsc

dX = ufl.TrialFunction(MixedSpace)
Jacobian = derivative(WeakForm,X,dX)
problem = dolfinx.fem.petsc.NonlinearProblem(WeakForm,X,bcs=bcs,J=Jacobian)
solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm,problem)
solver.rtol = GetValue(params,"NRrtol","float")
solver.atol = GetValue(params,"NRatol","float")
solver.relaxation_parameter = GetValue(params,"NRrelaxation","float")
solver.max_it = GetValue(params,"NRmax_iter","int")
solver.convergence_criterion = GetValue(params,"NRcriterion","str")

ofreq = GetValue(params,"OutputFreq","int")
ofile1 = VTKFile(mesh.comm,GetValue(params,"OutputPath","str")+"/3D/u.pvd","w")
ofile2 = VTKFile(mesh.comm,GetValue(params,"OutputPath","str")+"/3D/p.pvd","w")
ofile1.write_mesh(mesh)
ofile2.write_mesh(mesh)

t = 0.0
dt = dt.value
for i in range(GetValue(params,"NumOfTimeStep","int")):
  t += dt

  NRiter, IsConverged = solver.solve(X)
  X.x.scatter_forward()

  # output
  if i % ofreq == 0:
    U.x.array[:] = X.x.array[dofV]
    P.x.array[:] = X.x.array[dofQ]
    ofile1.write_function(U,t)
    ofile2.write_function(P,t)

  # update
  X_prev.x.array[:] = X.x.array
  X_prev.x.scatter_forward()

  if mesh.comm.rank == 0:
    print(f"t={t}\tNRiter={NRiter}")

ofile1.close()
ofile2.close()

end = time.perf_counter()
if mesh.comm.rank == 0:
  print(f"calculation time: {(end-start)/60} min.")