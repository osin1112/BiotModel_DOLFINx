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
from dolfinx.io import VTKFile, XDMFFile

width, height = 1, 15 # [m]
nx, ny = 5, 75
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD,
                                     np.array([[0.0, 0.0], [width, height]]),
                                     [nx, ny],
                                     cell_type=dolfinx.mesh.CellType.quadrilateral)
fdim = mesh.topology.dim - 1
tdim = mesh.topology.dim
mesh.topology.create_connectivity(fdim,tdim) # estsablish surface (fdim) -> cell (tdim) connection

bd = {
  "bottom" : 1,
  "right"  : 2,
  "top"    : 3,
  "left"   : 4
}
boundaries = [
  (bd["bottom"], lambda x : np.isclose(x[1], 0.0)),
  (bd["right"],  lambda x : np.isclose(x[0], width)),
  (bd["top"],    lambda x : np.isclose(x[1], height)),
  (bd["left"],   lambda x : np.isclose(x[0], 0.0))
]
def GetFacets() -> dolfinx.mesh.MeshTags: # You have to use 32bit int for facets
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

with XDMFFile(mesh.comm,GetValue(params,"OutputPath","str")+"/boundary.xdmf","w") as xdmf:
  xdmf.write_mesh(mesh)
  xdmf.write_meshtags(facets,mesh.geometry)

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

U_prev = dolfinx.fem.Function(V, name="displacement")
with U_prev.vector.localForm() as local:
  local.set(PETSc.ScalarType(0.0))

# functionspace for p
Q, dofQ = MixedSpace.sub(1).collapse()
P = dolfinx.fem.Function(Q, name="pressure")
with P.vector.localForm() as local:
  local.set(PETSc.ScalarType(GetValue(params,"P_init","float")))
X_prev.x.array[dofQ] = P.x.array
X_prev.x.scatter_forward()

def calcVelocity() -> float:
  u1, u0 = [], []
  for i in range(tdim):
    dof = dolfinx.fem.locate_dofs_topological(V.sub(i),fdim,facets.find(bd["top"]))
    u1.append(U.x.array[dof])
    u0.append(U_prev.x.array[dof])
  u1 = np.mean(np.sqrt(u1[0]**2+u1[1]**2))
  u0 = np.mean(np.sqrt(u0[0]**2+u0[1]**2))
  return (u1-u0)/dt

import ufl
from ufl import dx, diff, inner, dot, grad, Identity, tr, det, ln, nabla_div, derivative, sym

def CauchyStress_LinearElasticBody(u,Mu,Lambda) :
  return 2*Mu*sym(grad(u)) + Lambda*tr(sym(grad(u)))*Identity(len(u))
def DilatationalStrain(u) :
  F = ufl.variable(Identity(len(u)) + grad(u))
  J = det(F)
  return J - 1

u, p = ufl.split(X)
u_prev, p_prev = ufl.split(X_prev)
wu, wp = ufl.TestFunctions(MixedSpace)

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
epsilon = DilatationalStrain(u)
epsilon_dot = DilatationalStrain((u-u_prev)/dt)
traction = dolfinx.fem.Constant(mesh,PETSc.ScalarType(GetValue(params,"traction","float")))
source = dolfinx.fem.Constant(mesh,PETSc.ScalarType(0.0)) # 湧き出しなし．

theta = 0.5 # Crank-Nicolson
p_theta = p*theta + p_prev*(1-theta)
# Solid equilibrium
WeakForm  =  inner(grad(wu),sigma)*dx              \
           - p*nabla_div(wu)*dx                    \
           - traction*inner(wu,nVec)*ds(bd["top"])

# Mass conservation
WeakForm +=  S*wp*((p-p_prev)/dt)*dx          \
           + k*dot(grad(p_theta),grad(wp))*dx \
           - wp*source*dx                     \
           + alpha*wp*epsilon_dot*dx          

# uy = 0 on bottom
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(1),fdim,facets.find(bd["bottom"]))
bc1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(1))
# ux = 0 on left & right
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(0),fdim,facets.find(bd["left"]))
bc2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(0))
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(0).sub(0),fdim,facets.find(bd["right"]))
bc3 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(0).sub(0))
# p = 0 on top
dof = dolfinx.fem.locate_dofs_topological(MixedSpace.sub(1),fdim,facets.find(bd["top"]))
bc4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0),dof,MixedSpace.sub(1))
DirichletBC = [bc1,bc2,bc3,bc4]

import dolfinx.fem.petsc
import dolfinx.nls.petsc

dX = ufl.TrialFunction(MixedSpace)
Jacobian = derivative(WeakForm,X,dX)
problem = dolfinx.fem.petsc.NonlinearProblem(WeakForm,X,bcs=DirichletBC,J=Jacobian)
solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm,problem)
solver.rtol = GetValue(params,"NRrtol","float")
solver.atol = GetValue(params,"NRatol","float")
solver.relaxation_parameter = GetValue(params,"NRrelaxation","float")
solver.max_it = GetValue(params,"NRmax_iter","int")
solver.convergence_criterion = GetValue(params,"NRcriterion","str")

def normalizedError(p,t: float) -> float:
  # first-class object for interpolate
  def Terzaghi(x) :
    p = 0.0
    L = 15
    c = 1.02e-9/(1.65e-10 + 1/120e6)
    Ku = 40e6 + 2*40e6/3 + 1/1.65e-10
    p0 = 1e4 * (1/1.65e-10)/(Ku+4*40e6/3)
    for m in range(int(1e4)):
      p += (1/(2*m+1)) * np.exp(- (2*m+1)**2 * np.pi**2 * c * t / (4*L**2)) * np.sin((2*m+1)*np.pi*(L-x[1])/(2*L))
    p = p * (4*p0/np.pi)
    return p
  
  p_ana = dolfinx.fem.Function(Q)
  p_ana.interpolate(Terzaghi)
  numerator   = dolfinx.fem.form(inner(p_ana,p_ana)*dx)
  denominator = dolfinx.fem.form(inner(p-p_ana,p-p_ana)*dx)
  error = dolfinx.fem.assemble_scalar(denominator) / dolfinx.fem.assemble_scalar(numerator)
  error = np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM))
  return error

ofreq = GetValue(params,"OutputFreq","int")
ofile1 = VTKFile(mesh.comm,GetValue(params,"OutputPath","str")+"/2D/u.pvd","w")
ofile2 = XDMFFile(mesh.comm,GetValue(params,"OutputPath","str")+"/2D/p.xdmf","w")
ofile3 = open(GetValue(params,"OutputPath","str")+"/2D/err.dat","w")
ofile1.write_mesh(mesh)
ofile2.write_mesh(mesh)

t = 0.0
dt = dt.value
for i in range(GetValue(params,"NumOfTimeStep","int")):
  t += dt

  NRiter, IsConverged = solver.solve(X)
  X.x.scatter_forward()

  U.x.array[:] = X.x.array[dofV]
  P.x.array[:] = X.x.array[dofQ]
  # output
  if i % ofreq == 0:
    ofile1.write_function(U,t)
    ofile2.write_function(P,t)
    ofile3.write(f"{normalizedError(P,t)}\n")

  print(calcVelocity())

  # update
  X_prev.x.array[:] = X.x.array
  X_prev.x.scatter_forward()

  U_prev.x.array[:] = X.x.array[dofV]

  if mesh.comm.rank == 0:
    print(f"t={t}\tNRiter={NRiter}")

ofile1.close()
ofile2.close()
ofile3.close()

end = time.perf_counter()
if mesh.comm.rank == 0:
  print(f"calculation time: {(end-start)/60} min.")