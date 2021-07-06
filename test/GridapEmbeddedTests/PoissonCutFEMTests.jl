module PoissonCutFEMTests

using Gridap
using GridapEmbedded
using Test

# Manufactured solution
u(x) = x[1] + x[2] - x[3]
∇u(x) = ∇(u)(x)
f(x) = -Δ(u)(x)
ud(x) = u(x)

# Select geometry
const R = 0.4
geom = disk(R,x0=Point(0.75,0.5))
n = 10
partition = (n,n)

# Setup background model
L=1
domain = (0,L,0,L)
bgmodel = simplexify(CartesianDiscreteModel(domain,partition))
const h = L/n

# Cut the background model
cutdisc = cut(bgmodel,geom)

# Setup integration meshes
Ω = Triangulation(cutdisc)
Γ = EmbeddedBoundary(cutdisc)
Γg = GhostSkeleton(cutdisc)

# Setup normal vectors
n_Γ = get_normal_vector(Γ)
n_Γg = get_normal_vector(Γg)

# Setup Lebesgue measures
order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

# Setup FESpace
model = DiscreteModel(cutdisc)
V = TestFESpace(model,ReferenceFE(lagrangian,Float64,order),conformity=:H1)
U = TrialFESpace(V)

uh = interpolate(u, U)
uh_Γn = restrict(uh,trian_Γn)

# Weak form Nitsche + ghost penalty (CutFEM paper Sect. 6.1)
const γd = 10.0
const γg = 0.1

a(u,v) =
  ∫( ∇(v)⋅∇(u) ) * dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ +
  ∫( (γg*h)*jump(n_Γg⋅∇(v))*jump(n_Γg⋅∇(u)) ) * dΓg

l(v) =
  ∫( v*f ) * dΩ +
  ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ

# FE problem
op = AffineFEOperator(a,l,U,V)
uh = solve(op)

e = u - uh

# Postprocess
l2(u) = sqrt(sum( ∫( u*u )*dΩ ))
h1(u) = sqrt(sum( ∫( u*u + ∇(u)⋅∇(u) )*dΩ ))

el2 = l2(e)
eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

# writevtk(Ω,"results",cellfields=["uh"=>uh])
@test el2/ul2 < 1.e-8
@test eh1/uh1 < 1.e-7




end # module
