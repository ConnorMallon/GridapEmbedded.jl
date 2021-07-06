
#using Pkg
#Pkg.activate("/home/user/Documents/GitHub/cmallon-amsi-summer-project/src/Testing_Code/4_ProjectcodeTrial")

using Gridap
using GridapEmbedded.CSG
using GridapEmbedded.Interfaces
using GridapEmbedded.LevelSetCutters

domain = (0,1,0,1,0,1)
partition = (10,10,10)
model = CartesianDiscreteModel(domain,partition)

trianbg = Triangulation(model)
writevtk(trianbg,"trianbg")

geo5 = sphere(0.5,x0=Point(0.5,0.5,0.5))
geo6 = cube(L=1,x0=Point(0.2,0.2,0.2))
geo7 = intersect(geo6,geo5)

cutgeo = cut(model,geo7)

trian = Triangulation(cutgeo)
writevtk(trian,"trian")

cutgeo_facets = cut_facets(model,geo6)

trian_Γn = BoundaryTriangulation(cutgeo_facets)

writevtk(trian_Γn,"trian_Γn")


2

#=
geo8=discretize(geo5,model)
geo9=discretize(geo7,model)
geo10 = intersect(geo8,geo9)

cutgeo2 = cut(model,geo10)
trian2 = Triangulation(cutgeo2)
writevtk(trian2,"trian2")

cutgeo_facets = cut_facets(model,geo10)
trian_Γn = BoundaryTriangulation(cutgeo_facets,"boundary",geo10)

writevtk(trian_Γn,"trian_Γn")
=#