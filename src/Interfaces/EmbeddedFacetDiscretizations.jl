
struct EmbeddedFacetDiscretization{Dc,Dp,T} <: GridapType
  bgmodel::DiscreteModel{Dp,Dp}
  ls_to_facet_to_inoutcut::Vector{Vector{Int8}}
  subfacets::SubTriangulation{Dc,Dp,T}
  ls_to_subfacet_to_inout::Vector{Vector{Int8}}
  oid_to_ls::Dict{UInt,Int}
  geo::CSG.Geometry
end

function SkeletonTriangulation(cut::EmbeddedFacetDiscretization)
  facets = SkeletonTriangulation(cut.bgmodel)
  SkeletonTriangulation(cut,facets,cut.geo,(CUTIN,IN))
end

function SkeletonTriangulation(cut::EmbeddedFacetDiscretization,tags)
  SkeletonTriangulation(cut,tags,cut.geo,(CUTIN,IN))
end

function SkeletonTriangulation(cut::EmbeddedFacetDiscretization,tags,name::String)
  geo = get_geometry(cut.geo,name)
  SkeletonTriangulation(cut,tags,geo)
end

function SkeletonTriangulation(
  cut::EmbeddedFacetDiscretization,tags,geo::CSG.Geometry)
  SkeletonTriangulation(cut,tags,geo,(CUTIN,IN))
end

function SkeletonTriangulation(cut::EmbeddedFacetDiscretization,tags,name::String,in_or_out)
  geo = get_geometry(cut.geo,name)
  SkeletonTriangulation(cut,tags,geo,in_or_out)
end

function SkeletonTriangulation(
  cut::EmbeddedFacetDiscretization,tags,geo::CSG.Geometry,in_or_out)
  facets = SkeletonTriangulation(cut.bgmodel,tags)
  SkeletonTriangulation(cut,facets,geo,in_or_out)
end

function SkeletonTriangulation(
  cut::EmbeddedFacetDiscretization,facets::SkeletonTriangulation,geo::CSG.Geometry,in_or_out)
  facets1 = get_left_boundary(facets)
  facets2 = get_right_boundary(facets)
  trian1 = BoundaryTriangulation(cut,facets1,geo,in_or_out)
  trian2 = BoundaryTriangulation(cut,facets2,geo,in_or_out)
  SkeletonTriangulation(trian1,trian2)
end

function BoundaryTriangulation(cut::EmbeddedFacetDiscretization)
  facets = BoundaryTriangulation(cut.bgmodel)
  BoundaryTriangulation(cut,facets,cut.geo,(CUTIN,IN))
end

function BoundaryTriangulation(cut::EmbeddedFacetDiscretization,tags)
  BoundaryTriangulation(cut,tags,cut.geo,(CUTIN,IN))
end

function BoundaryTriangulation(cut::EmbeddedFacetDiscretization,tags,name::String)
  geo = get_geometry(cut.geo,name)
  BoundaryTriangulation(cut,tags,geo)
end

function BoundaryTriangulation(cut::EmbeddedFacetDiscretization,tags,geo::CSG.Geometry)
  BoundaryTriangulation(cut,tags,geo,(CUTIN,IN))
end

function BoundaryTriangulation(cut::EmbeddedFacetDiscretization,tags,name::String,in_or_out)
  geo = get_geometry(cut.geo,name)
  BoundaryTriangulation(cut,tags,geo,in_or_out)
end

function BoundaryTriangulation(cut::EmbeddedFacetDiscretization,tags,geo::CSG.Geometry,in_or_out)
  facets = BoundaryTriangulation(cut.bgmodel,tags)
  BoundaryTriangulation(cut,facets,geo,in_or_out)
end

function BoundaryTriangulation(
  cut::EmbeddedFacetDiscretization,
  facets::BoundaryTriangulation,
  geo::CSG.Geometry,
  in_or_out::Tuple)

  trian1 = BoundaryTriangulation(cut,facets,geo,in_or_out[1])
  trian2 = BoundaryTriangulation(cut,facets,geo,in_or_out[2])
  lazy_append(trian1,trian2)
end

function BoundaryTriangulation(
  cut::EmbeddedFacetDiscretization,
  facets::BoundaryTriangulation,
  geo::CSG.Geometry,
  in_or_out::Integer)

  bgfacet_to_inoutcut = compute_bgfacet_to_inoutcut(cut,geo)
  bgfacet_to_mask = apply( a->a==in_or_out, bgfacet_to_inoutcut)
  _restrict_boundary_triangulation(cut.bgmodel,facets,bgfacet_to_mask)
end

function BoundaryTriangulation(
  cut::EmbeddedFacetDiscretization,
  _facets::BoundaryTriangulation,
  geo::CSG.Geometry,
  in_or_out::CutInOrOut)

  bgfacet_to_inoutcut = compute_bgfacet_to_inoutcut(cut,geo)
  bgfacet_to_mask = apply( a->a==CUT, bgfacet_to_inoutcut)
  facets = _restrict_boundary_triangulation(cut.bgmodel,_facets,bgfacet_to_mask)

  facet_to_bgfacet = get_face_to_face(facets)
  n_bgfacets = num_facets(cut.bgmodel)
  bgfacet_to_facet = zeros(Int,n_bgfacets)
  bgfacet_to_facet[facet_to_bgfacet] .= 1:length(facet_to_bgfacet)

  subfacet_to_inoutcut = reindex(bgfacet_to_inoutcut,cut.subfacets.cell_to_bgcell)
  _subfacet_to_facet = reindex(bgfacet_to_facet,cut.subfacets.cell_to_bgcell)

  subfacet_to_inout = compute_subfacet_to_inout(cut,geo)
  pred(a,b,c) = c != 0 && a==CUT && b==in_or_out.in_or_out
  mask = apply( pred, subfacet_to_inoutcut, subfacet_to_inout, _subfacet_to_facet )
  newsubfacets = findall(mask)
  subfacets = SubTriangulation(cut.subfacets,newsubfacets)
  subfacet_to_facet = bgfacet_to_facet[subfacets.cell_to_bgcell]

  BoundarySubTriangulationWrapper(facets,subfacets,subfacet_to_facet)
end

function _restrict_boundary_triangulation(model,facets,bgfacet_to_mask)

  facet_to_bgfacet = get_face_to_face(facets)
  facet_to_mask = reindex(bgfacet_to_mask,facet_to_bgfacet)
  n_bgfacets = length(bgfacet_to_mask)
  bgfacet_to_mask2 = fill(false,n_bgfacets)
  bgfacet_to_mask2[facet_to_bgfacet] .= facet_to_mask

  BoundaryTriangulation(model,bgfacet_to_mask2,get_cell_around(facets))
end

function compute_bgfacet_to_inoutcut(cut::EmbeddedFacetDiscretization,geo::CSG.Geometry)

  tree = get_tree(geo)

  function conversion(data)
    f,name,meta = data
    oid = objectid(f)
    ls = cut.oid_to_ls[oid]
    cell_to_inoutcut = cut.ls_to_facet_to_inoutcut[ls]
    cell_to_inoutcut, name, meta
  end

  newtree = replace_data(identity,conversion,tree)
  compute_inoutcut(newtree)
end

function compute_subfacet_to_inout(cut::EmbeddedFacetDiscretization,geo::CSG.Geometry)

  tree = get_tree(geo)

  function conversion(data)
    f,name,meta = data
    oid = objectid(f)
    ls = cut.oid_to_ls[oid]
    cell_to_inoutcut = cut.ls_to_subfacet_to_inout[ls]
    cell_to_inoutcut, name, meta
  end

  newtree = replace_data(identity,conversion,tree)
  compute_inoutcut(newtree)
end

struct BoundarySubTriangulationWrapper{Dc,Dp,T} <: Triangulation{Dc,Dp}
  facets::BoundaryTriangulation{Dc,Dp}
  subfacets::SubTriangulation{Dc,Dp,T}
  subfacet_to_facet::AbstractArray
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  cell_ids
  cell_normals
  subfacet_to_facet_map

  function BoundarySubTriangulationWrapper(
    facets::BoundaryTriangulation{Dc,Dp},
    subfacets::SubTriangulation{Dc,Dp,T},
    subfacet_to_facet::AbstractArray) where {Dc,Dp,T}

    reffe = LagrangianRefFE(Float64,Simplex(Val{Dc}()),1)
    cell_types = fill(Int8(1),length(subfacets.cell_to_points))
    reffes = [reffe]
    cell_ids = reindex(get_cell_id(facets),subfacet_to_facet)
    cell_normals = reindex(get_normal_vector(facets),subfacet_to_facet)
    subfacet_to_facet_map = _setup_subcell_to_cell_map(subfacets,reffe,cell_types)
    new{Dc,Dp,T}(facets,subfacets,subfacet_to_facet,reffes,cell_types,cell_ids,cell_normals,subfacet_to_facet_map)
  end
end

function get_node_coordinates(trian::BoundarySubTriangulationWrapper)
  trian.subfacets.point_to_coords
end

function get_cell_nodes(trian::BoundarySubTriangulationWrapper)
  trian.subfacets.cell_to_points
end

function get_cell_coordinates(trian::BoundarySubTriangulationWrapper)
  node_to_coords = get_node_coordinates(trian)
  cell_to_nodes = get_cell_nodes(trian)
  LocalToGlobalArray(cell_to_nodes,node_to_coords)
end

function get_reffes(trian::BoundarySubTriangulationWrapper)
  trian.reffes
end

function get_cell_type(trian::BoundarySubTriangulationWrapper)
  trian.cell_types
end

function get_normal_vector(trian::BoundarySubTriangulationWrapper)
  trian.cell_normals
end

function get_cell_id(trian::BoundarySubTriangulationWrapper)
  trian.cell_ids
end

function restrict(f::AbstractArray,trian::BoundarySubTriangulationWrapper)
  g = restrict(f,trian.facets)
  h = reindex(g,trian.subfacet_to_facet)
  compose_field_arrays(h,trian.subfacet_to_facet_map)
end

