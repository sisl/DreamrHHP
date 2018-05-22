mutable struct AStarStates{D<:Number,Heap,H}
    parent_indices::Vector{Int}
    dists::Vector{D}
    colormap::Vector{Int}
    heap::Heap
    hmap::Vector{H}
end

struct AStarHEntry{D}
    vIdx::Int
    gvalue::D
    fvalue::D
end

function Base.isless(e1::AStarHEntry, e2::AStarHEntry)
    return e1.fvalue < e2.fvalue
end

# create Astar states

function create_astar_states(g::AbstractGraph{V}, ::Type{D}) where {V, D <: Number}
    n = num_vertices(g)
    parent_indices = zeros(Int, n)
    dists = fill(typemax(D), n)
    colormap = zeros(Int, n)
    heap = mutable_binary_minheap(AStarHEntry{D})
    hmap = zeros(Int, n)

    return AStarStates(parent_indices, dists, colormap, heap, hmap)
end


# NO VISITOR DEFS NEEDED AS IT INCLUDES GRAPHS

###################################################################
#
#   core algorithm implementation
#
###################################################################

function set_source!(state::AStarStates{D}, g::AbstractGraph{V}, s::V) where {D, V}
    i = vertex_index(s, g)
    state.parent_indices[i] = i
    state.dists[i] = 0
    state.colormap[i] = 2
end

function process_neighbors!(
    state::AStarStates{D,Heap,H},
    graph::AbstractGraph{V},
    edge_dists::AbstractEdgePropertyInspector{D},
    u::V, du::D, visitor::AbstractDijkstraVisitor,
    heuristic::Function) where {V, D, Heap, H}

    dv::D = zero(D)

    for e in out_edges(u, graph)
        v::V = target(e, graph)
        iv::Int = vertex_index(v, graph)
        v_color::Int = state.colormap[iv]

        if v_color == 0
            state.dists[iv] = dv = du + edge_property(edge_dists, e, graph)
            state.parent_indices[iv] = vertex_index(u, graph)
            state.colormap[iv] = 1
            Graphs.discover_vertex!(visitor, u, v, dv)

            # push new vertex to the heap
            state. hmap[iv] = push!(state.heap, AStarHEntry(iv, dv, dv + heuristic(v)))

        elseif v_color == 1
            dv = du + edge_property(edge_dists, e, graph)
            if dv < state.dists[iv]
                state.dists[iv] = dv
                state.parent_indices[iv] = vertex_index(u, graph)

                # update the value on the heap
                Graphs.update_vertex!(visitor, u, v, dv)
                update!(state.heap, state.hmap[iv], AStarHEntry(iv, dv, dv + heuristic(v)))
            end
        end
    end
end


function astar_shortest_path!(
    graph::AbstractGraph{V},                # the graph
    edge_dists::AbstractEdgePropertyInspector{D}, # distances associated with edges
    source::V,             # the source
    visitor::AbstractDijkstraVisitor,# visitor object
    heuristic::Function,      # Heuristic function for vertices
    state::AStarStates{D,Heap,H}) where {V, D, Heap, H}

    @graph_requires graph incidence_list vertex_map vertex_list
    edge_property_requirement(edge_dists, graph)

    # initialize for source

    d0 = zero(D)

    set_source!(state, graph, source)
    if !Graphs.include_vertex!(visitor, source, source, d0)
        return state
    end

    # process direct neighbors of source
    process_neighbors!(state, graph, edge_dists, source, d0, visitor,heuristic)
    Graphs.close_vertex!(visitor, source)

    # main loop

    while !isempty(state.heap)

        # pick next vertex to include
        entry = pop!(state.heap)
        u::V = graph.vertices[entry.vIdx]
        du::D = entry.gvalue

        ui = vertex_index(u, graph)
        state.colormap[ui] = 2
        if !Graphs.include_vertex!(visitor, graph.vertices[state.parent_indices[ui]], u, du)
            return state
        end

        # process u's neighbors

        process_neighbors!(state, graph, edge_dists, u, du, visitor,heuristic)
        Graphs.close_vertex!(visitor, u)
    end

    state
    end


function astar_shortest_path(
    graph::AbstractGraph{V},                # the graph
    edge_len::AbstractEdgePropertyInspector{D}, # distances associated with edges
    source::V,
    visitor::AbstractDijkstraVisitor,
    heuristic::Function = n -> 0) where {V, D}
    state = create_astar_states(graph, D)
    astar_shortest_path!(graph, edge_len, source, visitor, heuristic, state)
end