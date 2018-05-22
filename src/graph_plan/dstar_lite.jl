abstract type AbstractDStarLitePredecessorGenerator end

mutable struct DStarLiteStates{D<:Number,Heap,H}
  parent_indices::Vector{Int}
  gvalue::Vector{D}
  rhsvalue::Vector{D}
  km::D
  heap::Heap
  hmap::Vector{H}
  inHeap::Vector{Bool}
end

struct DStarLiteHEntry{D}
  vIdx::Int
  key::Tuple{D,D}
end

function Base.isless(e1::DStarLiteHEntry, e2::DStarLiteHEntry)
  return e1.key < e2.key
end

function create_dstarlite_states(g::AbstractGraph{V}, ::Type{D}) where {V,D<:Number}
  n = num_vertices(g)
  parent_indices = zeros(Int,n)
  gvalue = fill(typemax(D),n)
  rhsvalue = fill(typemax(D),n)
  km = 0.0
  heap = mutable_binary_minheap(DStarLiteHEntry{D})
  hmap = zeros(Int, n)
  inHeap = fill(false,n)

  return DStarLiteStates(parent_indices, gvalue, rhsvalue, km, heap, hmap,inHeap)
end

function calculate_key(state::DStarLiteStates, s::Int, heuristic::Function)

  key = (min(state.gvalue[s], state.rhsvalue[s]) +  heuristic(s) + state.km, min(state.gvalue[s], state.rhsvalue[s]) )
  return key
end


function initialize!(state::DStarLiteStates{D}, goalIdx::Int, heuristic::Function) where {D}
  
  state.rhsvalue[goalIdx] = zero(D)
  state.hmap[goalIdx] = push!(state.heap, DStarLiteHEntry(goalIdx, ( heuristic(goalIdx), zero(D) ) ) )
  state.inHeap[goalIdx] = true
  state.parent_indices[goalIdx] = goalIdx
end


function update_vertex!(state::DStarLiteStates, u::Int, heuristic::Function)

  u_key = calculate_key(state,u,heuristic)

  if state.gvalue[u] != state.rhsvalue[u]
    # Locally inconsistent
    if state.inHeap[u] == true
        update!(state.heap, state.hmap[u], DStarLiteHEntry(u, (u_key)))
    else
        state.hmap[u] = push!(state.heap, DStarLiteHEntry(u, (u_key)))
        state.inHeap[u] = true
    end
  else
    # Locally consistent
    # TODO : Remove (semantically) from heap if already present, need to check while popping
    state.inHeap[u] = false
  end

end


function compute_shortest_path!(
  graph::AbstractGraph{V},
  edge_dists::AbstractEdgePropertyInspector{D},
  source_vert::V,
  target_vert::V,
  generator::AbstractDStarLitePredecessorGenerator,
  heuristic::Function,
  state::DStarLiteStates{D,Heap,H}) where {V,D,Heap,H}

  @graph_requires graph incidence_list vertex_map vertex_list
  edge_property_requirement(edge_dists, graph)

  source_idx = vertex_index(source_vert,graph)
  target_idx = vertex_index(target_vert,graph)

  while !isempty(state.heap)
    
    entry = top(state.heap)

    # If it was flagged as needing to be removed from the priority queue, look for next
    if state.inHeap[entry.vIdx] == false
      pop!(state.heap)
      continue
    end

    if entry.key < calculate_key(state,source_idx,heuristic) || state.rhsvalue[source_idx] > state.gvalue[source_idx]
      new_key = calculate_key(state,entry.vIdx,heuristic)

      if isless(entry.key,new_key)
        # Update entry with new key since made worse
        update!(state.heap, state.hmap[entry.vIdx], DStarLiteHEntry(entry.vIdx, new_key))
      else     
        # Put in separate block as predecessors of top will need to be computed
        entry_vert::V = graph.vertices[entry.vIdx]
        generate_predecessors!(generator,entry_vert)

        if state.gvalue[entry.vIdx] > state.rhsvalue[entry.vIdx]
          # Locally overconsistent
          state.gvalue[entry.vIdx] = state.rhsvalue[entry.vIdx]
          state.inHeap[entry.vIdx] = false # 'Remove' from heap
          pop!(state.heap)

          # Iterate over predecessors
          for e in in_edges(entry_vert, graph)
            iv::Int = vertex_index(source(e, graph),graph)

            # Update rhs value if not goal
            if iv != target_idx
              state.rhsvalue[iv] = min(state.rhsvalue[iv], edge_property(edge_dists, e, graph) + state.gvalue[entry.vIdx])
            end

            # Update vertex regardless
            update_vertex!(state,iv,heuristic)
          end
        else
          # Locally underconsistent
          g_old = state.gvalue[entry.vIdx]
          state.gvalue[entry.vIdx] = typemax(D)

          for e in in_edges(entry_vert, graph)
            v::V = source(e, graph)
            iv::Int = vertex_index(v,graph)

            if state.rhsvalue[iv] == edge_property(edge_dists, e, graph) + g_old
              if iv != target_idx
                minrhsval = typemax(D)
                for oe in out_edges(v,graph)
                  ivp::Int = vertex_index(target(oe,graph),graph)
                  val = edge_property(edge_dists, oe, graph) + state.rhsvalue[ivp]
                  if val < minrhsval
                    minrhsval = val
                  end
                end
              end
            end

            update_vertex!(state,iv,heuristic)
          end
          

          # Now do same for entry_vert
          if state.rhsvalue[entry.vIdx] == g_old
            if entry.vIdx != target_idx
              minrhsval = typemax(D)
              for oe in out_edges(entry_vert,graph)
                ivp::Int = vertex_index(target(oe,graph),graph)
                val = edge_property(edge_dists, oe, graph) + state.rhsvalue[ivp]
                if val < minrhsval
                  minrhsval = val
                end
              end
            end
          end
          update_vertex!(state,entry.vIdx,heuristic)
        end
      end
    else
      # True loop condition is violated
      break
    end
  end

end

function dstarlite_solve(
  graph::AbstractGraph{V},
  edge_len::AbstractEdgePropertyInspector{D},
  source_vert::V,
  target_vert::V,
  generator::AbstractDStarLitePredecessorGenerator,
  heuristic::Function = n -> zero(D)) where {V,D}

  # Solve for the first time
  state = create_dstarlite_states(graph,D)
  initialize!(state,vertex_index(target_vert,graph),heuristic)
  compute_shortest_path!(graph,edge_len,source_vert,target_vert,generator,heuristic,state)

  return state
end


function dstarlite_resolve(
  graph::AbstractGraph{V},
  edge_len::AbstractEdgePropertyInspector{D},
  s_last::V,
  curr_source::V,
  target_vert::V,
  generator::AbstractDStarLitePredecessorGenerator,
  state::DStarLiteStates{D,Heap,H},
  heuristic::Function = n -> zero(D)) where {V,D,Heap,H}

  # ASSUME EDGE COSTS CHANGED
  state.km = state.km + heuristic(s_last.idx)
  compute_shortest_path!(graph,edge_len,curr_source,target_vert,generator,heuristic,state)

  return state

end


    