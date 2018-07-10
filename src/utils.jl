function log2space_symmetric(symm_val::Float64, n::Int64, base_val::Int64=2)

    @assert n%2==1
    @assert symm_val > 0

    vals = Vector{Float64}()
    idx = 1
    midpt = convert(Int64,round((n-1)/2))

    for i=1:midpt
        push!(vals, -(symm_val/idx))
        idx = idx*base_val
    end

    # Add 0.0 to axis
    symm_vect = reverse(vals)
    push!(vals, 0.0)
    append!(vals, symm_vect)

    return vals
end