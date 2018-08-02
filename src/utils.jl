function log2space_symmetric(symm_val::Float64, n::Int64, base_val::Int64=2)

    # Ensure that the number of terms is odd (so around 0.0)
    @assert n%2==1
    # Ensure that the value itself is positive
    @assert symm_val > 0

    vals = Vector{Float64}()
    idx = 1
    midpt = convert(Int64,round((n-1)/2))

    for i=1:midpt
        push!(vals, -(symm_val/idx))
        idx = idx*base_val
    end

    # Add 0.0 to axis
    symm_vect = reverse(-vals)
    push!(vals, 0.0)
    append!(vals, symm_vect)

    return vals
end


function polyspace_symmetric(symm_val::Float64, n::Int64, exp_val::Int64=3)

    # Ensure that the number of terms is odd (so around 0.0)
    @assert n%2==1
    # Ensure that the value itself is positive
    @assert symm_val > 0

    vals = Vector{Float64}()
    idx = 1
    midpt = convert(Int64,round((n-1)/2))

    x = (symm_val/(midpt^exp_val))^(1/exp_val)

    for i=midpt:-1:1
        val = -1*(i*x)^exp_val
        push!(vals,val)
    end

    symm_vect = reverse(-vals)
    push!(vals, 0.0)
    append!(vals, symm_vect)

    return vals
end