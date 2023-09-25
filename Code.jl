# Problem Set 1: Demand Estimation and BLP
# Joe Wilske
# September 2023

using Distributions, LinearAlgebra, Statistics, Optim

# Data Generation:

# (j = 3) x (m = 100) x (# of product characteristics = 3)
function define_X()
    X = []
    for j in 1:3
        mlayer = []
        for m in 1:100
            push!(mlayer, [1, rand(Uniform(0, 1)), rand(Normal(0, 1))])
        end
        push!(X, mlayer)
    end
    return X
end
X = define_X()
# (j = 3) x (m = 100) vector
ξ = [rand(Normal(0,1), 100), rand(Normal(0,1), 100), rand(Normal(0,1), 100)]
# (j = 3) vector
W = rand(Normal(0,1), 3)
# (j = 3) x (m = 100) vector
Z = [rand(Normal(0,1), 100), rand(Normal(0,1), 100), rand(Normal(0,1), 100)] 
# (j = 3) vector, constants
β = [5, 1, 1]; α = 1; σ_α = 1; γ1 = 2; γ2 = 1; γ3 = 1; γ = [γ1, γ2, γ3]

## The following still needs to be calculated quasi-analytically. The specifications below
## are placeholders until I figure out how to do that.
# (j = 3) x (m = 100) vectors
p = [rand(Uniform(0, 3)), rand(Uniform(0, 3)), rand(Uniform(0, 3))]
# (j = 3) x (m = 100) vectors
function define_s()
    a = rand(Uniform(0, 1), 100)
    a_difference = 1 .- a
    b = []
    for i in 1:100
        b = vcat(b, rand(Uniform(0, a_difference[i]), 1))
    end
    c = 1 .- a .- b
    return [a, b, c]
end
s = define_s()

# Initial θ values:
θ_hat = (β_hat, α_hat, σ_α_hat) = θ_hat0 = (β_hat0, α_hat0, σ_α_hat0) = ([1, 1, 1], 1, 1)

# Simulation estimator for s:
function estimate_sjm(θ_hat, δ, p, X, j, m)
    σ_α = θ_hat[3]
    v_dist = LogNormal(0, σ_α)
    sum = 0
    for v_draw in 1:1000
        v = rand(v_dist)
        top = exp(δ[j][m] - σ_α*v*p[j][m])
        bottom = 1
        for product in eachindex(p)
            bottom += exp(δ[product][m] - σ_α*v*p[product][m])
        end
        sum += top / bottom
    end
    sjm = sum / 1000
    return sjm
end

function calculate_δjm(s, θ_hat, δ, p, X, j, m)
    sjm_estimate = estimate_sjm(θ_hat, δ, p, X, j, m)
    δjm = δ[j][m]
    while abs(log(s[j][m]) - log(sjm_estimate)) > 0.01
        δjm += log(s[j][m]) - log(sjm_estimate)
        sjm_estimate = estimate_sjm(θ_hat, δ, p, X, j, m)
        # this is the problem function. sjm estimate is not converging to actual. Could be because I didn't calculate s properly.
        display(δjm)
    end
    return δjm
end

function calculate_δ(s, θ_hat, p, X)
    δ = []
    mlayer = repeat([1], length(p[1]))
    for j in eachindex(p)
        push!(δ, mlayer)
    end
    for j in eachindex(δ)
        for m in eachindex(δ[j])
            δ[j][m] = calculate_δjm(s, θ_hat, δ, p, X, j, m)
        end
    end
end
δ = calculate_δ(s, θ_hat, p, X)

function calculate_ξ(δ, X, β_hat, α_hat, p)
    ξ = repeat([0], length(p))
    for j in eachindex(p)
        ξ[j] = repeat([0], length(p[j]))
    end
    for j in eachindex(p)
        for m in eachindex(p[j])
            ξ[j][m] = δ[j][m] - X[j][m]'β_hat + α_hat*p[j][m]
        end
    end
end
ξ = calculate_ξ(δ, X, β_hat, α_hat, p)
            



