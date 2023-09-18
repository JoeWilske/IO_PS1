# Problem Set 1
# Joe Wilske
# September 2023

using Distributions, LinearAlgebra, Statistics

# Data Generation:

# (m = 100) vectors
X1 = repeat([1], 100)
X2 = rand(Uniform(0, 1), 100)
X3 = rand(Normal(0, 1), 100)
# (j = 100) x (m = 3) matrix
X = hcat(X1, X2, X3)
# (j = 3) x (m = 100) vector
ξ = [rand(Normal(0,1), 100), rand(Normal(0,1), 100), rand(Normal(0,1), 100)]
# (j = 3) vector
W = rand(Normal(0,1), 3)
# (j = 3) x (m = 100) vector
Z = [rand(Normal(0,1), 100), rand(Normal(0,1), 100), rand(Normal(0,1), 100)] 
# (j = 3) vector, constants
β = [5, 1, 1]; α = 1; σ_α = 1; γ1 = 2; γ2 = 1; γ3 = 1; γ = [γ1, γ2, γ3]


