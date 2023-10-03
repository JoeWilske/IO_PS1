# Problem Set 1: Demand Estimation and BLP
# Joe Wilske
# September 2023

using Distributions, LinearAlgebra, Statistics, Optim, 
      Roots, Random, StatsPlots
Random.seed!(7)

#####################################
# Problem 0: Data Generation
#####################################

initial3() = [zeros(100), zeros(100), zeros(100)]
initial4() = [zeros(100), zeros(100), zeros(100), zeros(100)]
# (j = 3) x (m = 100) x (# of product characteristics = 3) vectors
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
# (j = 3) x (m = 100) vector
η = [rand(Normal(0,1), 100), rand(Normal(0,1), 100), rand(Normal(0,1), 100)]
# (j = 3) vector, constants
β = [5, 1, 1]; α = 1; σ_α = 1; γ0 = 2; γ1 = 1; γ2 = 1; γ = [γ0, γ1, γ2]
# (j = 3) x (m = 100) vector
function define_MC(γ, W, Z, η)
    MC = initial3()
    for j in eachindex(MC)
        for m in eachindex(MC[j])
            if γ[1] + γ[2]*W[j] + γ[3]*Z[j][m] + η[j][m] > 0
                MC[j][m] = γ[1] + γ[2]*W[j] + γ[3]*Z[j][m] + η[j][m]
            else
                MC[j][m] = 0
            end
        end
    end
    return MC
end
MC = define_MC(γ, W, Z, η)
# (j = 3) x (m = 100) vector
function generate_s(X, β, α, p, ξ, σ_α)
    s = initial4()
    for j in eachindex(p)
        for m in eachindex(p[j])
            sjm_estimate = 0
            for draw in 1:1000
                ν_draw = rand(LogNormal(0,1), 1000)
                sjm_estimate += (1/1000)*(exp(X[j][m]'*β - α*p[j][m] + ξ[j][m] - σ_α*ν_draw[draw]*p[j][m])) / 
                            (1 + exp(X[1][m]'*β - α*p[1][m] + ξ[1][m] - σ_α*ν_draw[draw]*p[1][m]) + 
                            exp(X[2]'[m]*β - α*p[2][m] + ξ[2][m] - σ_α*ν_draw[draw]*p[2][m]) + 
                            exp(X[3]'[m]*β - α*p[3][m] + ξ[3][m] - σ_α*ν_draw[draw]*p[3][m]))
            end
            s[j][m] = deepcopy(sjm_estimate)
        end
    end
    for m in eachindex(p[1])
        s[4][m] = 1 - s[1][m] - s[2][m] - s[3][m]
        for j in eachindex(s)
            if s[j][m] < 0 
                s[j][m] = 0.000000000000000000000001
            end
        end
    end
    return s
end
# (j = 3) x (m = 100) vector
function calculate_elasticity(X, β, α, p, ξ, σ_α)
    ϵ = initial3()
    s = generate_s(X, β, α, p, ξ, σ_α)
    for j in eachindex(p)
        for m in eachindex(p[j])
            ν = rand(LogNormal(0,1),1000)
            α_i = α .+ σ_α*ν
            Σ(i, m) = 1 + exp(X[1][m]'*β - α*p[1][m] + ξ[1][m] - σ_α*ν[i]*p[1][m]) + 
                          exp(X[2][m]'*β - α*p[2][m] + ξ[2][m] - σ_α*ν[i]*p[2][m]) +
                          exp(X[3][m]'*β - α*p[3][m] + ξ[3][m] - σ_α*ν[i]*p[3][m])
            dsdp_jm = 0
            for i in 1:1000
                dsdp_jm += α_i[i] * exp(X[j][m]'*β - α*p[j][m] + ξ[j][m] - σ_α*ν[i]*p[j][m]) * 
                           (Σ(i, m) - α_i[i]*exp(X[j][m]'*β - α*p[j][m] + ξ[j][m] - σ_α*ν[i]*p[j][m])) / 
                           (Σ(i, m)^2)
            end
            dsdp_jm /= 1000
            ϵ[j][m] = dsdp_jm * p[j][m] / s[j][m]

            if isnan(ϵ[j][m]) == true
                ϵ[j][m] = -2
            elseif ϵ[j][m] == 1.0
                ϵ[j][m] = -2
            end
        end
    end
    return ϵ
end
# (j = 3) x (m = 100) vector
function calculate_p(MC, X, β, α, ξ, σ_α)
    p_try = [rand(Uniform(0, 5), 100), rand(Uniform(0, 5), 100), rand(Uniform(0, 5), 100)]
    p = initial3()
    test = 1000
    while test > 100
        ϵ = calculate_elasticity(X, β, α, p_try, ξ, σ_α)
        p = deepcopy(p_try)
        for j in eachindex(p)
            for m in eachindex(p[j])
                p_try[j][m] = (MC[j][m]*ϵ[j][m]) / (1 + ϵ[j][m])
                if isnan(p_try[j][m]) == true
                    p_try[j][m] = 0
                end
            end
        end
        test = norm(p - p_try)
        display("Latest iterative norm (threshold = 100):")
        display(test)
    end
    for j in eachindex(p)
        for m in eachindex(p[j])
            if p_try[j][m] < 0.0
                p_try[j][m] = 0.0
            end
        end
    end
    return p_try
end
p = calculate_p(MC, X, β, α, ξ, σ_α)
s = generate_s(X, β, α, p, ξ, σ_α)

# This function is a placeholder until I figure out why I have NaN
# elements in p and s
function fix_broken_estimates(ω)
    for j in eachindex(ω)
        for m in 1:100
            if isnan(ω[j][m]) == true
                ω[j][m] = 1/3
            end
            if ω[j][m] < 0
                ω[j][m] = 0.000000000000000000000001
            end
        end
    end
    return ω
end
p = fix_broken_estimates(p)
s = fix_broken_estimates(s)

##########################################
# Problem 1: Demand Side
##########################################

##### 2 #####

# Initial θ values:
(β1_hat, β2_hat, β3_hat, α_hat, σ_α_hat) = (6, .75, 1.25, 1.2, 0.8)
θ_hat = copy([β1_hat, β2_hat, β3_hat, α_hat, σ_α_hat])

# Simulation estimator for s:
function estimate_s(θ_hat, δ, p)
    σ_α_hat = θ_hat[5]
    s = initial4()
    for j in eachindex(p)
        for m in eachindex(p[j])
            sjm_estimate = 0
            for draw in 1:1000
                ν_draw = rand(LogNormal(0,1),1000)
                sjm_estimate += (1/1000)*(exp(δ[j][m] - σ_α_hat*ν_draw[draw]*p[j][m])) / 
                            (1 + exp(δ[j][m] - σ_α_hat*ν_draw[draw]*p[1][m]) + 
                            exp(δ[j][m] - σ_α_hat*ν_draw[draw]*p[2][m]) + 
                            exp(δ[j][m] - σ_α_hat*ν_draw[draw]*p[3][m]))
            end
            s[j][m] = sjm_estimate
        end
    end
    for m in eachindex(p[1])
        if 1 - s[1][m] - s[2][m] - s[3][m] >= 0
            s[4][m] = 1 - s[1][m] - s[2][m] - s[3][m]
        else
            s[4][m] = 0.000000000000000000000000001
        end
    end
    return s
end

# Contraction mapping algorithm to calculate δ from s:
function contraction_map(s, θ_hat, p)
    δ_next = initial4()
    δ = [ones(100), ones(100), ones(100), ones(100)]
    test = initial4()
    while norm(test) > 0.01
        s_prediction = estimate_s(θ_hat, δ, p)
        for j in eachindex(δ)
            for m in eachindex(δ[j])
                δ_next[j][m] = δ[j][m] + log(s[j][m]) - log(s_prediction[j][m])
            end
        end
        for m in 1:100
            for j in 1:4
                δ_next[j][m] -= δ_next[4][m]
                test[j][m] = δ[j][m] - δ_next[j][m]
                δ[j][m] += log(s[j][m]) - log(s_prediction[j][m]) - δ_next[4][m]
            end
        end
        display(norm(test))
    end
    return δ_next
end

# Find ξ from estimated δ and θ_hat.
# (j = 3) x (m = 100) vector
function calculate_ξ(X, θ_hat, p)
    δ = contraction_map(s, θ_hat, p)
    ξ = initial3()
    β_hat = [θ_hat[1], θ_hat[2], θ_hat[3]]
    α_hat = θ_hat[4]
    σ_α_hat = θ_hat[5]
    for j in eachindex(p)
        for m in eachindex(p[j])
            ξ[j][m] = δ[j][m] - X[j][m]'*β_hat + α_hat*p[j][m]
        end
    end
    return ξ
end

##### 2 (a) #####

# Objective function.
# Consists of 24 moments of instrumental variables.
function G(θ_hat, X, p, W, Z)
    ξ_hat = calculate_ξ(X, θ_hat, p)

    # 18 competitor characteristics moments:
    IV_comp_char = []
    for j in eachindex(p)
        for k in eachindex(p)
            if k != j
                IV1 = 0.0; IV2 = 0.0; IV3 = 0.0
                for m in eachindex(p[j])
                    IV1 += X[k][m][1]*(ξ_hat[j][m])
                    IV2 += X[k][m][2]*(ξ_hat[j][m])
                    IV3 += X[k][m][3]*(ξ_hat[j][m])
                end
                IV1 /= length(p[j]); IV2 /= length(p[j]); IV3 /= length(p[j])
                push!(IV_comp_char, IV1, IV2, IV3)
            end
        end
    end

    # 3 cost shifter W moments:
    IV_W = []
    for j in eachindex(p)
        IV = 0
        for m in eachindex(p[j])
            IV += W[j] * ξ_hat[j][m]
        end
        IV /= length(p[j])
        push!(IV_W, IV)
    end

    # 3 cost shifter Z moments:
    IV_Z = []
    for j in eachindex(p)
        IV = 0
        for m in eachindex(p[j])
            IV += Z[j][m] * ξ_hat[j][m]
        end
        IV /= length(p[j])
        push!(IV_Z, IV)
    end
    moments = append!(IV_comp_char, IV_W, IV_Z)
    return moments
end

##### 2 (b) #####

# Norm of the objective function. Takes only one input so it can be used with Optim
G_norm(θ_hat) = norm(G(θ_hat, X, p, W, Z))

# Define initial θ for optimization. Optimize.
(β1_hat, β2_hat, β3_hat, α_hat, σ_α_hat) = (5.0, 1.0, 1.0, 1.0, 1.0)
θ_initial = copy([β1_hat, β2_hat, β3_hat, α_hat, σ_α_hat])
res = optimize(θ_hat-> G_norm(θ_hat), θ_initial)
θ_hat = Optim.minimizer(res)

#################################
# Problem 2: Supply Side
#################################

##### 1 (b) #####

# Perfect Competition:
# (m = 100) x (j = 3) vector
function switch_p(p)
    p_switch = []
    if length(p) < 50
        for m in 1:100
            p_switch = append!(p_switch, [[p[1][m], p[2][m], p[3][m]]])
        end
    else
        for j in 1:3
            temp = []
            for m in 1:100
                temp = append!(temp, [p[m][j]])
            end
            p_switch = append!(p_switch, [temp])
        end
    end
    return p_switch
end
MC_comp = switch_p(p)

# Perfect 3-Way Collusion:
# scalar
function dsdp(θ_hat, p, X, j, k, m, ξ_hat)
    β = [θ_hat[1], θ_hat[2], θ_hat[3]]
    α = θ_hat[4]
    σ_α = θ_hat[5]
    ξ = ξ_hat

    ν = rand(LogNormal(0,1),1000)
    α_i = α .+ σ_α*ν
    Σ(i, m) = 1 + exp(X[1][m]'*β - α*p[1][m] + ξ[1][m] - σ_α*ν[i]*p[1][m]) + 
                  exp(X[2][m]'*β - α*p[2][m] + ξ[2][m] - σ_α*ν[i]*p[2][m]) +
                  exp(X[3][m]'*β - α*p[3][m] + ξ[3][m] - σ_α*ν[i]*p[3][m])
    
    dsdp_jm = 0
    if j == k
        for i in 1:1000
            dsdp_jm += α_i[i] * exp(X[j][m]'*β - α*p[j][m] + ξ[j][m] - σ_α*ν[i]*p[j][m]) * 
                       (Σ(i, m) - α_i[i]*exp(X[j][m]'*β - α*p[j][m] + ξ[j][m] - σ_α*ν[i]*p[j][m])) / 
                       (Σ(i, m)^2)
        end
        dsdp_jm /= 1000

    else
        for i in 1:1000
            dsdp_jm += exp(X[j][m]'*β - α*p[j][m] + ξ[j][m] - σ_α*ν[i]*p[j][m]) *
                       Σ(i, m)^(-2) * α_i[i]
        end
        dsdp_jm /= (-1000)
    end
    return dsdp_jm
end
# (m = 100) x (3 x 3 matrix) vector
function calculate_Δ(θ_hat, p, X, ξ_hat)
    Δ = []
    for m in 1:100
        one1 = - dsdp(θ_hat, p, X, 1, 1, m, ξ_hat)^(-1)
        one2 = - dsdp(θ_hat, p, X, 1, 2, m, ξ_hat)^(-1)
        one3 = - dsdp(θ_hat, p, X, 1, 3, m, ξ_hat)^(-1)
        two1 = - dsdp(θ_hat, p, X, 2, 1, m, ξ_hat)^(-1)
        two2 = - dsdp(θ_hat, p, X, 2, 2, m, ξ_hat)^(-1)
        two3 = - dsdp(θ_hat, p, X, 2, 3, m, ξ_hat)^(-1)
        thr1 = - dsdp(θ_hat, p, X, 3, 1, m, ξ_hat)^(-1)
        thr2 = - dsdp(θ_hat, p, X, 3, 2, m, ξ_hat)^(-1)
        thr3 = - dsdp(θ_hat, p, X, 3, 3, m, ξ_hat)^(-1)
        Δ = append!(Δ, [[one1 two1 thr1; one2 two2 thr2; one3 two3 thr3]])
    end
    return Δ
end
# (m = 100) x (j = 3) vector
function calculate_MC_col(p, s, θ_hat, X)
    ξ_hat = calculate_ξ(X, θ_hat, p)
    Δ = calculate_Δ(θ_hat, p ,X, ξ_hat)
    MC_col = []
    p_switch = []
    s_switch = []
    for m in 1:100
        p_switch = append!(p_switch, [[p[1][m], p[2][m], p[3][m]]])
        s_switch = append!(s_switch, [[s[1][m], s[2][m], s[3][m]]])
    end
    for m in 1:100
        MC_col = append!(MC_col, [p_switch[m] + inv(Δ[m])*s_switch[m]])
    end
    for m in 1:100
        for j in 1:3
            if MC_col[m][j] < 0
                MC_col[m][j] = 0
            end
        end
    end
    return MC_col
end
MC_col = calculate_MC_col(p, s, θ_hat, X)

# Oligopoly:
# (m = 100) x (3 x 3 matrix) vector
function calculate_Δ_again(θ_hat, p, X, ξ_hat)
    Δ = []
    for m in 1:100
        one1 = - dsdp(θ_hat, p, X, 1, 1, m, ξ_hat)^(-1)
        one2 = 0
        one3 = 0
        two1 = 0
        two2 = - dsdp(θ_hat, p, X, 2, 2, m, ξ_hat)^(-1)
        two3 = 0
        thr1 = 0
        thr2 = 0
        thr3 = - dsdp(θ_hat, p, X, 3, 3, m, ξ_hat)^(-1)
        Δ = append!(Δ, [[one1 two1 thr1; one2 two2 thr2; one3 two3 thr3]])
    end
    return Δ
end
# (m = 100) x (j = 3) vector
function calculate_MC_oli(p, s, θ_hat, X)
    ξ_hat = calculate_ξ(X, θ_hat, p)
    Δ = calculate_Δ_again(θ_hat, p ,X, ξ_hat)
    MC_oli = []
    p_switch = []
    s_switch = []
    for m in 1:100
        p_switch = append!(p_switch, [[p[1][m], p[2][m], p[3][m]]])
        s_switch = append!(s_switch, [[s[1][m], s[2][m], s[3][m]]])
    end
    for m in 1:100
        MC_oli = append!(MC_oli, [p_switch[m] + inv(Δ[m])*s_switch[m]])
    end
    for m in 1:100
        for j in 1:3
            if MC_oli[m][j] < 0
                MC_oli[m][j] = 0
            end
        end
    end
    return MC_oli
end
MC_oli = calculate_MC_oli(p, s, θ_hat, X)

# Convert to 300-element vectors for plotting
function MC_expand(MC)
    MC_expanded = []
    for m in 1:100
        for j in 1:3
            MC_expanded = append!(MC_expanded, [MC[m][j]])
        end
    end
    return MC_expanded
end
# Remove outiers to make the plot look nicer because my data is wack :)
function clean_it_up_MC(MC)
    thing = []
    for element in eachindex(MC)
        if abs(MC[element]) < 7
            thing = append!(thing, [MC[element]])
        end
    end
    return thing
end
MC_PC = clean_it_up_MC(MC_expand(MC_comp))
MC_C = clean_it_up_MC(MC_expand(MC_col))
MC_O = clean_it_up_MC(MC_expand(MC_oli))
MC_T = clean_it_up_MC(MC_expand(switch_p(MC)))

# Box Plot:
data_to_plot = [MC_PC, MC_O, MC_C, MC_T]
plotage = boxplot(["Perfect Competition ", "Oligopoly", "Collusion", "True"], data_to_plot, 
                  legend = false, xlabel="Market Structures", ylabel="Marginal Costs",
                  title="Estimated Marginal Costs of Various Market Structures", marker = false)
savefig(plotage, "C:\\Users\\wilsk\\OneDrive\\Documents\\Industrial Organization 1\\PS1\\box_plot.png")

####################################
# Problem 3: Merger
####################################

##### 1 (c) #####

# (m = 100) x (3 x 3 matrix) vector
function calculate_Δ_another_time(θ_hat, p, X, ξ_hat)
    Δ = []
    for m in 1:100
        one1 = - dsdp(θ_hat, p, X, 1, 1, m, ξ_hat)^(-1)
        one2 = - dsdp(θ_hat, p, X, 1, 2, m, ξ_hat)^(-1)
        one3 = 0
        two1 = - dsdp(θ_hat, p, X, 2, 1, m, ξ_hat)^(-1)
        two2 = - dsdp(θ_hat, p, X, 2, 2, m, ξ_hat)^(-1)
        two3 = 0
        thr1 = 0
        thr2 = 0
        thr3 = - dsdp(θ_hat, p, X, 3, 3, m, ξ_hat)^(-1)
        Δ = append!(Δ, [[one1 two1 thr1; one2 two2 thr2; one3 two3 thr3]])
    end
    return Δ
end
# (j = 3) x (m = 100) vecto
function merge(θ_hat, p, X, s)
    ξ_hat = calculate_ξ(X, θ_hat, p)
    Δ = calculate_Δ_another_time(θ_hat, p, X, ξ_hat)
    p_hat = []
    s = switch_p(s)
    for m in 1:100
        pm = MC_oli[m] - inv(Δ[m])*s[m]
        p_hat = append!(p_hat, [pm])
    end
    p_hat = switch_p(p_hat)
    return p_hat
end
p_merge = merge(θ_hat, p, X, s)

# Prepare data to make a scatter plot of post-merge minus pre-merge prices
post_merger_prices = MC_expand(switch_p(p_merge))
pre_merger_prices = MC_expand(switch_p(p))
price_diff = post_merger_prices - pre_merger_prices
function clean_it_up(price_diff)
    thing = []
    for element in eachindex(price_diff)
        if (abs(price_diff[element]) < 2) && (abs(price_diff[element]) > .01)
            thing = append!(thing, [price_diff[element]])
        end
    end
    return thing
end
cleaned = clean_it_up(price_diff)

# Scatter plot
plotty = scatter(1:length(cleaned), cleaned, xlabel="Data Points", ylabel="Price Differences",
        legend=false, title="Post-Merger Prices minus Pre-Merger Prices", 
        markershape=:circle, linewidth=2, line=:dash, color=:blue)
savefig(plotty, "C:\\Users\\wilsk\\OneDrive\\Documents\\Industrial Organization 1\\PS1\\scatter_plot.png")

        