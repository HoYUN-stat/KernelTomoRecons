module KernelTomoRecons
# export unicdf   #CDF of N(0, 1)
# export bvncdf   #CDF of bivariate standard normal distribution
# export antid_erf   #Exact antiderivative of erf
# export gker   #2D Gaussian kernel
# export gind_ker   #Induced 1D Gaussian kernel
# export ginn_pr   #Inner product of the backprojection
# export gweight   #Weight matrix
export thrd_gweight   #Weight matrix with multi-threading : Useful for Cross-validation
#export gridge_coefs   #Ridge regression coefficients
#export thrd_gridge_coefs   #Ridge regression coefficients with multi-threading
export thrd_gbckproj   #Gaussian backprojection with multi-threading : Useful for Cross-validation
export gker_recons   #Gaussian kernel reconstruction

export shepp_logan  #Shepp-Logan Phantom
export random_circles #Random Circles inside a unit ball

## Package loading
using LinearAlgebra
using SpecialFunctions: erf #v2.3.1
using LoopVectorization #v0.12.170
using Distributions

#using Random: seed!

## Shepp_Logan Phantom (Deleted from Images.jl)
# Initially proposed in Shepp, Larry; B. F. Logan (1974).
# "The Fourier Reconstruction of a Head Section". IEEE Transactions on Nuclear Science. NS-21.
"""
    ```
        phantom = shepp_logan(N,[M]; highContrast=true)
    ```
    Output the NxM Shepp-Logan phantom, which is a benchmark image in the field of CT and MRI.
    If the argument M is omitted, the phantom is of size NxN. When setting the keyword argument
    `highConstrast` to false, the CT version of the phantom is created. 
    Otherwise, the high contrast MRI version is calculated.

    # Examples
    ```julia-repl
    julia> shepp_logan(5, 7)
    5×7 Matrix{Float64}:
    0.0  0.0   0.0          0.0   0.0          0.0  0.0
    0.0  0.0   0.2          0.3   0.2          0.0  0.0
    0.0  1.0  -5.55112e-17  0.2  -5.55112e-17  1.0  0.0
    0.0  0.0   0.2          0.2   0.2          0.0  0.0
    0.0  0.0   0.0          0.0   0.0          0.0  0.0
"""
function shepp_logan(M::Int64, N::Int64; highContrast=true)

    P = zeros(M, N)

    x = ((0:(N-1)) / div(N, 2) .- 1)'
    y = ((0:(M-1)) / div(M, 2) .- 1)

    centerX = [0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06]
    centerY = [0, -0.0184, 0, 0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605]
    majorAxis = [0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 0.046, 0.046, 0.023, 0.023]
    minorAxis = [0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 0.046, 0.023, 0.023, 0.046]
    theta = [0, 0, -18.0, 18.0, 0, 0, 0, 0, 0, 0]

    # original (CT) version of the phantom
    grayLevel = [2, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    if (highContrast)
        # high contrast (MRI) version of the phantom
        grayLevel = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    end

    for l in eachindex(theta)
        P += grayLevel[l] * (
            ((cos(theta[l] / 360 * 2 * pi) * (x .- centerX[l]) .+
              sin(theta[l] / 360 * 2 * pi) * (y .- centerY[l])) / majorAxis[l]) .^ 2 .+
            ((sin(theta[l] / 360 * 2 * pi) * (x .- centerX[l]) .-
              cos(theta[l] / 360 * 2 * pi) * (y .- centerY[l])) / minorAxis[l]) .^ 2 .< 1)
    end

    return P
end

shepp_logan(N::Int64; highContrast=true) = shepp_logan(N, N; highContrast=highContrast)

#Create Phantom Image with Random Circles inside a unit ball with raidus 1
function random_circles(M::Int64, N::Int64; n_circles::Int64=10)
    P = zeros(M, N)
    x = ((0:(N-1)) / div(N, 2) .- 1)'
    y = ((0:(M-1)) / div(M, 2) .- 1)

    for i in 1:n_circles
        #Centroid of ellipses with centerX^2 + centerY^2 <1
        R = rand()
        α = rand(-180:180)
        centerX = R * cos(α)
        centerY = R * sin(α)

        #The contour of each ellipse is given by
        # [x, y] ↦ [z,w] = [cos(theta) sin(theta) ; -sin(theta) cos(theta)] * diag(1/majorAxis, 1/minorAxis) * [x - centerX, y - centerY]
        # The inverse is given by
        # [z, w] ↦ [x, y] = diag(majorAxis, minorAxis) * [cos(theta) -sin(theta) ; sin(theta) cos(theta)] * [z, w] + [centerX, centerY]
        # To ensure that this lies in the unit ball, we need
        # maximum(majorAxis, minorAxis) + sqrt(centerX^2 + centerY^2) < 1, or equivalently,
        # majorAxis, minorAxis < 1 - sqrt(centerX^2 + centerY^2)

        #Shape of ellipses inside a unit ball
        majorAxis = rand() * (1 - R)
        minorAxis = rand() * (1 - R)
        #Orientation (deg) of ellipses
        theta = rand(-180:180)
        #Gray Level of ellipses
        grayLevel = rand(-1:0.1:1)
        P += grayLevel * (
            ((cos(theta / 360 * 2 * pi) * (x .- centerX) .+
              sin(theta / 360 * 2 * pi) * (y .- centerY)) / majorAxis) .^ 2 .+
            ((sin(theta / 360 * 2 * pi) * (x .- centerX) .-
              cos(theta / 360 * 2 * pi) * (y .- centerY)) / minorAxis) .^ 2 .< 1)
    end
    P = P ./ maximum(abs.(P))
    return P
end

random_circles(N::Int64; n_circles::Int64=10) = random_circles(N, N; n_circles=n_circles)


### In the Python package `radon`, the centered mesh of length M is given by
# mesh = collect((0:(M-1)) / div(M, 2) .- 1) 
# mesh[j] = -1 + (j-1) / div(M, 2)

## Define Special Functions
unicdf(x::Float64) = 0.5 * (1 + erf(x / sqrt(2))) #CDF of N(0, 1)

const c1 = -1.0950081470333
const c2 = -0.75651138383854

"""
        bvncdf(p::Float64, q::Float64, ρ::Float64)

    Output the CDF of the bivariate standard normal distribution with correlation ρ using the error function.

    # Examples
    ```
    julia> 10^5 * bvncdf(-2.0, -2.0 , 0.0)
    51.75685036595643
    ```

    # Reference
    Tsay, Wen-Jen, and Peng-Hsuan Ke, A simple approximation for the bivariate normal integral (2021)
"""
function bvncdf(p::Float64, q::Float64, ρ::Float64)::Float64
    a = -ρ / sqrt(1 - ρ^2)
    b = p / sqrt(1 - ρ^2)
    if a > 0
        if a * q + b ≥ 0
            cdf = 0.5 * (erf(q / sqrt(2)) + erf(b / (sqrt(2) * a))) +
                  0.25 / sqrt(1 - a^2 * c2) * exp((a^2 * c1^2 - 2 * sqrt(2) * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2))) *
                  (1 - erf((sqrt(2) * b - a^2 * c1) / (2 * a * sqrt(1 - a^2 * c2)))) -
                  0.25 / sqrt(1 - a^2 * c2) * exp((a^2 * c1^2 + 2 * sqrt(2) * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2))) *
                  (erf((sqrt(2) * q - sqrt(2) * a^2 * c2 * q - sqrt(2) * a * b * c2 - a * c1) / (2 * sqrt(1 - a^2 * c2))) +
                   erf((a^2 * c1 + sqrt(2) * b) / (2 * a * sqrt(1 - a^2 * c2))))
        else
            cdf = 0.25 / sqrt(1 - a^2 * c2) * exp((a^2 * c1^2 - 2 * sqrt(2) * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2))) *
                  (1 + erf((sqrt(2) * q - sqrt(2) * a^2 * c2 * q - sqrt(2) * a * b * c2 + a * c1) / (2 * sqrt(1 - a^2 * c2))))
        end
    elseif a == 0
        cdf = unicdf(p) * unicdf(q)
    else
        if a * q + b ≥ 0
            cdf = 0.5 + 0.5 * erf(q / sqrt(2)) - 0.25 / sqrt(1 - a^2 * c2) * exp((a^2 * c1^2 + 2 * sqrt(2) * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2))) *
                                                 (1 + erf((sqrt(2) * q - sqrt(2) * a^2 * c2 * q - sqrt(2) * a * b * c2 - a * c1) / (2 * sqrt(1 - a^2 * c2))))
        else
            cdf = 0.5 - 0.5 * erf(b / (sqrt(2) * a)) -
                  0.25 / sqrt(1 - a^2 * c2) * exp((a^2 * c1^2 + 2 * sqrt(2) * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2))) *
                  (1 - erf((sqrt(2) * b + a^2 * c1) / (2 * a * sqrt(1 - a^2 * c2)))) +
                  0.25 / sqrt(1 - a^2 * c2) * exp((a^2 * c1^2 - 2 * sqrt(2) * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2))) *
                  (erf((sqrt(2) * q - sqrt(2) * a^2 * c2 * q - sqrt(2) * a * b * c2 + a * c1) / (2 * sqrt(1 - a^2 * c2))) +
                   erf((-a^2 * c1 + sqrt(2) * b) / (2 * a * sqrt(1 - a^2 * c2))))
        end
    end
    return cdf
end


"""
        antid_erf(z::Float64)

    Compute the exact antiderivative F(z) of f(z) = √π * erf with F(0) = ℯ⁻¹, ``F(z) = √π * z * erf(z) + exp(- z²)``.

    # Examples
    ```
    julia> antid_erf(2.0)
    3.5466412019384204
    ```
"""
function antid_erf(z::Float64)::Float64
    sqrt(π) * z * erf(z) + exp(-z^2)
end

"""
        gker(z1x::Float64, z1y::Float64, z2x::Float64, z2y::Float64, γ::Int64)

    Compute the 2D Guassian kernel with parameter γ at z1 = [z1x, z1y] and z2 = [z2x, z2y], ``K(z1, z2 ; γ) = exp(-γ ‖z1-z2‖^2)``, from [Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).

    # Examples
    ```julia-repl
    julia> gker(.1, .3, .2, .8, 20)
    0.0055165644207607716
```
"""
function gker(z1x::Float64, z1y::Float64, z2x::Float64, z2y::Float64, γ::Int64)::Float64
    dist = (z1x - z2x)^2 + (z1y - z2y)^2
    return exp(-γ * dist)
end

## Compute the Gram Matrix
# Diagonal elements
"""
        gind_ker(x1::Float64, x2::Float64, γ::Int64)

    Compute the induced 1D Guassian kernel K̃ (x1, x2 ; γ) with parameter γ at x1 ∈ [-1, 1] and x2 ∈ [-1, 1].

    # Examples
    ```julia-repl
    julia> gind_ker(.7, .7, 20)
    0.5160763646948695
    ```
"""
function gind_ker(x1::Float64, x2::Float64, γ::Int64)::Float64
    ix1 = sqrt(γ * (1 - x1^2))
    ix2 = sqrt(γ * (1 - x2^2))
    K = (antid_erf(ix1 + ix2) - antid_erf(ix1 - ix2) - antid_erf(-ix1 + ix2) + antid_erf(-ix1 - ix2))
    return 0.5 * exp(-γ * (x1 - x2)^2) * K / γ
end

#Off-diagonal elements
"""
        ginn_pr(x1::Float64, x2::Float64, ϕ1::Float64, ϕ2::Float64, γ::Int64)

    Compute the inner product between the backprojection at angles ϕ1, ϕ2 of the induced generators at x1, x2 ∈ [-1, 1],
    ``<P*_{ϕ1}(k̃ _{x1}),P*_{ϕ2}(k̃_{x2})> ``.

    # **Warning**: Use only when ϕ1 ≠ ϕ2. If ϕ1=ϕ2, use gind_ker(x1, x2, γ), as ``ginn_pr(x1, x2, ϕ1, ϕ1, γ) = gind_ker(x1, x2, γ)``.

    # Examples
    ```julia-repl
    julia> ginn_pr(0.5, 0.3, 0.0, 0.1, 20)
    0.3079567702892215

    julia> ginn_pr(0.5, 0.3, 0.0, 0.0, 20) 
    NaN
    ```
"""
function ginn_pr(x1::Float64, x2::Float64, ϕ1::Float64, ϕ2::Float64, γ::Int64)::Float64
    ϕ = ϕ1 - ϕ2
    c = cos(ϕ)
    s = sin(ϕ)
    #if isapprox(s, 0, atol=eps(Float64))
    #    return gind_ker(c * x1, x2, γ)
    #else
    mu1 = c * x1 - x2
    mu2 = x1 - c * x2
    ix1 = sqrt(1 - x1^2)
    ix2 = sqrt(1 - x2^2)
    PPk = (bvncdf(sqrt(2 * γ) * (s * ix1 - mu1), sqrt(2 * γ) * (s * ix2 - mu2), c)
           -
           bvncdf(sqrt(2 * γ) * (-s * ix1 - mu1), sqrt(2 * γ) * (s * ix2 - mu2), c)
           -
           bvncdf(sqrt(2 * γ) * (s * ix1 - mu1), sqrt(2 * γ) * (-s * ix2 - mu2), c)
           +
           bvncdf(sqrt(2 * γ) * (-s * ix1 - mu1), sqrt(2 * γ) * (-s * ix2 - mu2), c))
    return pi * PPk / (γ * abs(s))
    #end
end

#Weight Matrix
"""
        gweight(M::Int64, Φ::Vector{Float64}, γ::Int64)

    For the mesh X = collect((0:(M-1)) / div(M, 2) .- 1), compute the weight matrix W given by the mesh grid X ⊂ [-1, +1] and angle grid Φ, 
    ``W[j1+M*(i1-1), j2+M*(i2-1)]=<P*_{Φ[i1]}(k̃_{X[j1]}),P*_{Φ[i2]}(k̃_{X[j2]})>``.
    
    Return the SPD matrix.

    # Examples
    ```julia-repl
    julia> gweight(4, [0, π/3], 20)
    8×8 Symmetric{Float64, Matrix{Float64}}:
    0.0  0.0         0.0         0.0         0.0  0.0         0.0         0.0
    0.0  0.636468    0.00453207  1.31186e-9  0.0  0.181074    0.179383    0.0604274
    0.0  0.00453207  0.742665    0.00453207  0.0  0.179332    0.18138     0.179332
    0.0  1.31186e-9  0.00453207  0.636468    0.0  0.0604274   0.179383    0.181074
    0.0  0.0         0.0         0.0         0.0  0.0         0.0         0.0
    0.0  0.181074    0.179332    0.0604274   0.0  0.636468    0.00453207  1.31186e-9
    0.0  0.179383    0.18138     0.179383    0.0  0.00453207  0.742665    0.00453207
    0.0  0.0604274   0.179332    0.181074    0.0  1.31186e-9  0.00453207  0.636468
    ```
"""
function gweight(M::Int64, Φ::Vector{Float64}, γ::Int64)::Symmetric{Float64,Matrix{Float64}}
    #mesh = collect((0:(M-1)) / div(M, 2) .- 1)  #Grid that matches python radon package
    N = length(Φ)
    W = zeros(M * N, M * N)
    #Upper triangular part: M*(i1-1)+j1 ≤ M*(i2-1)+j2 ⟺ (i1 < i2) || ((i1==i2) && j1 ≤ j2)
    for j2 ∈ 1:M
        xj2 = -1 + (j2 - 1) / div(M, 2)
        for j1 ∈ 1:M
            xj1 = -1 + (j1 - 1) / div(M, 2)
            diag = gind_ker(xj1, xj2, γ)
            for i2 ∈ 1:N
                for i1 ∈ 1:(i2-1)
                    ϕ = Φ[i1] - Φ[i2]
                    @inbounds W[j1+M*(i1-1), j2+M*(i2-1)] = ginn_pr(xj1, xj2, ϕ, 0.0, γ)
                end
                @inbounds W[j1+M*(i2-1), j2+M*(i2-1)] = diag
            end
        end
    end
    return Symmetric(W)
end

"""
        thrd_gweight(M::Int64, Φ::Vector{Float64}, γ::Int64)

    Variant of `gweight()` with multi-threading and simd macros. Set an environment variable a priori, such as export `JULIA_NUM_THREADS=4`.
    Check the number of threads with `Threads.nthreads()`. To change the number of threads, use `Threads.nthreads(4)`.
    # Examples
    ```julia-repl
    julia> M = 100 #Number of meshpoints
        N = 20 #Number of orientations# Meshpoints {x₁, x₂, …, xₘ}
        ang_rad = collect(range(start=0.0, step=pi/N, length=N)); #Angles (rad) for Julia
                
        @time gram = gweight(M, ang_rad, 20);
        @time thrd_gram = thrd_gweight(M, ang_rad, 20);
        gram == thrd_gram
    0.429915 seconds (3 allocations: 30.518 MiB, 1.71% gc time)
    0.123405 seconds (41 allocations: 30.521 MiB)
    true
    ```
"""
function thrd_gweight(M::Int64, Φ::Vector{Float64}, γ::Int64)::Symmetric{Float64,Matrix{Float64}}
    #mesh = collect((0:(M-1)) / div(M, 2) .- 1)  #Grid that matches python radon package
    N = length(Φ)
    W = zeros(M * N, M * N)
    #Upper triangular part: M*(i1-1)+j1 ≤ M*(i2-1)+j2 ⟺ (i1 < i2) || ((i1==i2) && j1 ≤ j2)
    Threads.@threads for j2 ∈ 1:M
        xj2 = -1 + (j2 - 1) / div(M, 2)
        for j1 ∈ 1:M
            xj1 = -1 + (j1 - 1) / div(M, 2)
            diag = gind_ker(xj1, xj2, γ)
            @simd for i2 ∈ 1:N
                for i1 ∈ 1:(i2-1)
                    ϕ = Φ[i1] - Φ[i2]
                    @inbounds W[j1+M*(i1-1), j2+M*(i2-1)] = ginn_pr(xj1, xj2, ϕ, 0.0, γ)
                end
                @inbounds W[j1+M*(i2-1), j2+M*(i2-1)] = diag
            end
        end
    end
    return Symmetric(W)
end


"""
        gridge_coefs(sino::Matrix{Float64}, W::Symmetric{Float64,Matrix{Float64}}, ν::Float64)

    Provided the weight matrix W, compute the ridge regression coefficients for a sinogram : Useful for Cross-validation

    # Examples
    ```julia-repl
    julia> M = 5; N = 4;
    range_ang_deg = range(start=0.0, step=180.0 / N, length=N);
    ang_deg = collect(range_ang_deg); #Angles (deg) for py"radon"
    ang_rad = collect(range(start=0.0, step=pi / N, length=N)); #Angles (rad) for Julia
    W = thrd_gweight(M, ang_rad, 20);
    
    sino=py"radon"(10 .* shepp_logan(M),ang_deg); #Sinogram of the image.
    alpha1 = gridge_coefs(sino, W, 1.0);
    alpha2 = gridge_coefs(sino, ang_rad, 20, 1.0);
    alpha1 == alpha2
    true
    ```
"""
function gridge_coefs(sino::Matrix{Float64}, W::Symmetric{Float64,Matrix{Float64}}, ν::Float64)::Vector{Float64}
    M, N = size(sino)
    Y = reshape(sino, M * N) ### INDEXING: Y[j + (i-1) * M] = sino[j,i] COLUMN MAJOR ORDERING
    W += ν * I
    alp = \(W, Y)
    return alp
end

"""
        thrd_gridge_coefs(sino::Matrix{Float64}, Φ::Vector{Float64}, γ::Int64, ν::Float64)

    Variant of `gridge_coefs()` with multi-threading and simd macros. Set an environment variable a priori, such as export `JULIA_NUM_THREADS=4`.

    # Examples
    ```julia-repl
    julia> M = 100; N = 80;
    range_ang_deg = range(start=0.0, step=180.0 / N, length=N);
    ang_deg = collect(range_ang_deg); #Angles (deg) for py"radon"
    ang_rad = collect(range(start=0.0, step=pi / N, length=N)); #Angles (rad) for Julia
    sino=py"radon"(10 .* shepp_logan(M),ang_deg); #Sinogram of the image.
    @time alpha1 = gridge_coefs(sino, ang_rad, 20, 1.0);
    @time alpha2 = thrd_gridge_coefs(sino, ang_rad, 20, 1.0);
    alpha1 == alpha2
    8.784304 seconds (20 allocations: 1.434 GiB, 0.20% gc time)
    4.228911 seconds (52 allocations: 1.434 GiB, 0.69% gc time)
    true
    ```
"""
function thrd_gridge_coefs(sino::Matrix{Float64}, Φ::Vector{Float64}, γ::Int64, ν::Float64)::Vector{Float64}
    M, N = size(sino)
    Y = reshape(sino, M * N) ### INDEXING: Y[M*(i-1)+j]=sino[j,i] COLUMN MAJOR ORDERING
    W = thrd_gweight(M, Φ, γ)
    W += ν * I
    alp = \(W, Y)
    return alp
end


## Gaussian Backprojection
"""
        thrd_gbckproj(M::Int64, Φ::Vector{Float64}, γ::Int64) # M × M × N*M array

    Gaussian backprojection when applied to vectors for Φ and mesh = collect((0:(M-1)) / div(M, 2) .- 1), 
        
        thrd_gbckproj(M, Φ, γ)[k1, k2, M*(i-1)+j] = P^{*}_{Φ[i]}(k̃_{X[j]})(mesh[k1], mesh[k2]).

    # Examples
    ```julia-repl
    julia> M = 100; N = 80;
       range_ang_deg = range(start=0.0, step=180.0 / N, length=N);
       ang_rad = collect(range(start=0.0, step=pi / N, length=N)); #Angles (rad) for Julia
       @time back = thrd_gbckproj(M, ang_rad, 20);
    0.709285 seconds (39 allocations: 610.355 MiB, 0.47% gc time)
    ```
"""
function thrd_gbckproj(M::Int64, Φ::Vector{Float64}, γ::Int64)::Array{Float64,3}
    N = length(Φ)
    backproj = zeros(M, M, M * N)
    Threads.@threads for l in axes(backproj, 3)
        #l-1 == (i-1) * M + (j-1) = div(l-1, M) * M + rem(l-1, M)
        i, j = divrem(l - 1, M) .+ 1
        c = cos(Φ[i])
        s = sin(Φ[i])
        x = -1 + (j - 1) / div(M, 2)
        for k1 in axes(backproj, 2)
            zx = -1 + (k1 - 1) / div(M, 2)
            @simd for k2 in axes(backproj, 1)
                zy = -1 + (k2 - 1) / div(M, 2)
                μ1 = c * zx - s * zy
                μ2 = s * zx + c * zy
                Pk = (zx^2 + zy^2 < 1) * (erf(sqrt(γ) * (sqrt(1 - x^2) - μ2)) - erf(sqrt(γ) * (-sqrt(1 - x^2) - μ2)))
                @inbounds backproj[k2, k1, l] = 0.5 * exp(-γ * (x - μ1)^2) * sqrt(pi / γ) * Pk
            end
        end
    end
    return backproj
end


## Gaussian Tomographic Reconstruction
"""
        gker_recons(sino::Matrix{Float64}, Φ::Vector{Float64}, γ::Int64, ν::Float64)

    Gaussian kernel reconstruction from a sinogram at the angle grid Φ, ``f̂(z₁, z₂)=∑ᵢ,ⱼ αᵢⱼ P^{*}_{ϕᵢ}(k̃_{xⱼ})(z₁, z₂)``, i.e. ``gker_recons[j1, j2]=∑ᵢ,ⱼ α[M*(i-1)+j] * gadj_ind_ker(mesh[j1], mesh[j2], Φ[i], X[j], γ)``.

    # Arguments
    - `sino::Matrix{Float64}`: the sinogram of an image.
    - `Φ::Vector{Float64}`: the angle grid where the sinogram is evaluated.
    - `γ::Int64`: the parameter of the Gaussian kernel.
    - `ν::Float64`: the parameter for the ridge regression.
"""
function gker_recons(sino::Matrix{Float64}, Φ::Vector{Float64}, γ::Int64, ν::Float64)::Matrix{Float64}
    M, N = size(sino)
    alp = thrd_gridge_coefs(sino, Φ, γ, ν) # MN x 1 vector
    back = thrd_gbckproj(M, Φ, γ)  # M x M x MN Array
    back_reshaped = reshape(back, :, M * N) # M^2 x MN matrix
    recons = back_reshaped * alp    # M^2 x 1 vector
    return reshape(recons, M, M)    # M x M matrix
end

"""
    gker_recons(sino::Matrix{Float64}, back::Array{Float64,3}, W::Symmetric{Float64,Matrix{Float64}}, ν::Float64)

    Provided the weight matrix W and the backprojection 3D array, compute the Gaussian kernel reconstruction from a sinogram: Useful for Cross-validation.

    # Examples
    ```julia-repl
    julia> M = 200; N = 40; γ = 200;
       ang_deg = collect(range(start=0.0, step=180.0/N, length=N));
       ang_rad = pi ./180.0 .*ang_deg;
       @time W = thrd_gweight(M, ang_rad, γ);
       @time back = thrd_gbckproj(M, ang_rad, γ);
    1.663110 seconds (42 allocations: 488.285 MiB, 6.73% gc time)
    2.168664 seconds (41 allocations: 2.384 GiB, 0.14% gc time)

    julia> img = 10 .* shepp_logan(M); #Image of which to generate a sinogram.
        sino=py"radon"(img, ang_deg);
        sino_scaled=sino .*2 ./M;
        @time recons1 = gker_recons(sino_scaled, ang_rad, γ, .1);
        @time recons2 = gker_recons(sino_scaled, back, W, .1);
    5.886421 seconds (107 allocations: 3.819 GiB, 10.99% gc time)
    2.266546 seconds (18 allocations: 980.897 MiB, 0.07% gc time)
    ```
"""
function gker_recons(sino::Matrix{Float64}, back::Array{Float64,3}, W::Symmetric{Float64,Matrix{Float64}}, ν::Float64)::Matrix{Float64}
    M, N = size(sino)
    alp = gridge_coefs(sino, W, ν) # MN x 1 vector
    back_reshaped = reshape(back, :, M * N) # M^2 x MN matrix
    recons = back_reshaped * alp    # M^2 x 1 vector
    return reshape(recons, M, M)    # M x M matrix
end

end #module