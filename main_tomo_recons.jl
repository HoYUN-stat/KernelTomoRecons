###Main file for tomography reconstruction with Gaussian kernel###

# Load the necessary packages
using CairoMakie #v0.10.12
using LaTeXStrings #v1.3.1
using Distributions #v0.25.108
using Dates
using Random: seed!

# Load the necessary Python packages
using PyCall
py"""
import numpy as np
from skimage.transform import radon, iradon
"""

# Include the necessary Julia files
include("src/KernelTomoRecons.jl")
include("src/PhantomImage.jl")

# Use the custom modules
using .KernelTomoRecons
using .MyPhantomTestImg

# Set the global parameters
M::Int64 = 200 #Number of meshpoints
N::Int64 = 40 #Number of angles
nu::Float64 = 2^(-17) #ν
sigma::Float64 = 0.0
g1::Int64 = 2^12 #γ
λ::Int64 = 0 #0 for random angles, 1 for equiangular angles

# Load the image (M × M matrix)
img = 10 .* MyPhantomTestImg.shepp_logan(M);
#img = 10 .* MyPhantomTestImg.random_circles(M; n_circles=20);

# Set the random angle grid
if λ == 0 #Random angles
    seed!(123)
    ang = sort(rand(N))
    ang_deg = 180.0 .* ang #Angles (deg) for py"radon"
    ang_rad = pi .* ang #Angles (rad) for Julia
else    #Equiangular angles
    ang_deg = collect(range(start=0.0, step=180.0 / N, length=N)) #Angles (deg) for py"radon"
    ang_rad = collect(range(start=0.0, step=pi / N, length=N)) #Angles (rad) for Julia        
end

# Load the sinogram (M × N matrix)
sino_original = py"radon"(img, ang_deg);
seed!(123);
sino = sino_original + rand(Normal(0.0, sigma), (M, N));
sino_scaled = sino ./ div(M, 2);


# Calculate the SNR of the sinogram
SNR = sum(sino_original .^ 2) / sum((sino_original .- sino) .^ 2)
round_SNR = round(SNR; digits=2) #When M = 200, N = 40, σ = 0.0, SNR = 197.07

# Perform the FBP reconstruction
FBP_recons = py"iradon"(sino, ang_deg; filter_name="ramp");
FBP_error = img .- FBP_recons;

# Calculate the Root Mean Square Error for FBP
FBP_RMSE = sqrt(sum(FBP_error .^ 2) / (M^2 * π / 4));
round_FBP_RMSE = round(FBP_RMSE; digits=2) #When M = 100, N = 40, σ = 20.0, RMSE = 2.45

# Plot the FBP reconstruction
function doit2()
    fig = Figure(resolution=(1200, 1000), fontsize=25)
    ax1 = Axis(fig[1, 1], title="True Image")
    p1 = heatmap!(ax1, img', colormap=:greys)
    Colorbar(fig[1, 2], p1)

    #Sinogram
    ax2 = Axis(fig[1, 3], title="Sinogram (SNR = $round_SNR)", xlabel="Angle (deg)", ylabel="Projection position")
    p2 = heatmap!(ax2, ang_deg, 1:M, sino_scaled', colormap=:greys)
    #vlines on the borders
    vlines!(ax2, [0.5 * (ang_deg[i] + ang_deg[i+1]) for i in 1:N-1], color=:red, linewidth=2)
    Colorbar(fig[1, 4], p2)

    #FBP
    ax3 = Axis(fig[2, 1], title="FBP reconstruction")
    p3 = heatmap!(ax3, FBP_recons', colormap=:greys)
    Colorbar(fig[2, 2], p3)

    ax4 = Axis(fig[2, 3], title="FBP error (RMSE = $round_FBP_RMSE)")
    p4 = heatmap!(ax4, FBP_error', colormap=:greys)
    Colorbar(fig[2, 4], p4)

    fig
end
doit2()

# Perform the KR reconstruction with the Gaussian kernel
@time KR_recons = gker_recons(sino_scaled, ang_rad, g1, nu);

KR_error = img .- KR_recons;
KR_RMSE = sqrt(sum(KR_error .^ 2) / (M^2 * π / 4)); #Root Mean Square Error for KR
round_KR_RMSE = round(KR_RMSE; digits=2)

# Plot the image, sinogram, FBP reconstruction, FBP error, KR reconstruction, and KR error
function doit3()
    fig = Figure(resolution=(1200, 1200), fontsize=25)
    ax1 = Axis(fig[1, 1], title="True Image")
    p1 = heatmap!(ax1, img', colormap=:greys)
    Colorbar(fig[1, 2], p1)

    #Sinogram
    ax2 = Axis(fig[1, 3], title="Sinogram (SNR = $round_SNR)", xlabel="Angle (deg)", ylabel="Projection position")
    p2 = heatmap!(ax2, ang_deg, 1:M, sino_scaled', colormap=:greys)
    #vlines on the borders
    vlines!(ax2, [0.5 * (ang_deg[i] + ang_deg[i+1]) for i in 1:N-1], color=:red, linewidth=2)
    Colorbar(fig[1, 4], p2)

    #FBP
    ax3 = Axis(fig[2, 1], title="FBP")
    p3 = heatmap!(ax3, FBP_recons', colormap=:greys)
    Colorbar(fig[2, 2], p3)
    ax4 = Axis(fig[2, 3], title="FBP error (RMSE = $round_FBP_RMSE)")
    p4 = heatmap!(ax4, FBP_error', colormap=:greys)
    Colorbar(fig[2, 4], p4)

    #KR
    log2_nu = log2(nu)
    ax5 = Axis(fig[3, 1], title="KR (γ = $g1, log₂ν = $log2_nu)")
    p5 = heatmap!(ax5, KR_recons', colormap=:greys)
    Colorbar(fig[3, 2], p5)
    ax6 = Axis(fig[3, 3], title="KR error (RMSE = $round_KR_RMSE)")
    p6 = heatmap!(ax6, KR_error', colormap=:greys)
    Colorbar(fig[3, 4], p6)

    fig

    # filename_title = "_makie_Recons_l,,$λ,M,,$M,N,,$N,sig,,$sigma).png"
    # file_folder = "simul/"
    # file_date = Dates.format(now(), "yymmddHHMM")
    # CairoMakie.save(file_folder * file_date * filename_title, fig)
end

doit3()
