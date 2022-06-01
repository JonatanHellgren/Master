using Flux
using Random
using LinearAlgebra
using StatsBase
using Formatting
using StaticArrays
using JLD2

include("StochicWorld.jl")

@load "critic_ppo" 
critic_ppo = critic_ppo |> gpu
