export SwimmerEnv
# using ReinforcementLearning
# using Random
# using StableRNGs
using IntervalSets
"""
Minimum to setup for the SwimmerEnv
action_space(env::YourEnv)
state(env::YourEnv)
state_space(env::YourEnv)
reward(env::YourEnv)
is_terminated(env::YourEnv)
reset!(env::YourEnv)
(env::YourEnv)(action)
"""

include("..\\src\\FiniteDipole.jl")
struct SwimmerParams{T}
    #parameters for specific set of swimming agents
    ℓ ::T#dipole length - vortex to vortex
    Γ0::T #init circulation
    Γa::T #incremental circulation
    v0::T #cruise velocity
    va::T #incremental velocity
    ρ::T  #turn radius    
    Γt ::T#turning circ change
    wa ::T    # Reward values 
    wd ::T
    D ::T # max distance state ; \theta is boring - just cut up 2π
    Δt :: T
end 
#Default constructor uses
SwimmerParams(ℓ,T) = SwimmerParams(ℓ/2,   #ℓ
                                convert(T,10*π*ℓ^2), #Γ0
                                π*ℓ^2,    #Γa
                                5*ℓ,      #v0
                                ℓ/2,    #va
                                10ℓ,      #ρ
                                convert(T,sqrt(2π^3)*ℓ^2), #Γt,
                                convert(T,0.1), #wa
                                convert(T,0.9), #wd
                                convert(T,2*ℓ*sqrt(2π)),
                                convert(T, 0.1))

mutable struct SwimmingEnv{A,T,R} <: AbstractEnv
    params::SwimmerParams{T}
    action_space::A
    action::T
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    done::Bool
    t::Int
    max_steps::Int
    rng::R
    reward::T
    rewards
    n_actions::Int
    swimmer::FD_agent{T}
    target::Vector{T}
    # swimmers::Vector{FD_agent{T}} #multiple swimmers 
end


function SwimmerEnv(; T = Float32,  max_steps = 200, continuous::Bool = false, n_actions::Int = 5, rng = Random.GLOBAL_RNG, ℓ=5e-4)
    # high = T.([1, 1, max_speed])
    ℓ = convert(T,ℓ)
    action_space = Base.OneTo(n_actions) # A = [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    swim =  SwimmerParams(ℓ,T)
    ηs = [0,-1,1,-1,-1] # [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    env = SwimmingEnv(swim,
        action_space,
        0|>T,
        Space(ClosedInterval{T}.([0,0], [20*ℓ*sqrt(2π),2π])), #obs(radius, angle) = [0->10D][0->2π]
        zeros(T, 2), #r,θ to target
        false,
        0,
        max_steps,
        rng,
        zero(T),
        ηs,
        n_actions,
        FD_agent(Vector{T}([0.0,0.0]),Vector{T}([swim.Γ0,-swim.Γ0]), T(π/2), Vector{T}([0.0,0.0]),Vector{T}([0.0,0.0])), 
        Vector{T}([0.0, 5.0 * swim.D])
    )
    reset!(env)
    env
end

Random.seed!(env::SwimmingEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::SwimmingEnv) = env.action_space
RLBase.state_space(env::SwimmingEnv) = env.observation_space
RLBase.reward(env::SwimmingEnv) = env.reward
RLBase.is_terminated(env::SwimmingEnv) = isapprox(env.state[1], 0.0, atol=env.params.ℓ/4.) || env.done #within a quarter of a fish?

function RLBase.state(env::SwimmingEnv{A,T}) where {A,T} 
    dn,θn =  dist_angle(env.swimmer,env.target)
    #39 is a magic number?????
    [clamp(dn, 0, 39.99*env.params.ℓ*sqrt(2π))|>T, clamp(θn, 0 , 2π)|>T ]
end

function RLBase.reset!(env::SwimmingEnv{A,T}) where {A,T}
    #go back to the original state, not random?
    env.state[1] = (rand(env.rng, T)) *10*env.params.D
    env.state[2] = 2π * (rand(env.rng, T) )
    env.action = one(T)
    env.t = 0
    env.done = false
    env.reward = zero(T)
    nothing
end

function (env::SwimmingEnv)(a::Union{Int, AbstractFloat})
    @assert a in env.action_space
    env.action = change_circulation!(env, a)
    _step!(env, env.action)
end

function _step!(env::SwimmingEnv, a)
    env.t += 1
    r, th = env.state
   
    
    v2v = vortex_to_vortex_velocity([env.swimmer];env.params.ℓ)   
    avg_v = add_lr(v2v) 
    siv = self_induced_velocity([env.swimmer];env.params.ℓ)
    angles = angle_projection([env.swimmer],v2v)
    ind_v = siv + avg_v #eqn 2.a
    for (i,b) in enumerate([env.swimmer])
        # b.v = ind_v[:,i]
        b.position += ind_v[:,i] .* env.params.Δt
        b.angle += angles[i] .* env.params.Δt #eqn 2.b
    end
    dn,θn = dist_angle(env.swimmer, env.target)
    # NATE : TODO ADD TARGET MOTION HERE - below is a crappy circle 
    # path(x,y) = env.params.ℓ.*[cos(x),sin(y)]
    # a = path.(range(0,2pi,env.max_steps),range(0,2pi,env.max_steps))
    # env.target = a[env.t]

    env.state[1] = clamp(dn, 0, 39.99*env.params.ℓ*sqrt(2π))
    env.state[2] = clamp(θn, 0 , 2π)        

    
    costs = env.params.wa*(1- dn/env.params.ℓ) + env.rewards[Int(a)]*env.params.wd 
    env.done = env.t >= env.max_steps
    env.reward = -costs
    nothing
end

function dist_angle(agent::FD_agent, target)
    """ 
    Grab the distance and angle between an agent and the target

    ```jldoctest 
    julia> b = boid([1.0,0.5],[-1.0,1.0], π/2, [0.0,0.0],[0.0,0.0])
    julia> target = (b.position[1]-1.0,b.position[2] )
    julia> distance_angle_between(b,target)
    (1.0, 3.141592653589793)
    julia> target = (b.position[1]+1.0,b.position[2] )
    julia> distance_angle_between(b,target)
    (1.0, 0.0)
    
    """
    d =  target .- agent.position
    sqrt(sum(abs2,d)), atan(d[2],d[1])-agent.angle
    
end

function change_circulation!(env::SwimmingEnv{<:Base.OneTo}, a::Int)
    if a == 1
        nothing
    elseif a == 2   
        env.swimmer.gamma += [env.params.Γa , env.params.Γa]
    elseif a == 3  
        env.swimmer.gamma += [-env.params.Γa, -env.params.Γa]
    elseif a == 4  
        env.swimmer.gamma += [env.params.Γt, -env.params.Γt]
    elseif a == 5   
        env.swimmer.gamma += [-env.params.Γt, env.params.Γt]
    else 
        @error "unknown action of $action"
    end 
    a
end
env = SwimmerEnv()
RLBase.test_runnable!(env)
