export SwimmerEnv
using ReinforcementLearning
using Random
using StableRNGs
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
    ℓ ::T #dipole length - vortex to vortex
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


function SwimmerEnv(; T = Float32,  max_steps = 500, n_actions::Int = 5, rng = Random.GLOBAL_RNG, ℓ=5e-4, target=[5.0,0.0])
    # high = T.([1, 1, max_speed])
    ℓ = convert(T,ℓ)
    action_space = Base.OneTo(n_actions) # A = [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    ηs = [1, -1, 1, -1, -1] # [CRUISE, FASTER, SLOWER, LEFT, RIGHT]
    swim =  SwimmerParams(ℓ,T)
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
        # Vector{T}([-5.0 * swim.D, 0.0])
        # Vector{T}([5.0 * swim.D, 0.0])
        convert(Vector{T}, target.*swim.D)
    )
    reset!(env)
    env
end

Random.seed!(env::SwimmingEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::SwimmingEnv) = env.action_space
RLBase.state_space(env::SwimmingEnv) = env.observation_space
RLBase.reward(env::SwimmingEnv) = env.reward
RLBase.is_terminated(env::SwimmingEnv) = isapprox(env.state[1], 0.0, atol=env.params.ℓ/4.) || env.done #within a quarter of a fish?
# RLBase.is_terminated(env::SwimmingEnv) = env.done #within a quarter of a fish?

function RLBase.state(env::SwimmingEnv{A,T}) where {A,T} 
    dn,θn =  dist_angle(env.swimmer,env.target)
    [clamp(dn, 0, env.observation_space[1].right)|>T, mod2pi(θn)|>T ]     
end

function RLBase.reset!(env::SwimmingEnv{A,T}) where {A,T}
    #go back to the original state, not random?  
    θ = rand()*2π  #π/2 old
    r = rand()*env.observation_space[1].right #0.0 old

    env.swimmer = FD_agent(Vector{T}([state2xy(r,θ)...]),Vector{T}([env.params.Γ0,-env.params.Γ0]), 
                                     T(θ), Vector{T}([0.0,0.0]),Vector{T}([0.0,0.0]))
    dn,θn = dist_angle(env.swimmer, env.target)    
    env.state[1] = clamp(dn, 0, env.observation_space[1].right )
    env.state[2] = mod2pi(θn)       
    env.action = zero(T)
    env.t = 0
    env.done = false
    env.reward = zero(T)
    #if we add a dynamic target throw code here too!
    nothing
end

function (env::SwimmingEnv)(a::Union{Int, AbstractFloat})
    @assert a in env.action_space
    env.action = change_circulation!(env, a)
    _step!(env, env.action)
end

function _step!(env::SwimmingEnv, a)
    env.t += 1
    
    v2v = vortex_to_vortex_velocity([env.swimmer]; env.params.ℓ)   
    avg_v = sum(v2v,dims=2) #add_lr(v2v) 
    
    siv = self_induced_velocity([env.swimmer];env.params.ℓ)
    ind_v = siv + avg_v #eqn 2.a

    angles = angle_projection([env.swimmer],v2v)
    # @show norm(ind_v) #siv,avg_v
    for (i,b) in enumerate([env.swimmer])
        # log these values and look at them, are they physical? 
        b.position += ind_v[:,i] .* env.params.Δt
        b.angle = mod2pi(b.angle + angles[i] .* env.params.Δt) #eqn 2.b
    end
    
    dn,θn = dist_angle(env.swimmer, env.target)
    # NATE : TODO ADD TARGET MOTION HERE - below is a crappy circle 
    # path(omega,t) = omega*env.params.ℓ*5.0 .*[-sin(omega*t),cos(omega*t)]    
    # env.target += path(2π, env.t/env.max_steps) .* env.params.Δt
    
    env.state[1] = clamp(dn, 0, env.observation_space[1].right ) #make sure it is in the domain
    env.state[2] = mod2pi(θn)       

    #NATE: changed up costs to allow wildly negative costs if outside of observation_space
    costs = env.params.wd*(1- dn /env.observation_space[1].right) + env.rewards[Int(a)]*env.params.wa 
    @assert costs <= 1.0
    env.done = env.t >= env.max_steps
    env.reward = costs
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
    sqrt(sum(abs2,d)), mod2pi(mod2pi(atan(d[2],d[1])) - agent.angle)
    
end

dist_angle(env::SwimmingEnv) = dist_angle(env.swimmer, env.target)   


# function change_circulation!(env::SwimmingEnv{<:Base.OneTo}, a::Int)
#     """ Agent starts with circulation of from
#         [swim.Γ0,-swim.Γ0]
#         so we have to inc/dec to change the magnitude not absolutes
#         [CRUISE, FASTER, SLOWER, LEFT, RIGHT]"""
#     # TODO: Add sign(enb.swimmer.gamma)?
    
#     if a == 1
#         env.swimmer.gamma = [env.params.Γ0, -env.params.Γ0]
#     elseif a == 2   
#         env.swimmer.gamma = [env.params.Γ0 + env.params.Γa, -env.params.Γ0 - env.params.Γa]
#     elseif a == 3  
#         env.swimmer.gamma = [env.params.Γ0 - env.params.Γa, -env.params.Γ0+ env.params.Γa]
#     elseif a == 4  
#         env.swimmer.gamma = [env.params.Γ0 + env.params.Γt, -env.params.Γ0 + env.params.Γt]
#     elseif a == 5   
#         env.swimmer.gamma = [ env.params.Γ0-env.params.Γt,  -env.params.Γ0-env.params.Γt]
#     else 
#         @error "unknown action of $action"
#     end 
    
#     a
# end

function change_circulation!(env::SwimmingEnv{<:Base.OneTo}, a::Int)
    """ Agent starts with circulation of from
        [-swim.Γ0, swim.Γ0]
        so we have to inc/dec to change the magnitude not absolutes
        [CRUISE, FASTER, SLOWER, LEFT, RIGHT]"""
    # TODO: Add sign(enb.swimmer.gamma)?
    
    if a == 1
        nothing
        # env.swimmer.gamma = [-env.params.Γ0, env.params.Γ0]
    elseif a == 2   
        env.swimmer.gamma = [-env.params.Γ0 - env.params.Γa, env.params.Γ0 + env.params.Γa]
    elseif a == 3  
        env.swimmer.gamma = [-env.params.Γ0 + env.params.Γa, env.params.Γ0 - env.params.Γa]
    elseif a == 4  
        env.swimmer.gamma = [-env.params.Γ0 - env.params.Γt, env.params.Γ0 - env.params.Γt]
    elseif a == 5   
        env.swimmer.gamma = [-env.params.Γ0 + env.params.Γt, env.params.Γ0  + env.params.Γt]
    else 
        @error "unknown action of $action"
    end 
    
    a
end

state2xy(r,θ) = [r*cos(θ),r*sin(θ)]
env = SwimmerEnv()
RLBase.test_runnable!(env)
