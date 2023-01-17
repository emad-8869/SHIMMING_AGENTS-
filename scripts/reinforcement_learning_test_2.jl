using ReinforcementLearning
using Random
using StableRNGs
using Flux
using Flux.Losses
using Plots


Base.@kwdef mutable struct DistRewardPerEpisode <: AbstractHook    
    dists::Vector = []
    positions:: Vector = []
    rewards::Vector = []
    gammas::Vector = []
    reward = 0.0
    is_display_on_exit::Bool = true
end

function (h::DistRewardPerEpisode)(::PostEpisodeStage, policy, env) 
    push!(h.dists, dist_angle(env))#sqrt(sum(abs2, env.target .- state2xy(env.state...))))
    push!(h.positions, env.swimmer.position)
    push!(h.gammas, env.swimmer.gamma)
    h.reward += reward(env)
    push!(h.rewards,h.reward)
end

include(".\\SwimmerEnv.jl")
function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    # ::Val{:PendulumDiscrete},
    ::Val{:LTSDiscrete},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(187)
    # env2 = PendulumEnv(continuous = false, max_steps = 5000, rng = rng)
    env = SwimmerEnv(max_steps = 1000, target=[1,1])
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = 
            # RandomPolicy(),
            QBasedPolicy(
            # learner = BasicDQNLearner(
            #     approximator = NeuralNetworkApproximator(
            #         model = Chain(
            #             Dense(ns, 64, relu; init = glorot_uniform(rng)),
            #             Dense(64, 64, relu; init = glorot_uniform(rng)),
            #             Dense(64, na; init = glorot_uniform(rng)),
            #         ) |> gpu,
            #         optimizer = ADAM(),
            #     ),
            #     batch_size = 32,
            #     min_replay_history = 100,
            #     loss_func = huber_loss,
            #     rng = rng,
            # ),
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, na; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, na; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(),
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 5000,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 50_000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = DistRewardPerEpisode() #TotalRewardPerEpisode()

    Experiment(agent, env, stop_condition, hook, "LTS")
end

ex = E`JuliaRL_BasicDQN_LTSDiscrete`

run(ex)
plot(ex.hook.rewards,marker=:dot)
plot(ex.hook.gammas ,label="")
rs = [r for (r,t) in ex.hook.dists]
ts = [t for (r,t) in ex.hook.dists]
plot(rs)
plot!(ts)
begin
    lim = ex.env.observation_space[1].right
    plot([x for (x,y) in ex.hook.positions],[y for (x,y) in ex.hook.positions],
        m=:dot,ls=:dash)
    plot!([ex.env.target[1]],[ex.env.target[2]],
        st=:scatter,c=:red,label="target")
        # xlims=[-lim*2.,lim*2.], ylims=[-lim*2.,lim*2.])

end

begin
    S = state_space(ex.env)
    R = range(S[1].left, stop=S[1].right, length= 25)
    Θ = range(S[2].left, stop=S[2].right, length= 32)
    function show_approx(n)
        run(ex.policy,ex.env, StopAfterEpisode(n))
        [ex.policy.policy.learner.approximator([p, v]) |> maximum for p in R, v in Θ]
    end

    n = 50
    field = -show_approx(n)
    # plot(R,Θ, field , linetype=:wireframe,
        # xlabel="R", ylabel="θ", zlabel="cost", title="Episode $n")
    plot(R,Θ, field' , linetype=:contourf,
    ylabel="θ", xlabel="R",  title="Episode $n",c=:thermal)
end
plot(R,Θ, field' ,proj=:polar,
ylabel="θ", xlabel="R",  title="Episode $n",c=:thermal)
"""
Bullshit for testing 
"""
act = env |> policy
@show env.swimmer
(env)(act)
