using ReinforcementLearning
using Random
using StableRNGs
using Flux
using Flux.Losses
using Plots

include(".\\SwimmerEnv.jl")
function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    # ::Val{:PendulumDiscrete},
    ::Val{:LTSDiscrete},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(123)
    # env2 = PendulumEnv(continuous = false, max_steps = 5000, rng = rng)
    env = SwimmerEnv()
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, na; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 5_000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    Experiment(agent, env, stop_condition, hook, "LTS")
end

ex = E`JuliaRL_BasicDQN_LTSDiscrete`

run(ex)
plot(ex.hook.rewards)



"""
Bullshit for testing 
"""
act = env |> policy
@show env.swimmer
(env)(act)
