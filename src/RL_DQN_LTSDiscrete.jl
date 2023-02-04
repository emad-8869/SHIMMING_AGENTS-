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
    rng = StableRNG(187)
    # env2 = PendulumEnv(continuous = false, max_steps = 5000, rng = rng)
    env = SwimmerEnv(max_steps = 100, target=[0,0])
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = 
          
            QBasedPolicy(
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
    hook = DistRewardPerEpisode() 

    Experiment(agent, env, stop_condition, hook, "LTS")
end

ex = E`JuliaRL_BasicDQN_LTSDiscrete`

run(ex)
plot(ex.hook.rewards)

begin
    #animate the path of an Episode
    epNum = 99
    plot([ex.env.target[1]],[ex.env.target[2]],st=:scatter,marker=:star,color=:green,label="target")
    anim = @animate for pos in ex.hook.positions[epNum]
            plot!([pos[1]],[pos[2]],st=:scatter,
                 aspect_ration=:equal,label="",markershape=:octagon,
                 color=:blue)
    end
    gif(anim)
end



begin
    i = argmax(ex.hook.rewards)
    xs = []
    ys = []
    for pos in ex.hook.positions[i]
        push!(xs,pos[1])
        push!(ys,pos[2])
    end
    plot([ex.env.target[1]],[ex.env.target[2]],st=:scatter,marker=:star,color=:green,label="target")
    plot!([xs[1]],[ys[1]],marker=:circle,st=:scatter,color=:green,label="start")
    plot!(xs,ys,label=ex.hook.rewards[i])
    plot!([xs[end]],[ys[end]],marker=:circle,st=:scatter,color=:red, label="end")
end