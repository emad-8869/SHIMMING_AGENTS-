include("../src/BemRom.jl")
using Plots, CSV, DataFrames

ang = deepcopy(defaultDict)
ang[:N] = 64      #number of panels (elements) to discretize the foil
ang[:Nt] = 64     #number of timesteps per period of motion
ang[:Ncycles] = 3 #number of periods to simulate
ang[:f] = 1.0      #frequency of wave motion 
ang[:Uinf] = 1.0    #free stream velocity 
ang[:kine] = :make_ang 
a0 = 0.1 # how much the leading edge heaves up and down wrt the chord(length) of the swimmer
ang[:motion_parameters] = a0


begin
    foil, flow = init_params(;ang...)
    k = foil.f*foil.chord/flow.Uinf
    @show k
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)        
        phi =  get_phi(foil, wake)    # (output space) <-probably not that important                            
        p = panel_pressure(foil, flow,  old_mus, old_phis, phi)    
        # (output space) <- p is a function of μ and we should be able to recreate this     
        old_mus = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        # the coefficients of PERFROMANCE are important, but are just a scaling of P
        # if we can recreate p correctly, this will be easy to get also (not important at first)
        ps[:,i] = p # storage container of the output, nice!
    end
    t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1,start:end], label="Force"  ,lw = 3, marker=:circle)
    b = plot(t[start:end], coeffs[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    c = plot(t[start:end], coeffs[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
    d = plot(t[start:end], coeffs[4,start:end], label="Power"  ,lw = 3, marker=:circle)
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
end

begin
    # watch a video of the motion, does it blow up? if so, what went wrong? 
    foil, flow = init_params(;ang...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:flow.Ncycles*flow.N*1.75
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)       
        win=nothing
        f = plot_current(foil, wake;window=win)
        plot!(f, ylims=(-1,1))
        
    end
    gif(movie, "handp.gif", fps=10)
end

#save data

function save_data(input_data, output_data, output_dir)
    # Save input data
    input_file = joinpath(output_dir, "input_data.csv")
    CSV.write(input_file, input_data)

    # Save output data
    output_file = joinpath(output_dir, "output_data.csv")
    CSV.write(output_file, output_data)
end





function run_simulations(output_dir)
    # Define parameter ranges
    Strouhal_values = [0.1, 0.2, 0.3]
    reduced_freq_values = [0.1, 0.2, 0.3]
    wave_number_values = [0.1, 0.2, 0.3]

    allin = Vector{DataFrame}()
    allout = Vector{DataFrame}()

    # Nested loops to vary parameters
    for St in Strouhal_values
        for reduced_freq in reduced_freq_values
            for wave_number in wave_number_values
                # Set motion parameters
                ang = deepcopy(defaultDict)
                ang[:N] = 64
                ang[:Nt] = 64
                ang[:Ncycles] = 5 # number of cycles
                ang[:Uinf] = 1.0
                ang[:f] = reduced_freq * ang[:Uinf]
                ang[:kine] = :make_ang
                a0 = 0.1 #St * ang[:Uinf] / ang[:f]
                ang[:motion_parameters] = [a0]

                # Initialize Foil and Flow objects
                foil, fp = init_params(; ang...)
                wake = Wake(foil)
                (foil)(fp)

                # Perform simulations and save results
                old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
                phi = zeros(foil.N)
                input_data = DataFrame(St = Float64[], reduced_freq = Float64[], wave_number = Float64[], σ = Matrix{Float64}, panel_velocity = Matrix{Float64})
                output_data = DataFrame(mu = Float64[], pressure = Float64[])

                for i in 1:fp.Ncycles * foil.N
                    time_increment!(fp, foil, wake)
                    phi = get_phi(foil, wake)
                    p = panel_pressure(foil, fp, old_mus, old_phis, phi)
                    old_mus = [foil.μs'; old_mus[1:2, :]]
                    old_phis = [phi'; old_phis[1:2, :]]

                    if i == 1
                        input_data = DataFrame(St = [St], reduced_freq = [reduced_freq], wave_number = [wave_number], σ = [foil.σs'], panel_velocity = [foil.panel_vel'])
                        output_data = DataFrame(mu = [foil.μs'], pressure = [p'])
                    else
                        append!(input_data, DataFrame(St = [St], reduced_freq = [reduced_freq], wave_number = [wave_number], σ = [foil.σs'], panel_velocity = [foil.panel_vel']))
                        append!(output_data, DataFrame(mu = [foil.μs'], pressure = [p']))
                    end
                end

                push!(allin, input_data)
                push!(allout, output_data)
            end
        end
    end

    input_data = vcat(allin...)
    output_data = vcat(allout...)

    save_data(input_data, output_data, output_dir)
end




function init_flow(Nt)
    # Initialize the flow object with the given parameters
    return flow
end



function create_plot(output_dir)
    # Load output data
    output_file = joinpath(output_dir, "output_data.csv")
    output_data = CSV.read(output_file, DataFrame)

    # Extract relevant columns
    μ_data = output_data.mu
    pressure_data = output_data.pressure

    # Create plot
    t = collect(1:size(μ_data, 2))
    plot(t, μ_data, label = "μ", xlabel = "Time", ylabel = "μ Value", lw = 2)
    plot!(t, pressure_data, label = "Pressure", lw = 2)

    # Save plot
    plot_file = joinpath(output_dir, "simulation_plot.png")
    savefig(plot_file)
end





output_dir = "path/to/output/directory"
run_simulations(output_dir)
input_data_file = joinpath(output_dir, "input_data.csv")
output_data_file = joinpath(output_dir, "output_data.csv")
input_data = CSV.read(input_data_file, DataFrame)
output_data = CSV.read(output_data_file, DataFrame)


# Surrogates model
using Surrogates
using SurrogatesFlux
using Random
using Flux

# Define the function to approximate (you can replace this with your actual function)
f(x) = x^2

# Define the input space and sample the training data
bounds = Dict("x" => (0.0, 1.0))  # Bounds for the input variable x
n_samples = 100  # Number of training samples

# Sample training data
rng = Random.default_rng()
x_train = rand(rng, n_samples) .* (bounds["x"][2] .- bounds["x"][1]) .+ bounds["x"][1]
y_train = f.(x_train)

# Convert and reshape the input data
x_train_reshaped = Float32.(reshape(x_train, 1, length(x_train)))
y_train_reshaped = Float32.(reshape(y_train, 1, length(y_train)))


# Define the model architecture
model = Chain(
    Dense(1, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 8),
    Dense(8, 1)
)

# Define the loss function
loss(x, y) = mean(Flux.mse(model(x), y))

# Define the optimizer
optimizer = Descent(0.1)

# Define the number of epochs
n_epochs = 50

# Train the model
for epoch in 1:n_epochs
    Flux.train!(loss, Flux.params(model), [(x_train_reshaped, y_train_reshaped)], optimizer)
end


# Sample testing data and calculate the test errorcne
x_test = rand(rng, n_samples) .* (bounds["x"][2] .- bounds["x"][1]) .+ bounds["x"][1]
x_test_reshaped = Float32.(reshape(x_test, 1, length(x_test)))

# NOTE: Flux doesn't have a `predict` function by default. Instead, you can just call the model.
predictions = model(x_test_reshaped)
test_error = mean(abs2, predictions .- f.(x_test))

















# Function to visualize model performance
function visualize_model_performance(model, x_train, y_train, x_test)
    using Plots
    x_plot = range(0, stop=1, length=100)
    y_true = f.(x_plot)
    y_pred = Flux.predict(model, reshape(x_plot, (100, 1)))
    p = plot(x_plot, y_true, label="True", linewidth=2)
    plot!(p, x_plot, y_pred, label="Predicted", linewidth=2)
    scatter!(p, x_train, y_train, label="Training Data", color="red", markersize=5)
    return p
end

# Visualize the model performance
plot = visualize_model_performance(model, x_train, y_train, x_test)
display(plot)




#presentation plots
using Plots

function traditional_vs_biomimetic_gif()
    # Function to plot Traditional Propulsion
    function plot_traditional_propulsion()
        performance_labels = ["Energy Consumption", "Thrust Force", "Lift Force", "Maneuverability"]
        performance_values = [100, 30, 40, 20]

        bar(performance_labels, performance_values, ylabel="Performance", ylims=(0, 120),
            legend=:bottomright, title="Traditional Propulsion")
    end

    # Function to plot Biomimetic Swimming Foils
    function plot_biomimetic_swimming_foils()
        performance_labels = ["Energy Consumption", "Thrust Force", "Lift Force", "Maneuverability"]
        performance_values = [40, 80, 90, 80]

        bar(performance_labels, performance_values, ylabel="Performance", ylims=(0, 120),
            legend=:bottomright, title="Biomimetic Swimming Foils")
    end

    # Create the plots for traditional and biomimetic propulsion
    plot1 = plot_traditional_propulsion()
    plot2 = plot_biomimetic_swimming_foils()

    # Create the animation
    anim = @animate for i in 1:100
        offset = 0.01 * i

        plot1 = plot_traditional_propulsion()
        plot2 = plot_biomimetic_swimming_foils()

        plot!(plot1, alpha=1 - offset, color=:red)
        plot!(plot2, alpha=offset, color=:blue)

        plot(plot1, plot2, layout=(1, 2), size=(800, 400), xlabel="Performance", ylabel="Comparison",
             legend=false, title=["Traditional Propulsion" "Biomimetic Swimming Foils"])
    end

    # Save the animation as GIF
    gif_filename = "propulsion_comparison.gif"
    gif(anim, gif_filename, fps=10)

    # Return the animation
    return anim
end

traditional_vs_biomimetic_gif()



# Function to compare Traditional Propulsion with Biomimetic Swimming Foils
function traditional_vs_biomimetic_gif()
    function plot_traditional_propulsion()
        performance_labels = ["Energy Consumption", "Thrust Force", "Lift Force", "Maneuverability"]
        performance_values = [100, 30, 40, 20]

        bar(performance_labels, performance_values, ylabel="Performance", ylims=(0, 120),
            legend=:bottomright, title="Traditional Propulsion")
    end

    function plot_biomimetic_swimming_foils()
        performance_labels = ["Energy Consumption", "Thrust Force", "Lift Force", "Maneuverability"]
        performance_values = [40, 80, 90, 80]

        bar(performance_labels, performance_values, ylabel="Performance", ylims=(0, 120),
            legend=:bottomright, title="Biomimetic Swimming Foils")
    end

    # Create the plots for traditional and biomimetic propulsion
    plot1 = plot_traditional_propulsion()
    plot2 = plot_biomimetic_swimming_foils()

    # Create the animation
    anim = @animate for i in 1:100
        offset = 0.01 * i

        plot1 = plot_traditional_propulsion()
        plot2 = plot_biomimetic_swimming_foils()

        plot!(plot1, alpha=1 - offset, color=:red)
        plot!(plot2, alpha=offset, color=:blue)

        plot(plot1, plot2, layout=(1, 2), size=(800, 400), xlabel="Performance", ylabel="Comparison",
             legend=false, title=["Traditional Propulsion" "Biomimetic Swimming Foils"])
    end

    # Save the animation as GIF
    gif_filename = "propulsion_comparison.gif"
    gif(anim, gif_filename, fps=10)

    # Display the animation inline
    display(anim)

    return anim
end

# Call the function to display the animation immediately
traditional_vs_biomimetic_gif()





function show_thrust_lift_comparison()
    # Thrust and lift forces data for traditional and biomimetic foils
    thrust_values = [30, 80] # Example values, replace with actual data
    lift_values = [40, 90]   # Example values, replace with actual data

    # Labels for the bar plot
    labels = ["Traditional Propulsion", "Biomimetic Swimming Foils"]

    # Create the bar plot
    p = bar(labels, [thrust_values, lift_values], group=[:Thrust, :Lift],
            xlabel="Propulsion System", ylabel="Force", ylims=(0, 100),
            legend=:topright, title="Comparison of Thrust and Lift Forces",
            bar_width=0.5, bar_edges=true)

    # Add annotations to the plot
    annotate!([(1, thrust_values[1], text(string(thrust_values[1], " N"), :top)),
               (1, lift_values[1], text(string(lift_values[1], " N"), :bottom)),
               (2, thrust_values[2], text(string(thrust_values[2], " N"), :top)),
               (2, lift_values[2], text(string(lift_values[2], " N"), :bottom))])

    return p
end

# Show the comparison of thrust and lift forces
plot_comparison = show_thrust_lift_comparison()
display(plot_comparison)





function show_cfd_simulation()
    # Create a dummy plot to represent the swimming foils
    foil_plot = plot([0, 1, 2], [0, 1, 0], color=:blue, linewidth=2, label="Foil Shape",
                     xlabel="X-axis", ylabel="Y-axis", title="Numerical Simulation using CFD")

    # Add arrows to represent fluid flow around the foils
    plot!([0, 1], [0.5, 0.8], arrow=true, arrowhead=0.05, color=:red, linewidth=2, label="Fluid Flow")
    plot!([1, 2], [0.8, 0.5], arrow=true, arrowhead=0.05, color=:red, linewidth=2)

    # Add annotation for CFD description
    annotation = "CFD allows for accurate analysis of hydrodynamic forces and their effects on the foils' motion."
    annotate!([(0.5, 0.9, text(annotation, :left, :top, 10))])

    return foil_plot
end

# Show the CFD simulation plot
plot_cfd_simulation = show_cfd_simulation()
display(plot_cfd_simulation)





function show_foil_discretization()
    # Create a dummy plot to represent the swimming foils
    foil_plot = plot([0, 1, 2], [0, 1, 0], color=:blue, linewidth=2, label="Foil Shape",
                     xlabel="X-axis", ylabel="Y-axis", title="Foil Discretization")

    # Divide the foil into panels
    panel_x = [0, 0.5, 1, 1.5, 2]
    panel_y = [0, 0.2, 0.6, 0.2, 0]

    # Plot the panels
    plot!(panel_x, panel_y, seriestype=:shape, color=:orange, alpha=0.5, label="Panels")

    # Add arrows to represent hydrodynamic forces on each panel
    arrows_x = [0.5, 0.75, 1, 1.25, 1.5]
    arrows_y = [0.3, 0.4, 0.5, 0.4, 0.3]
    plot!(arrows_x, arrows_y, arrow=true, arrowhead=0.05, color=:red, linewidth=2, label="Hydrodynamic Forces")

    # Add annotation for Foil Discretization description
    annotation = "Foil discretization enables precise calculation of the forces experienced by each element of the foil."
    annotate!([(1, 0.7, text(annotation, :center, :top, 10))])

    return foil_plot
end

# Show the Foil Discretization plot
plot_foil_discretization = show_foil_discretization()
display(plot_foil_discretization)






function show_kinematic_parameters()
    # Create a dummy plot to represent the swimming foils
    foil_plot = plot([0, 1, 2], [0, 1, 0], color=:blue, linewidth=2, label="Foil Shape",
                     xlabel="X-axis", ylabel="Y-axis", title="Kinematic Parameters")

    # Add arrows to represent different kinematic parameters
    arrows_x = [0.3, 0.7, 1.3, 1.7]
    arrows_y = [0.8, 0.9, 0.8, 0.9]
    plot!(arrows_x, arrows_y, arrow=true, arrowhead=0.05, color=:green, linewidth=2, label="Kinematic Parameters")

    # Add annotation for Kinematic Parameters description
    annotation = "The study explores a wide range of kinematic parameters to understand the foils' performance under various flow conditions."
    annotate!([(1, 0.7, text(annotation, :center, :top, 10))])

    return foil_plot
end

# Show the Kinematic Parameters plot
plot_kinematic_parameters = show_kinematic_parameters()
display(plot_kinematic_parameters)





