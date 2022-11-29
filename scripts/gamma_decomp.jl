########## SHIT SHOW -> first pass, read in time varying data of body circulation to reduce it down

# using CSV
# using DataFrames
using DataDrivenDiffEq 
using DelimitedFiles
using Plots

file = "C:\\Users\\nate\\Desktop\\biofluids-coupled-bems\\offset_y0.5_x0.5\\gammas0.csv"
data = readdlm(file, ',', Float64)

size(data) #451, 132 -> Δt, 128 Body + 4 edge
begin 
    n = 451
    # T = LinRange(0, 1, n)
    x = LinRange(0,1,132)
    anim = @animate for t in 1:n
        plot(x,data[t,:], aspect_ratio=:equal,xlims=(-0.1,1.1),ylims=(-0.5,0.5))
        title!( "$(t%150)")
    end
    gif(anim,"circulation.gif", fps=12)
end

sum(data[:,1:128],dims=2)
sum(data[:,129:end],dims=2)
sum(data[:,:],dims=2)