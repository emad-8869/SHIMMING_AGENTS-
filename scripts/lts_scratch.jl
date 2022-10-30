using LazyGrids
using Plots
using SparseArrays
# using PlotlyJS


Base.@kwdef mutable struct boid{T<:Real}
        position::Vector{T} #2-D to start
        gamma::Vector{T} #Γ_l Γ_r
        angle::T #α
        v::Vector{T} #velocity
        a::Vector{T} #acceleration
end 
    # Print function for a boid
Base.show(io::IO,b::boid) = print(io,"Boid (x,y,α)=($(b.position),$(b.angle)), ̂v = $(b.v)")
   
b = boid([1.0,0.5],[-1.0,1.0], π/2, [0.0,0.0],[0.0,0.0])

function vortex_vel(boids::Vector{boid{T}} ;ℓ=0.001) where T<:Real
    """
    find vortex velocity from sources to itself
    sources - vector of vortex_particle type
    DOES NOT UPDATE STATE of sources
    returns velocity induced field
    """  

    n =size(boids)[1]
    vels = zeros(T, (2,n))
    vel = zeros(T, n)
    targets = zeros(T,(2,n))
    for (i,b) in enumerate(boids)
        targets[:,i] = b.position
    end
    # targets = [b.position[:] for b in boids]
    for i in 1:n            
        dx = targets[1,:] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle+π/2))
        dy = targets[2,:] .- (boids[i].position[1] .+ ℓ*sin(boids[i].angle+π/2))
        @. vel = boids[i].gamma[1]  / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
        dx = targets[1,:] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle-π/2))
        dy = targets[2,:] .- (boids[i].position[1] .+ ℓ*sin(boids[i].angle-π/2))
        @. vel = boids[i].gamma[2] / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
    end
    vels
end

function vortex_vel(boids ::Vector{boid{T}}, targets  ;ℓ=0.001) where T<:Real
    """
    find vortex velocity from sources to a LazyGrids of targets
    """

    n = size(boids)[1]
    vels = zeros(T, (2,size(targets[1])...))
    vel = zeros(T, size(targets[1]))
    for i in 1:n            
        #left vortex
        dx = targets[1] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle+π/2))
        dy = targets[2] .- (boids[i].position[2] .+ ℓ*sin(boids[i].angle+π/2))
        @. vel = boids[i].gamma[1]  / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
        #right vortex
        dx = targets[1] .- (boids[i].position[1] .+ ℓ*cos(boids[i].angle-π/2))
        dy = targets[2] .- (boids[i].position[2] .+ ℓ*sin(boids[i].angle-π/2))
        @. vel = boids[i].gamma[2] / (2π *(dx^2 + dy^2 ))
        @. vels[1,:,:] += dy * vel
        @. vels[2,:,:] -= dx * vel
    end
    vels
end


function potential(boids::Vector{boid{T}},targets; ℓ= 0.001) where T<: Real
    """
    find vortex potential from sources to a LazyGrids of targets
    mainly for plotting, but might be worth examining wtih Autodiff
    """
    pot = zeros(T, (size(targets[1])...))

    for b in boids            
        #left vortex
        dx = targets[1] .- (b.position[1] .+ ℓ*cos(b.angle+π/2))
        dy = targets[2] .- (b.position[2] .+ ℓ*sin(b.angle+π/2))
        @. pot += -b.gamma[1] *atan(dx,dy)
        #right vortex
        dx = targets[1] .- (b.position[1] .+ ℓ*cos(b.angle-π/2))
        dy = targets[2] .- (b.position[2] .+ ℓ*sin(b.angle-π/2))
        @. pot += -b.gamma[2] *atan(dx,dy)

        #self influence ? no
        # @show b
        # dx = targets[1] .- b.position[1] 
        # dy = targets[2] .- b.position[2] 
        # @. pot += -(b.gamma[2] - b.gamma[1])/(dx^2 + dy^2)*dy^2
    end
    pot./(2π)
end
#We are using 32-bits throughout
type =T= Base.Float32
n_boids = 3
# Make a grid - strictly for visualization (so far)
xs = LinRange{type}(-2,2,31)
ys = LinRange{type}(-2,2,21)
targets = ndgrid(xs,ys)
#do it a different way
X = repeat(reshape(xs, 1, :), length(ys), 1)
Y = repeat(ys, 1, length(xs))
targets = [X,Y]

boids = [boid([1.0, -0.3],  [-1.0,1.0], π/4, [0.0,0.0], [0.0,0.0])]#,
        #  boid([-0.0,-0.0],[-1.0,1.0],    π/3., [0.0,0.0], [0.0,0.0]),
        #  boid([-1.0,-1.0],[1.0,-1.0],  0.0, [0.0,0.0], [0.0,0.0])]




ind_v = vortex_vel(boids)
field_v = vortex_vel(boids, targets;ℓ=0.5)

field_pot = potential(boids, targets;ℓ=0.5)
plot(collect(xs),collect(ys), field_pot, st=:contourf)#,clim=(-0.05,0.05))
scatter!([boids[1].position[1]],[boids[1].position[2]],color=:red)
atanp(x,y) = begin atan(y,x) end
plot(X,X,atan,st=:contourf)
# plotlyjs()
# plot(contour(
#     x = collect(xs),
#     y = collect(ys),
#     z = field_pot
# ))
# X = repeat(reshape(xs, 1, :), length(ys), 1)
# Y = repeat(ys, 1, length(xs))
# targets = [X,Y]
quiver!(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec),
       xlims=(xs[1],xs[end]),ylims=(ys[1],ys[end]), aspect_ratio= :equal)
ℓ=0.5
begin
    plot(collect(xs),collect(ys), field_pot, st=:contourf)
    for b in boids
        @show (cos(b.angle),sin(b.angle),b.angle)
        scatter!([b.position[1]],[b.position[2]],markersize=4,color=:red,label="")
        scatter!([b.position[1]+ ℓ*cos(b.angle+π/2)],[b.position[2]+ ℓ*sin(b.angle+π/2)]
                ,markersize=4,color=:green,label="left")
        scatter!([b.position[1]+ ℓ*cos(b.angle-π/2)],[b.position[2]+ ℓ*sin(b.angle-π/2)]
                ,markersize=4,color=:blue,label="right")
        #Why is it only doing quiver in the x-dir?
        quiver!([b.position[1]], [b.position[2]], quiver= [cos(b.angle),sin(b.angle)])
        
    end
    plot!()
end
quiver!()



abs_v = field_v[1,:,:].^2 .+ field_v[2,:,:].^2
plot(collect(xs),collect(ys), field_pot, st=:contour)

quiver!(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec), aspect_ratio= :equal)

#make dipole swimmers
Γ = [-1,1,-0.1,0.1]
x = [-0.1, 0.1,-0.1, 0.1,]
y = [0,0,-0.1,-0.1]
pos = [x' ; y']
xs = LinRange{type}(-1,1,21)
ys = deepcopy(xs)
targets = ndgrid(xs,ys)
vel = zeros(type,size(pos))
init_state = [pos; vel ; Γ']
ind_v = vortex_vel(init_state)
field_v = vortex_vel(init_state, targets)
vort =   diff(field_v[2,:,:],dims=1)[:,2:end] .-diff(field_v[1,:,:],dims=2)[2:end,:]
# contourf(collect(xs)[2:end],collect(ys)[2:end], vort)
scatter(init_state[1,:], init_state[2,:], markersize=2)
quiver!(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec), aspect_ratio= :equal,
        xlims=(xs[1],xs[end]),ylims=(ys[1],ys[end]))

n=80
T = 20
Δt = 0.01f0
anim = @animate for i ∈ 1:n
            Γ = [-1,1,0.1,-0.1] .*(1 .+ cos(2*π*i/T))
            # @show Γ
            state = [pos; vel ; Γ']
            ind_v = vortex_vel(state)
            f_vels = vortex_vel(state, targets)
           
            quiver(targets[1]|>vec,targets[2]|>vec,
                   quiver = (f_vels[1,:,:]|>vec,f_vels[2,:,:]|>vec),
                   aspect_ratio= :equal,
                   xlim=(-1,1),ylim=(-1,1));
            scatter!(state[1,:],state[2,:],
                    markersize=abs.(state[5,:]),
                    palette=:balance,label=false);
end
gif(anim, "vortexswimmer.gif", fps = 20)
# Define some vortices
 num_vort = 200
samp_inds = [rand(1:num_vort) for i = 1:num_vort÷10]
x = range(-10,10,num_vort)|>Vector{type}
y = deepcopy(x).*rand(length(x))
pos = [x' ; y']
# Γ = range(1,10,num_vort÷2)|>Vector{Float32}
# Γ = [reverse(Γ)' Γ']
Γ = (rand(length(x)).-0.5).*20
vel = zeros(type,size(pos))

init_state = [pos; vel ; Γ']
sample_vortices = init_state[:,samp_inds]


ind_v = vortex_vel(init_state)
field_v = vortex_vel(init_state, targets)

quiver(targets[1]|>vec,targets[2]|>vec, quiver = (field_v[1,:,:]|>vec,field_v[2,:,:]|>vec), aspect_ratio= :equal)
scatter!(init_state[1,:], init_state[2,:], markersize=abs.(init_state[5,:]))

n=150
Δt = 0.01f0
state = deepcopy(init_state)
# old snap shot of all data
snaps = deepcopy([state...])
# snaps = deepcopy([state[1:2,:]...])

for i ∈ 1:n
    v_vels = vortex_vel(state);
    move_vortices(state, v_vels, Δt);
    # old for many valued state
    snaps = [snaps [state...]]
    # snaps = [ snaps [state[1:2,:]...]] #splat it flat
end

anim = @animate for i ∈ 1:n
    now = reshape(snaps[:,i],(5,num_vort))
    f_vels = vortex_vel(now ,targets);
   
    quiver(targets[1]|>vec,targets[2]|>vec,
           quiver = (f_vels[1,:,:]|>vec,f_vels[2,:,:]|>vec),
           aspect_ratio= :equal,
           xlim=(-20,20),ylim=(-20,20));
    scatter!(now[1,:],now[2,:],
            markersize=now[5,:],
            palette=:balance,label=false);
end
gif(anim, "line_vortex.gif", fps = 30)

sources =[-5.0f0  5.0f0 
           10.f0  10.0f0 
           0.0f0  0.0f0 
           0.0f0  0.0f0            
           100.0f0 100.0f0]

vel = vortex_vel(sources)