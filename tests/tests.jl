begin
    # How is the self-induced vel for turning boids? validate against Barba code--- its good
    ℓ = 1.0
    boids = [FD_agent([0.0, 0.0],  [1.0,-1], π/2.0, [0.0,swim.v₀], [0.0,0.0])]
    v2v = vortex_to_vortex_velocity(boids;ℓ)
    siv = self_induced_velocity(boids;ℓ)
    avg_v = add_lr(v2v)
    ind_v = siv + avg_v #eqn 2.a
    angles = angle_projection(boids,v2v)
    @assert ind_v ≈ vortex_to_swimmer_midpoint_velocity(boids;ℓ)
    @assert ind_v ≈ [0.0, -0.3183098861837907] #pulled from Barba AeroPython
end
begin
    # How is the self-induced vel for turning boids? 
    ℓ = 1.0
    boids = [FD_agent([0.0, 0.0],  [1.0,-1.0], 3π/4.0, [0.0,swim.v₀], [0.0,0.0])]
    v2v = vortex_to_vortex_velocity(boids;ℓ)
    siv = self_induced_velocity(boids;ℓ)
    avg_v = add_lr(v2v)
    ind_v = siv + avg_v #eqn 2.a
    angles = angle_projection(boids,v2v;ℓ)
    @assert ind_v  ≈ vortex_to_swimmer_midpoint_velocity(boids;ℓ)
    @assert ind_v ≈ [0.2250790790392765, -0.22507907903927654] #pulled from Barba AeroPython
end