using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate();
using LinearAlgebra
using RigidBodyDynamics
using MeshCat
using BlockDiagonals
using MeshCatMechanisms
using GeometryBasics
import ForwardDiff as FD
using FiniteDiff
using Statistics
using BenchmarkTools

### Hopper RigidBodyDynamics conventions
# Configuration is q = [quat_body_to_world; pos_in_world; link_0; link_2; reaction wheels]
# Velocity is v = [omega_in_body; vel_in_body; link velocities; reaction wheel velocities]
# Note: q̇ != v̇; Can go from v̇ to q̇ by using the attitude jacobian on omega, and rotation the body linear velocity
#               into the world frame

# cross(v, x) is equal to skew(v)*x (cross product is linear operator)
skew(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

# Visualize hopper
function visualize!(mvis::MechanismVisualizer,
    times::AbstractVector{<:Real},
    q::AbstractVector{<:AbstractVector{<:Real}}; fps::Integer = 30)

    # Create animation
    anim = MeshCat.Animation(fps)

    # Figure out skip
    dt = mean(diff(times))
    skip = max(1, floor(Int, 1/dt/fps))

    # Animate
    for ind in 1:skip:length(q)
        atframe(anim, Integer((ind - 1)/skip)) do 
            set_configuration!(mvis, q[ind][1:14])
        end
    end
    
    # Play animation
    MeshCat.setanimation!(mvis, anim)
end

# Attitude jacobian (from axis-angle)
function G(q)
    qs = q[1]
    qv = q[2:4]
    return [-qv'; qs*I + skew(qv)]
end

# Convert quaternion to rotation, derived from x̂_new = q*x̂*q† = L(q)*R(q)'*H*x
function quat_to_rot(q)
    skew_qv = skew(q[2:4])
    return 1.0I + 2*q[1]*skew_qv + 2*skew_qv^2
end

# Error state jacobian
E(x) = BlockDiagonal([0.5*G(x[1:4]), quat_to_rot(x[1:4]), convert.(eltype(x), I(length(x) - 7))])

# Error state jacobian transpose
E_T(x) = BlockDiagonal([2*G(x[1:4])', quat_to_rot(x[1:4])', 1.0*I(length(x) - 7)])

# Dynamics with constraints
function dynamics(state, x, u; k_p = 0, k_d = 0)
    # Get config and velocity
    q = x[1:14]
    v = x[15:end]

    # Update mechanism state for RigidBodyDynamics
    set_configuration!(state, q)
    set_velocity!(state, v)

    # Attitude kinematics, convert v̇ to q̇
    q̇ = [0.5*G(q[1:4])*v[1:3]; quat_to_rot(q[1:4])*v[4:6]; v[7:end]]

    # Dynamics (get M, C, and B)
    M = mass_matrix(state)
    C = Vector(dynamics_bias(state))
    B = zeros(13, 5)
    B[8, 1] = 1; # Link 0 
    B[10, 2] = 1; # Link 2
    B[11:13, 3:5] = I(3); # Reaction wheels

    # Constraint function on foot
    foot_c(q) = foot_pinned_constraint(q, robot)    # Constraint
    foot_c_d(q) = FD.jacobian(foot_c, q)*E(q)*v     # Constraint velocity
    J1 = FD.jacobian(foot_c, q)*E(q)                # Jacobian of constraint
    J_d1 = FD.jacobian(foot_c_d, q)*E(q)            # Jacobian of constraint velocity

    # Constraint function on closed loop
    loop_c(q) = closed_loop_constraint(q, robot)    # Constraint
    loop_c_d(q) = FD.jacobian(loop_c, q)*E(q)*v     # Constraint velocity
    J2 = FD.jacobian(loop_c, q)*E(q)                # Jacobian of constraint
    J_d2 = FD.jacobian(loop_c_d, q)*E(q)            # Jacobian of constraint velocity

    # Combined jacobians
    c = [foot_c(q); loop_c(q)];
    c_d = [foot_c_d(q); loop_c_d(q)];
    J = [J1; J2]
    J_d = [J_d1; J_d2]

    # Solve the following system:
    # Mv̇ + C = Bu + J'λ             # Dynamics with constraint forces λ
    # Jv̇ + J̇v̇ = - k_p*c - k_d*ċ     # Constraint acceleration equals constraint stabilization (PD on constraint)

    res = [M -J'; J zeros(5, 5)] \ [B*u - C; -J_d*v - k_p*c - k_d*c_d] # Solve system
    v̇ = res[1:length(v)] # Extract v̇

    return [q̇; v̇]
end

# Constraint expressing the position of link1's pin joint in link3's frame (should be 0)
# Only constraint x and z since link rotates around y-axis
function closed_loop_constraint(q, robot)
    state = MechanismState{eltype(q)}(robot)
    set_configuration!(state, q)
    link1 = findbody(robot, "link1")
    link3 = findbody(robot, "link3")
    point_in_world = transform(state, Point3D(default_frame(link1), [0.27; 0; 0]), default_frame(link3))
    return point_in_world.v[[1, 3]] - [0.1; 0] # Should always be zero
end

# Constraint expressing the foot in world coordinates
function foot_pinned_constraint(q, robot)
    state = MechanismState{eltype(q)}(robot)
    set_configuration!(state, q)
    link3 = findbody(robot, "link3")
    point_in_world = transform(state, Point3D(default_frame(link3), [0.27; 0; -.0205]), root_frame(robot))
    return point_in_world.v
end

# rk4 on dynamics (naive way, need to normalize quaternion)
function rk4(model, x_k, u, h; k_p = 0, k_d = 0)
    k1 = h*dynamics(model, x_k, u, k_p=k_p, k_d=k_d)
    k2 = h*dynamics(model, x_k + k1/2, u, k_p=k_p, k_d=k_d)
    k3 = h*dynamics(model, x_k + k2/2, u, k_p=k_p, k_d=k_d)
    k4 = h*dynamics(model, x_k + k3, u, k_p=k_p, k_d=k_d)
    x_next = x_k + (k1 + 2*k2 + 2*k3 + k4)/6;
    x_next[1:4] = normalize(x_next[1:4])
    return x_next
end

# Visualizer
vis = Visualizer()

# Load robot
urdf = joinpath(@__DIR__, "rexhopper/rexhopper.urdf")
meshes = joinpath(@__DIR__, "rexhopper/")
robot = parse_urdf(urdf, floating=true)
mvis = MechanismVisualizer(robot, URDFVisuals(urdf, package_path = [meshes]), vis)
state = MechanismState(robot)

# Problem
nq, nv, nx = 14, 13, 27
N = 100
h = 0.01
tf = h*(N - 1)

# TODO Find initial balancing state and control
x0 = [1; zeros(26)]
x0[5:7] -= foot_pinned_constraint(x0[1:14], robot) 
u0 = zeros(5)
maximum(abs.(rk4(state, x0, u0, h) - x0)) # Should be 0

# TODO Linearization and calculating LQR gain

# Simulation
X = [zeros(nx) for _ = 1:N] 
X[1] = [1; zeros(13); zeros(13)]
X[1][5:7] -= foot_pinned_constraint(X[1][1:14], robot) # Make sure foot constraint is satisfied at start
for k = 1:N - 1
    # TODO control on error state

    X[k + 1] = rk4(state, X[k], zeros(5), h, k_p = 100, k_d = 10)
end
visualize!(mvis, LinRange(0, tf, N), X)
vis