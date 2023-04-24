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
using SparseArrays


### Hopper RigidBodyDynamics conventions
# Configuration is q = [quat_body_to_world; pos_in_world; link_0; link_2; reaction wheels; unact_links]
# Velocity is v = [omega_in_body; vel_in_body; link velocities; reaction wheel velocities; unact link velocities]
# Note: q̇ != v̇; Can go from v̇ to q̇ by using the attitude jacobian on omega, and rotation the body linear velocity
#               into the world frame

# cross(v, x) is equal to skew(v)*x (cross product is linear operator)
skew(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

function ihlqr(A, B, Q, R, Qf; max_iters = 1000, tol = 1e-8)
    P = Qf
    K = zero(B')
    for _ = 1:max_iters
        P_prev = deepcopy(P)
        K = (R .+ B'*P*B) \ (B'*P*A)
        P = Q + A'P*(A - B*K)
        if norm(P - P_prev, 2) < tol
            return K
        end
    end
    @error "ihlqr didn't converge", norm(K - (R .+ B'*P*B) \ (B'*P*A), 2)
    return K
end

function axis_angle_to_quat(ω; tol = 1e-12)
    norm_ω = norm(ω)
    if norm_ω >= tol
        return [cos(norm_ω/2); ω/norm_ω*sin(norm_ω/2)]
    else
        return [1; 0; 0; 0]
    end
end

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

function L_mult(q)
    qs = q[1]
    qv = q[2:4]
    return [qs -qv'; qv qs*I + skew(qv)]
end

# Attitude jacobian (from axis-angle)
function G(q)
    qs = q[1]
    qv = q[2:4]
    return [-qv'; qs*I + skew(qv)]
end

function axis_angle_to_quat(ω; tol = 1e-12)
    norm_ω = norm(ω)
    
    if norm_ω >= tol
        return [cos(norm_ω/2); ω/norm_ω*sin(norm_ω/2)]
    else
        return [1; 0; 0; 0]
    end
end

function quat_to_axis_angle(q; tol = 1e-12)
    qs = q[1]
    qv = q[2:4]
    norm_qv = norm(qv)
    
    if norm_qv >= tol
        θ = 2*atan(norm_qv, qs)
        return θ*qv/norm_qv
    else
        return zeros(3)
    end
end

# Convert quaternion to rotation, derived from x̂_new = q*x̂*q† = L(q)*R(q)'*H*x
function quat_to_rot(q)
    skew_qv = skew(q[2:4])
    return 1.0I + 2*q[1]*skew_qv + 2*skew_qv^2
end

function state_error(x, x0)
    return [
        quat_to_axis_angle(L_mult(x0[1:4])'*x[1:4])
        x[5:end] - x0[5:end]
    ]
end

# Error state jacobian
E(x) = BlockDiagonal([0.5*G(x[1:4]), quat_to_rot(x[1:4]), convert.(eltype(x), I(length(x) - 7))])

# Error state jacobian transpose
E_T(x) = BlockDiagonal([0.5*G(x[1:4])', quat_to_rot(x[1:4])', 1.0*I(length(x) - 7)])

# Dynamics with constraints
function dynamics(robot, x, u, k_p, k_d)
    T = promote_type(eltype(x), eltype(u))
    state = MechanismState{T}(robot)
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
    B[7, 1] = 1; # Link 0 
    B[8, 2] = 1; # Link 2
    B[9:11, 3:5] = I(3); # Reaction wheels

    # Constraint function on foot
    foot_c(q) = foot_pinned_c(q, robot)    # Constraint
    foot_c_d(q) = foot_pinned_c_d(q, v, robot)    # Constraint velocity
    J1 = foot_pinned_J(q, robot)               # Jacobian of constraint
    J_d1 = foot_pinned_J_d(q, v, robot)            # Jacobian of constraint velocity

    # Constraint function on closed loop
    loop_c(q) = closed_loop_c(q, robot)    # Constraint
    loop_c_d(q) = closed_loop_c_d(q, v, robot)     # Constraint velocity
    J2 = closed_loop_J(q, robot)                # Jacobian of constraint
    J_d2 = closed_loop_J_d(q, v, robot)          # Jacobian of constraint velocity

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
function closed_loop_c(q, robot)
    state = MechanismState{eltype(q)}(robot)
    set_configuration!(state, q)
    link1 = findbody(robot, "link1")
    link3 = findbody(robot, "link3")
    point_in_world = transform(state, Point3D(default_frame(link1), [0.27; 0; 0]), default_frame(link3))
    return point_in_world.v[[1, 3]] - convert.(eltype(q), [0.1; 0]) # Should always be zero
end

function closed_loop_J(q, robot)
    return FiniteDiff.finite_difference_jacobian(q -> closed_loop_c(q, robot), q)*E(q)
end

function closed_loop_c_d(q, v, robot)
    return closed_loop_J(q, robot)*v
end

function closed_loop_J_d(q, v, robot)
    return FiniteDiff.finite_difference_jacobian(q -> closed_loop_c_d(q, v, robot), q)*E(q)
end

# Constraint expressing the foot in world coordinates
function foot_pinned_c(q, robot)
    state = MechanismState{eltype(q)}(robot)
    set_configuration!(state, q)
    link3 = findbody(robot, "link3")
    point_in_world = transform(state, Point3D(default_frame(link3), [0.27; 0; -.0205]), root_frame(robot))
    return point_in_world.v
end

function foot_pinned_J(q, robot)
    return FiniteDiff.finite_difference_jacobian(q -> foot_pinned_c(q, robot), q)*E(q)
end

function foot_pinned_c_d(q, v, robot)
    return foot_pinned_J(q, robot)*v
end

function foot_pinned_J_d(q, v, robot)
    return FiniteDiff.finite_difference_jacobian(q -> foot_pinned_c_d(q, v, robot), q)*E(q)
end

function c_viol(q, robot)
    return maximum(abs.([foot_pinned_c(q, robot); closed_loop_c(q, robot)]))
end

# rk4 on dynamics (naive way, need to normalize quaternion)
function rk4(model, x_k, u, h, k_p, k_d)
    k1 = h*dynamics(model, x_k, u, k_p, k_d)
    k2 = h*dynamics(model, x_k + k1/2, u, k_p, k_d)
    k3 = h*dynamics(model, x_k + k2/2, u, k_p, k_d)
    k4 = h*dynamics(model, x_k + k3, u, k_p, k_d)
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
nq, nv, nx, nu = 14, 13, 27, 5
N = 100
h = 0.001
tf = h*(N - 1)

# TODO Find initial balancing state and control
q0 = [1; zeros(13)]
q0[5:7] = -foot_pinned_c(q0, robot)
z = [0; 0; 1]
set_configuration!(state, q0)
com0 = normalize(center_of_mass(state).v)
quat = axis_angle_to_quat(-normalize(skew(z)*com0)*acos(z'*com0))
q0 = [quat; -foot_pinned_c([quat; zeros(10)], robot); zeros(7)]
x0 = [q0; zeros(nv)]
u0 = [-17.910887726940793; 17.666841134884233; zeros(3)]
maximum(abs.(rk4(robot, x0, u0, h, -10000, 100) - x0))

# TODO Linearization and calculating LQR gain
A = E_T(x0)*FiniteDiff.finite_difference_jacobian(x -> rk4(robot, x, u0, h, 0, 0), x0)*E(x0)
B = E_T(x0)*FiniteDiff.finite_difference_jacobian(u -> rk4(robot, x0, u, h, 0, 0), u0)

Q = sparse(1.0*I(26))
R = sparse(1.0*I(5))

K = ihlqr(A, B, Q, R, Q, max_iters = 1000)

# Simulation
X = [zeros(nx) for _ = 1:N] 
U = [zeros(nu) for _ = 1:N - 1]
X[1] = [1; zeros(13); zeros(13)]
X[1][5:7] -= foot_pinned_c(X[1][1:14], robot) # Make sure foot constraint is satisfied at start
for k = 1:(N-1)
    # TODO control on error state
    Δx = state_error(X[k], x0)

    U[k] = u0 - K*Δx

    X[k + 1] = rk4(robot, X[k], U[k], h, -10000, 100)

    # Check constraint violation
    if c_viol(X[k + 1][1:14], robot) > 1e-2
        @error "Constraints violated, stopping sim" k c_viol(X[k + 1][1:14], robot)
        [X[i] = copy(X[k + 1]) for i = k+2:N]
        break
    end
end
visualize!(mvis, LinRange(0, tf, N), X)
vis

