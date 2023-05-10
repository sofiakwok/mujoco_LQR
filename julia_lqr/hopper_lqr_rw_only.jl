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

function constrained_ihlqr(A, B_u, B_λ, C, Q, R, Qf; max_iters = 1000, tol = 1e-8)
    nu, nλ = size(B_u, 2), size(B_λ, 2)
    P = Qf
    K = zero(B_u')
    L = zero(B_λ)
    for k = 1:max_iters
        P_prev = deepcopy(P)

        # Calculate constrainted L and k
        D = B_u - B_λ*((C*B_λ) \ C*B_u)
        gains = [R + D'*P*B_u D'*P*B_λ; C*B_u C*B_λ] \ [D'*P*A; C*A]
        K = gains[1:nu,:]
        L = gains[nu .+ (1:nλ),:]
       
        # Update P
        Ā = A - B_u*K - B_λ*L
        P = Q + K'*R*K + Ā'*P*Ā
        if norm(P - P_prev, 2) < tol
            return K
        elseif k == max_iters
            println(norm(P - P_prev, 2))
        end
    end
    @error "ihlqr didn't converge"
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
E_T(x) = BlockDiagonal([2*G(x[1:4])', quat_to_rot(x[1:4])', 1.0*I(length(x) - 7)])

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
    return FD.jacobian(q -> closed_loop_c(q, robot), q)*E(q)
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
    return FD.jacobian(q -> foot_pinned_c(q, robot), q)*E(q)
end

# Dynamics with constraints
function dynamics(robot, x, u, λ)
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
    B = zeros(13, 3)
    B[9:11, :] = I(3); # Reaction wheels

    # Constraint function on foot
    J1 = foot_pinned_J(q, robot)                   # Jacobian of foot pinned constraint
    J2 = closed_loop_J(q, robot)                   # Jacobian of closed loop constraint
    J3 = zeros(2, 13); J3[1, 7] = 1; J3[2, 8] = 1  # Jacobian of linkage constraint
    J = [J1; J2; J3]

    # Solve the following system:
    # Mv̇ + C = Bu + J'λ             # Dynamics with constraint forces λ
    v̇ = M \ (B*u + J'*λ - C)

    return [q̇; v̇]
end

function constraints(robot, x)
    q = x[1:14]
    return [
        closed_loop_c(q, robot); # Constraint on linkage
        foot_pinned_c(q, robot); # Fix foot
        q[8:9]                   # Don't let linkage move
    ]
end

function implicit_midpoint(robot, x_k, x_next, u, λ, h)
    ẋ = dynamics(robot, (x_k + x_next)/2, u, λ)
    ω_next = x_next[14 .+ (1:3)]
    return [
        2*G(x_k[1:4])'*x_next[1:4] - h*ω_next;      # Torso rotation (body-to-world)
        x_next[5:end] - (x_k[5:end] + h*ẋ[5:end]);  # Torso position, joints, reaction wheels
    ]
end

function pinned_implicit_midpoint(robot, x_k, x_next, u, λ, h)
    return [
        implicit_midpoint(robot, x_k, x_next, u, λ, h);
        constraints(robot, (x_next + x_k)/2)
    ]
end

function newton_implicit_midpoint(robot, x_k, u, h; max_iters = 10, tol = 1e-14)
    nx, nλ = length(x_k), 7

    # Init guess
    x_guess = copy(x_k)
    λ_guess = zeros(7)
    y_guess = [x_guess; λ_guess] # Solve for x and λ together

    # Form residual function
    rFunc(y) = pinned_implicit_midpoint(robot, x_k, y[1:nx], u, y[nx .+ (1:nλ)], h)

    # Evaluate residual, check convergence
    r = rFunc(y_guess)
    if norm(r, Inf) < tol
        return y_guess[1:nx], y_guess[nx .+ (1:nλ)]
    end

    # Newton steps
    for _ = 1:max_iters
        # Eval residual jacobian
        dr_dy = FD.jacobian(rFunc, y_guess)

        # Apply attitude jacobian for body attitude derivative
        dr_dy = [dr_dy[:, 1:4]*0.5*G(y_guess[1:4]) dr_dy[:, 5:end]]

        # Solve for step
        Δy = -dr_dy \ r

        # Apply step
        y_guess = [
            L_mult(y_guess[1:4])*axis_angle_to_quat(Δy[1:3])
            y_guess[5:end] + Δy[4:end];
        ]

        # Evaluate residual, check convergence
        r = rFunc(y_guess)
        if norm(r, Inf) < tol
            return y_guess[1:nx], y_guess[nx .+ (1:nλ)]
        end
    end

    @error "Newton solve did not converge. Final residual: " norm(r, Inf)
    return y_guess[1:nx], y_guess[nx .+ (1:nλ)]
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
nq, nv, nx, nu = 14, 13, 27, 3
tf = 5
h = 0.001
N = Int(tf/h + 1)

# Find initial balancing state and control
q0 = [1; zeros(13)]
q0[5:7] = -foot_pinned_c(q0, robot)
z = [0; 0; 1]
set_configuration!(state, q0)
com0 = normalize(center_of_mass(state).v)
quat = axis_angle_to_quat(-normalize(skew(z)*com0)*acos(z'*com0))
q0 = [quat; -foot_pinned_c([quat; zeros(10)], robot); zeros(7)]
x0 = [q0; zeros(nv)]
for i = 1:27
    print(x0[i])
    print(", ")
end
u0 = zeros(3)
_, λ0 = newton_implicit_midpoint(robot, x0, u0, h)
maximum(abs.(implicit_midpoint(robot, x0, x0, u0, λ0, h)))

# Linearization and calculating LQR gain
A_k = FiniteDiff.finite_difference_jacobian(x_k -> implicit_midpoint(robot, x_k, x0, u0, λ0, h), x0)*E(x0)
A_next = FiniteDiff.finite_difference_jacobian(x_next -> implicit_midpoint(robot, x0, x_next, u0, λ0, h), x0)*E(x0)
B_u = FiniteDiff.finite_difference_jacobian(u -> implicit_midpoint(robot, x0, x0, u, λ0, h), u0)
B_λ = FiniteDiff.finite_difference_jacobian(λ -> implicit_midpoint(robot, x0, x0, u0, λ, h), λ0)

A = A_next \ A_k
B_u = A_next \ B_u
B_λ = A_next \ B_λ

# Check condition numbers
cond(A)
cond(hcat([A^k*[B_u B_λ] for k = 0:26]...)) # Bad (not controllable)

# Constraints
C = FiniteDiff.finite_difference_jacobian(x_next -> constraints(robot, x_next), x0)*E(x0)

# Order = [qx, qy, qz, x, y, z, l0, l2, rw, rw, rw, l1, l3]
pos_cost = [1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 10.0; 10.0; 1.0; 1.0; 1.0; 1.0; 1.0]
vel_cost = [1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]
Q = spdiagm([pos_cost..., vel_cost...])
R = sparse(100*I(3))

K = constrained_ihlqr(A, B_u, B_λ, C, Q, R, Q, max_iters = 100000)

for k = 1:3
    for j = 1:26
        print(K[k, j])
        print(", ")
    end
    println(" ")
end

# Simulation
X = [zeros(nx) for _ = 1:N] 
U = [zeros(nu) for _ = 1:N - 1]
X[1] = copy(x0)
θ = 10*pi/180
X[1][1:4] = L_mult([cos(θ/2); sin(θ/2)*[1; 0; 0]...])*X[1][1:4]
X[1][5:7] -= foot_pinned_c(X[1][1:14], robot) # Make sure foot constraint is satisfied at start

for k = 1:N - 1
    Δx = state_error(X[k], x0)

    U[k] = u0 - K*Δx

    X[k + 1], _ = newton_implicit_midpoint(robot, X[k], U[k], h)

    # Check constraint violation
    if maximum(abs.(constraints(robot, X[k + 1]))) > 1e-2
        @error "Constraints violated, stopping sim" k constraints(robot, X[k + 1])
        [X[i] = copy(X[k + 1]) for i = k+2:N]
        break
    end
end

visualize!(mvis, LinRange(0, tf, N), X)
vis


open("data.txt", "w") do file
    for i = 1:N
        quat = X[i][1:4]
        axis_angle = quat_to_axis_angle(quat)
        write(file, string(axis_angle[1]) * " " * string(axis_angle[2]) * " " * string(axis_angle[3]) * "\n")
    end
end

open("rwspeed_leg.txt", "w") do file
    for i = 1:N
        rw_speed = X[i][23:24]
        write(file, string(rw_speed[1]) * " " * string(rw_speed[2]) * "\n")
    end
end