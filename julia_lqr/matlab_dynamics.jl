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

function mass_mat(I1,Iwm,Iwp,theta1,theta2)
    I11_1,I11_2,I11_3,I12_1,I12_2,I12_3,I13_1,I13_2,I13_3 = [I1[i] for i=1:9]
    Iwm1,Iwm2,Iwm3 = [Iwm[i] for i=1:3]
    Iwp1,Iwp2,Iwp3 = [Iwp[i] for i=1:3]
    t2 = cos(theta1)
    t3 = sin(theta1)
    t4 = I11_2*t2*(1.0/2.0)
    t5 = I12_1*t2*(1.0/2.0)
    t6 = t4+t5-I12_3*t3*(1.0/2.0)-I13_2*t3*(1.0/2.0)
    t8 = sin(theta2)
    t11 = t3*t8
    t7 = t2+t11
    t13 = t2*t8
    t9 = t3-t13
    t10 = cos(theta2)
    t12 = t2-t11
    t14 = t3+t13
    t15 = t10^2
    M = [I12_2 t6 0.0 0.0; 
    t6 t2*(I11_1*t2-I13_1*t3)-t3*(I11_3*t2-I13_3*t3) 0.0 0.0;
    0.0 0.0 Iwp2*t15*(1.0/2.0)+Iwp1*t7^2*(1.0/2.0)+Iwp3*t9^2*(1.0/2.0) 0.0;
    0.0 0.0 0.0 Iwm2*t15*(1.0/2.0)+Iwm1*t12^2*(1.0/2.0)+Iwm3*t14^2*(1.0/2.0)]
    return M
end

function acting_forces(I1,Iwm,Iwp,K,b1,b2,bwm,bwp,dth1,dth2,dth3,dth4,
        g,im,ip,k1,k2,lc11,lc12,lc13,m1,theta1,theta2)
    #"""Forces from symbolic Matlab"""
    I11_1,I11_2,I11_3,I12_1,I12_2,I12_3,I13_1,I13_2,I13_3 = [I1[i] for i=1:9]
    Iwm1,Iwm2,Iwm3 = [Iwm[i] for i=1:3]
    Iwp1,Iwp2,Iwp3 = [Iwp[i] for i=1:3]
    t2 = dth2^2
    t3 = dth4^2
    t4 = sin(theta2)
    t5 = dth3^2
    t6 = cos(theta1)
    t7 = t6^2
    t8 = theta1*2.0
    t9 = sin(t8)
    t10 = sqrt(2.0)
    t11 = cos(theta2)
    t12 = sin(theta1)
    t13 = t11^2
    t14 = cos(t8)
    t15 = t10*t12*(1.0/2.0)
    t16 = t4*t6*t10*(1.0/2.0)
    t17 = t6*t10*(1.0/2.0)
    t18 = K*ip
    t19 = K*im
    t20 = t12^2
    t21 = t6*t12*t13
    Vv = [[I11_3*t2*(1.0/2.0)+I13_1*t2*(1.0/2.0)-b1*dth1-k1*theta1-
        I11_1*t2*t9*(1.0/2.0)-I11_3*t2*t7-I13_1*t2*t7+
        I13_3*t2*t9*(1.0/2.0)+Iwm1*t3*t4*(1.0/2.0)-Iwm3*t3*t4*(1.0/2.0)-
        Iwp1*t4*t5*(1.0/2.0)+Iwp3*t4*t5*(1.0/2.0)-Iwm1*t3*t4*t7+Iwm3*t3*t4*t7+Iwp1*t4*t5*t7-
        Iwp3*t4*t5*t7-bwm*dth4*t10*t11*(1.0/2.0)+bwp*dth3*t10*t11*(1.0/2.0)-
        g*lc11*m1*t6+K*im*t10*t11*(1.0/2.0)-K*ip*t10*t11*(1.0/2.0)-
        Iwm1*t3*t6*t12*t13*(1.0/2.0)+Iwm3*t3*t6*t12*t13*(1.0/2.0)-
        Iwp1*t5*t6*t12*t13*(1.0/2.0)+Iwp3*t5*t6*t12*t13*(1.0/2.0)-g*lc12*m1*t4*t12-
        g*lc13*m1*t11*t12],[-b2*dth2+dth1*(I11_1*dth2*t9*2.0+
        I11_2*dth1*t12+I11_3*dth2*t14*2.0+I12_3*dth1*t6+I12_1*dth1*t12+
        I13_2*dth1*t6-I13_3*dth2*t9*2.0+I13_1*dth2*t14*2.0)*(1.0/2.0)-k2*theta2-
        t10*(t19-bwm*dth4)*(1.0/2.0)-t10*(t18-bwp*dth3)*(1.0/2.0)-g*m1*(lc13*t4*t6-lc12*t6*t11)-
        Iwm2*t3*t4*t11*(1.0/2.0)-Iwp2*t4*t5*t11*(1.0/2.0)-
        Iwm1*t3*t10*t11*t12*(t17-t4*t10*t12*(1.0/2.0))*(1.0/2.0)+
        Iwp1*t5*t10*t11*t12*(t17+t4*t10*t12*(1.0/2.0))*(1.0/2.0)-
        Iwp3*t5*t6*t10*t11*(t15-t16)*(1.0/2.0)+Iwm3*t3*t6*t10*t11*(t15+t16)*(1.0/2.0)],
        [t18-bwp*dth3-dth3*(dth2*t11*(-Iwp2*t4+Iwp1*t9*(1.0/2.0)-
        Iwp3*t9*(1.0/2.0)+Iwp3*t4*t7+Iwp1*t4*t20)-dth1*(Iwp1-Iwp3)*(t4+t21-t4*t7*2.0))],
        [t19-bwm*dth4+dth4*(dth1*(Iwm1-Iwm3)*(-t4+t21+t4*t7*2.0)-
        dth2*t11*(-Iwm2*t4-Iwm1*t9*(1.0/2.0)+Iwm3*t9*(1.0/2.0)+Iwm3*t4*t7+Iwm1*t4*t20))]]
    
    return Vv
end

function model_dynamics(x,u,sys)
    #Dynamics equation from Matlab symbolic

    M = mass_mat(sys.I1,sys.Iwm,sys.Iwp,x[1],x[2])
    V = acting_forces(sys.I1,sys.Iwm,sys.Iwp,sys.K,sys.b1,sys.b2,sys.bwm,sys.bwp,x[3],
            x[4],x[5],x[6], sys.g,u[1],u[2],sys.k1,sys.k2,sys.lc1[1],sys.lc1[2],sys.lc1[3],sys.m1,x[1],x[2])
    dq = M \ V
    dx::Vector = [x[3],x[4],dq[1],dq[2],dq[3],dq[4]]
    
    return dx
end

function rk4(params::NamedTuple,x::Vector,u,dt::Float64)
    # vanilla RK4
    k1 = dt*model_dynamics(x, u, params)
    k2 = dt*model_dynamics(x + k1/2, u, params)
    k3 = dt*model_dynamics(x + k2/2, u, params)
    k4 = dt*model_dynamics(x + k3, u, params)
    x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end

nu = 2
nx = 6

# desired x and g (linearize about these)
xgoal = [0.0, 0, 0, 0, 0, 0]
ugoal = [0, 0]

# initial condition (slightly off of our linearization point)
x0 = [0.0, 0, 0, 0, 0, 0] + [1.5, deg2rad(-20), .3, 0, 0, 0]

# simulation size 
dt = 0.1 
tf = 5.0 
t_vec = 0:dt:tf
N = length(t_vec)
X = [zeros(nx) for i = 1:N]
X[1] = x0 

params = (a=5,
lc1 = [0,0,0.33],
m1 = 1.3,
I1 = [0.18, 0, 0, 0, 0.18, 0, 0, 0, 0.01],
Iwm = [1e-9*753238.87,1e-9*394859.64,1e-9*394859.64],
Iwp = [1e-9*753238.87,1e-9*394859.64,1e-9*394859.64],
K = 131e-3,
k1 = 13.70937911565217391304347826087,
k2 = 6.0195591156521739130434782608696,
g = 9.81,
b1 = 0.7e-1,
b2 =2.5e-1,
bwm =1.1087e-4,
bwp=9.4514e-5)

# cost terms 
Q = diagm([1,1,.05,.1,1,1])
Qf = 1*Q
R = 0.1*diagm(ones(2))

Kinf = zeros(2,6) 

A = FD.jacobian(dx -> rk4(params, dx, ugoal, dt), xgoal)
B = FD.jacobian(du -> rk4(params, xgoal, du, dt), ugoal)

# TODO: solve for the infinite horizon LQR gain Kinf
    # Ricatti 
    n = 2
    max_iter = 1000
    tol = 1e-5
    K = zeros(nu,nx,n-1)
    for ricatti_iter = 1:max_iter 
        # instantiate P and K 
        P = zeros(nx,nx,n)
        K = zeros(nu,nx,n-1)
        P[:,:,n] .= Qf
        for k = (n-1):-1:1
            K[:,:,k] .= (R + B'*P[:,:,k+1]*B)\(B'*P[:,:,k+1]*A)
            P[:,:,k] .= Q + A'*P[:,:,k+1]*(A-B*K[:,:,k])
        end
        if norm(P[:, :, 2] - P[:, :, 1]) <= tol
            break
        end
        n += 1
    end
    Kinf = K[:, :, 1]

# TODO: simulate this controlled system with rk4(params_real, ...)
for k = 1:N-1 
    u = -Kinf * (X[k] - xgoal)
    X[k+1] = rk4(params, X[k], u, dt)
end

visualize!(mvis, LinRange(0, tf, N), X)
vis