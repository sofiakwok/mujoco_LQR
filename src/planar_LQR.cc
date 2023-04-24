#include <stdbool.h> //for bool
#include <iostream>
#include <fstream>
#include <unistd.h> //for usleep
#include <math.h>
#include <cmath>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <chrono>
#include <vector>

#include "kalman.hpp"

using namespace std;
using namespace Eigen;

char filename[] = "../model/hopper/hopper_rev10_mjcf_fixed.xml";

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
double_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum ctrl;

// time-invariant K
double K[3] = {0, 0, 0};

int counter = 0;

//Kalman filter
MatrixXd A;
MatrixXd C;
MatrixXd Q_k;
MatrixXd R_k;
MatrixXd P_k;
KalmanFilter kf_x;
KalmanFilter kf_y;
VectorXd last_state_x(3);
VectorXd last_state_y(3);

vector<double> rw_x;
vector<double> rw_y;
vector<double> ctrl_rwx;
vector<double> ctrl_rwy;
vector<double> x_theta;
vector<double> y_theta;
vector<double> dot_thetax;
vector<double> dot_thetay;

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

MatrixXd LQR_controller(const mjModel* m, mjData* d)
{
    // update constants
    double g = 9.81;
    double mass = 8.5053868;
    double I_b = 1.78662281;
    double I_rw = 0.00607175;
    double L = 0.3;

    double a = I_b - I_rw; 
    //cout << "a: " << a << endl;
    double b = I_rw*(I_rw - I_b);

    Matrix3d A_cont;
    A_cont << 0, 1, 0, 
        mass*L*g/a, 0, 0,
        mass*L*g*I_rw/b, 0, 0;
    Matrix<double, 3, 1> B_cont = {{0}, {-1/a}, {-I_b/b}};

    // discretize continuous time model
    MatrixXd A_B(3, 4);
    A_B << A_cont, B_cont;
    MatrixXd discretize(4, 4);
    MatrixXd end_row(1, 4);
    end_row << 0, 0, 0, 0;
    discretize << A_B/200, end_row/200;
    MatrixXd expo;
    expo = discretize.exp();
    //cout << "expo: " << expo << endl;
    //getting 3x3 A matrix
    A = expo.block<3, 3>(0, 0);
    //cout << "A: " << A << endl;
    MatrixXd B = expo.block<3, 1>(0, 3);
    //cout << "B: " << B << endl;
    Matrix3d A_T = A.transpose();
    Matrix<double, 1, 3> B_T = B.transpose();

    Matrix<double, 1, 3> C = {1.0, 0, 0};
    Matrix<double, 3, 1> C_T = C.transpose();
    //not used at all
    Matrix<double, 3, 1> D = {0, 0, 0};

    Matrix3d Q;
    Q(0, 0) = 10;
    Q(1, 1) = 1;
    Q(2, 2) = 1;
    //cout << "Q: " << Q << endl;
    //this is sus
    Matrix<double, 1, 1> R;
    R(0, 0) = 100;
    Matrix3d P = Q;
    MatrixXd K;
    Matrix3d Pn;
    Matrix3d P2; 

    for (int ricatti = 2; ricatti < 1000; ricatti++){
        //backwards Ricatti recursion
        //change arbitary amount of timesteps here at some point
        for (int i = ricatti; i > 0; i--){
            // not using inv() here because R + B_T*P*B is a scalar
            K = (R + B_T*P*B).inverse()*B_T*P*A;
            //cout << "K: " <<K << endl;
            Pn = Q + A_T*P*(A - B*K);
            //cout << "Pn: " << Pn << endl;
            P2 = P;
            P = Pn;
        }

        if ((P - P2).norm() <= 1e-5){
            cout << "iters: " << ricatti << endl;
            cout << "K: " << K << endl;
            return K; 
        }
    }
    cout << "Did not converge" << endl;
    return K;
    
}

KalmanFilter Kalman(){
    double dt = 1/1000;
    C = MatrixXd::Identity(1, 3);
    Q_k = MatrixXd::Identity(3, 3) * 0.05;
    R_k = MatrixXd::Identity(1, 1);
    P_k = MatrixXd::Ones(3, 3) * 0.051;
    KalmanFilter kf(dt, A, C, Q_k, R_k, P_k);
    VectorXd x0 = VectorXd::Zero(3, 1);
    kf.init(0, x0);
    return kf;
}

void mycontroller(const mjModel* m, mjData* d)
{
    int bodyid;
    //getting current COM position and velocity
    //COM position is not right, need to add a body at COM
    mjtNum com_quat[4];
    mjtNum imu_pos[3];
    mjtNum vel_angles[3];
    mjtNum com_mat[9];
    bodyid = mj_name2id(m, mjOBJ_SITE, "imu");
    mju_copy(com_mat, d->site_xmat + bodyid, 9);
    mju_mat2Quat(com_quat, com_mat);
    //cout << com_quat[0] << " " << com_quat[1] << " " << com_quat[2] << " " << com_quat[3] << endl;
    mju_copy(imu_pos, d->site_xpos + bodyid, 3);
    cout << "imu: " << imu_pos[0] << " " << imu_pos[1] << " " << imu_pos[2] << endl;

    //rotating into body frame of hopper (45 deg)
    mjtNum ref_axis[3];
    ref_axis[0] = 1;
    ref_axis[1] = 0;
    ref_axis[2] = 0;
    mjtNum body_axis[3];
    mju_rotVecQuat(body_axis, ref_axis, com_quat);
    cout << "x axis: " << body_axis[0] << " " << body_axis[1] << " " << body_axis[2] << endl;

    //finding angle from z axis for x and y
    mjtNum reaction_angles[3];
    reaction_angles[0] = atan(imu_pos[0]/imu_pos[2]);
    reaction_angles[1] = atan(imu_pos[1]/imu_pos[2]);//atan(mju_sqrt(mju_pow(imu_pos[0], 2) + mju_pow(imu_pos[2], 2))/imu_pos[1]);
    reaction_angles[2] = 0;
    cout << "angles: " << reaction_angles[0] << " " << reaction_angles[1] << " " << reaction_angles[2] << endl;

    mjtNum theta_rot = atan(body_axis[1]/body_axis[0]);
    mjtNum body_quat[4];
    body_quat[0] = cos(theta_rot - M_PI_4/2);
    body_quat[1] = 0;
    body_quat[2] = 0;
    body_quat[3] = sin(theta_rot - M_PI_4/2);
    mjtNum angles[3];
    mju_rotVecQuat(angles, reaction_angles, body_quat);
    cout << "body: " << angles[0] << " " << angles[1] << " " << angles[2] << endl;
    
    //cout << "angles: " << angles[0] << " " << angles[1] << " " << angles[2] << endl;
    //cout << "angles: " << angles[0] << " " << angles[1] << " " << angles[2] << endl;

    //fix leg angles 
    //cout << "fixing leg angles" << endl;
    int joint_leg0 = mj_name2id(m, mjOBJ_JOINT, "Joint 0");
    int act_leg0 = mj_name2id(m, mjOBJ_ACTUATOR, "q0");
    d->ctrl[act_leg0] = -2000*d->qpos[m->jnt_qposadr[joint_leg0]] - 5*d->qvel[m->jnt_dofadr[joint_leg0]];
    int joint_leg2 = mj_name2id(m, mjOBJ_JOINT, "Joint 2");
    int act_leg2 = mj_name2id(m, mjOBJ_ACTUATOR, "q2");
    d->ctrl[act_leg2] = -2000*d->qpos[m->jnt_qposadr[joint_leg2]] - 5*d->qvel[m->jnt_dofadr[joint_leg2]];

    int actuator_x = mj_name2id(m, mjOBJ_ACTUATOR, "rw0");
    mjtNum xvel = d->actuator_velocity[actuator_x];
    int actuator_y = mj_name2id(m, mjOBJ_ACTUATOR, "rw1");
    mjtNum yvel = d->actuator_velocity[actuator_y];
    int actuator_z = mj_name2id(m, mjOBJ_ACTUATOR, "rwz");
    mjtNum zvel = d->actuator_velocity[actuator_z];
    rw_x.push_back(xvel);
    rw_y.push_back(yvel);
    cout << "rw speed: " << -xvel << " " << -yvel << " " << -zvel << endl;
    x_theta.push_back(angles[0]);
    y_theta.push_back(angles[1]);

    VectorXd meas(1);
    meas << angles[0];
    kf_x.update(meas);  
    meas << angles[1];
    kf_y.update(meas);
    
    vel_angles[0] = (kf_x.state()[0] - last_state_x[0])/0.001;
    vel_angles[1] = (kf_y.state()[1] - last_state_y[1])/0.001;
    cout << "theta dot: " << vel_angles[0] << " " << vel_angles[1] << " " << vel_angles[2] << endl;

    dot_thetax.push_back(vel_angles[0]);
    dot_thetay.push_back(vel_angles[1]);
    
    //running controller at 200 Hz
    if (counter % 5 == 0) {
        mjtNum state_x[3];
        mjtNum state_y[3];
        state_x[0] = angles[0];
        state_x[1] = vel_angles[0];
        state_x[2] = xvel;
        state_y[0] = angles[1];
        state_y[1] = vel_angles[1];
        state_y[2] = yvel;
                
        /*cout << "x: " << state_x[0] << " " << state_x[1] << " " << state_x[2] << endl;
        cout << "y: " << state_y[0] << " " << state_y[1] << " " << state_y[2] << endl;*/

        double ctrl_x;
        ctrl_x = mju_dot(K, state_x, 3);
        double ctrl_y;
        ctrl_y = mju_dot(K, state_y, 3);

        cout << "\n" << "K: " << K[0] << " " << K[1] << " " << K[2] << endl;
        cout << "ctrl: " << ctrl_x << " " << ctrl_y << endl;

        //reaction wheel 1 (x)
        d->ctrl[actuator_x] = -ctrl_x;

        //reaction wheel 2 (y)
        d->ctrl[actuator_y] = -ctrl_y;

        //cout << K[0] << " " << K[1] << " " << K[2] << endl;
        ctrl_rwx.push_back(-ctrl_x);
        ctrl_rwy.push_back(-ctrl_y);
        //ctrl_rwz.push_back(-ctrl_z);*/
        //cout << "done with controller" << endl;
    }
    counter += 1;
    last_state_x = kf_x.state();
    last_state_y = kf_y.state();
}

// main function
int main(int argc, const char** argv)
{
    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 )
        m = mj_loadXML(filename, 0, error, 1000);
    else
        if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
            m = mj_loadModel(argv[1], 0);
        else
            m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);

    mjtNum theta = 0;//0.17453/1.5; //10 degrees
    
    //change first 7 values of d to change starting position of hopper
    //changing xyz position
    //*
    d->qpos[0] = 0.3*sin(theta);
    d->qpos[1] = 0;//0.3*sin(theta);
    d->qpos[2] = 0.3*cos(theta);
    //changing quaternion 
    d->qpos[3] = cos(theta/2);
    d->qpos[4] = -sin(theta/2);
    d->qpos[5] = 0;
    d->qpos[6] = 0;//*/

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(2000, 1500, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    double arr_view[] = {90, -5, 5, 0.012768, -0.000000, 1.254336};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    //getting K matrix for LQR controls
    MatrixXd controls;
    controls = LQR_controller(m, d);

    //converting K matrix to K double
    Map<MatrixXd>(K, controls.rows(), controls.cols()) = controls;
    mjcb_control = mycontroller;

    //setting up Kalman filter
    kf_x = Kalman();
    kf_y = Kalman();

    // use the first while condition if you want to simulate for a period.
    while( !glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        //mj_forward(m, d);
        
        ///*
        while(d->time - simstart < 1.0/60.0)
        {
            mj_step(m, d);
        }//*/

        //get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

          // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        //printf("{%f, %f, %f, %f, %f, %f};\n",cam.azimuth,cam.elevation, cam.distance,cam.lookat[0],cam.lookat[1],cam.lookat[2]);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

    }

    std::ofstream myfile;
    myfile.open ("rwdata.csv");
    for (int i = 0; i < rw_x.size(); i++){
        myfile << to_string(rw_x[i]) + "," + to_string(rw_y[i]) + "\n";
    }
    myfile.close();

    std::ofstream ctrlfile;
    ctrlfile.open ("ctrldata.csv");
    for (int i = 0; i < ctrl_rwx.size(); i++){
        ctrlfile << to_string(ctrl_rwx[i]) + "," + to_string(ctrl_rwy[i]) + "\n";
    }
    ctrlfile.close();

    std::ofstream anglefile;
    anglefile.open ("angles.csv");
    for (int i = 0; i < x_theta.size(); i++){
        anglefile << to_string(x_theta[i]) + "," + to_string(y_theta[i]) + "\n";
    }
    anglefile.close();//*/

    std::ofstream dot_angle;
    dot_angle.open ("dotangles.csv");
    for (int i = 0; i < dot_thetax.size(); i++){
        dot_angle << to_string(dot_thetax[i]) + "," + to_string(dot_thetay[i]) + "\n";
    }
    dot_angle.close();//*/

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
    
}