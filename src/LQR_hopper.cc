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
#include "../include/mujoco/spline.h"

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
Matrix<double, 3, 9> K;

//last control output
double control[2] = {0, 0};

int counter = 0;

vector<double> rw_x;
vector<double> rw_y;
vector<double> rw_z;
vector<double> ctrl_rwx;
vector<double> ctrl_rwy;
vector<double> ctrl_rwz;
vector<double> x_theta;
vector<double> y_theta;
vector<double> z_theta;
vector<double> dot_thetax;
vector<double> dot_thetay;
vector<double> dot_thetaz;

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
    double l = 0.3;
    Matrix3d I_b;
    I_b << 1.78719272, -0.01520492, -0.00045163, 
            -0.01520942, 1.78662281, -0.00184158,
            -0.00045163, -0.00184158, 0.08749557;
    Matrix3d I_rw;
    I_rw << 0.00607175, 0, 0, 
            0, 0.00607175, 0,
            0, 0, 0.00067624;

    Matrix3d a = I_rw*(I_rw - I_b);
    Matrix3d b = I_b - I_rw;

    //cout << "calculating A" << endl;
    Matrix<double, 9, 9> A_cont;
    A_cont << MatrixXd::Zero(3, 3), MatrixXd::Identity(3, 3), MatrixXd::Zero(3, 3),
            mass*l*g*MatrixXd::Ones(3, 3)*b.inverse(), MatrixXd::Zero(3, 6),
            -mass*l*g*MatrixXd::Ones(3, 3)*b.inverse(), MatrixXd::Zero(3, 6);
    cout << "A_cont: \n" << A_cont << endl;
    Matrix<double, 9, 3> B_cont;
    B_cont << MatrixXd::Zero(3, 3), -MatrixXd::Ones(3, 3)*b.inverse(), -I_b*a.inverse();
    cout << "B_cont: " << B_cont << endl;

    // discretize continuous time model
    MatrixXd A_B(9, 12);
    A_B << A_cont, B_cont;
    MatrixXd discretize(12, 12);
    MatrixXd end_row = MatrixXd::Zero(3, 12);
    discretize << A_B/200, end_row;
    //cout << discretize << endl;
    MatrixXd expo;
    expo = discretize.exp();
    //cout << "expo: " << expo << endl;
    //getting 3x3 A matrix
    MatrixXd A;
    A = expo.block<9, 9>(0, 0);
    cout << "A: " << A << endl;
    MatrixXd B = expo.block<9, 3>(0, 9);
    cout << "B: " << B << endl;
    MatrixXd A_T = A.transpose();
    Matrix<double, 3, 9> B_T = B.transpose();

    MatrixXd Q = MatrixXd::Identity(9, 9);
    Q(0, 0) = 10;
    Q(1, 1) = 10;
    Q(2, 2) = 10;
    //cout << "Q: " << Q << endl;
    //this is sus
    Matrix<double, 3, 3> R;
    R(0, 0) = 1;
    R(1, 1) = 1;
    R(2, 2) = 1;
    MatrixXd P = 10*Q;
    MatrixXd K;
    MatrixXd Pn;
    MatrixXd P2; 

    Matrix<double, 3, 9> K_matlab;
    K_matlab << -65.2876, 6.2296, 352.4413, -0.1618, 0.7161, 1.1361, 0.9025, 0.4302, 0.0198, 
    28.5519, -26.0650, -14.4155, 0.1018, -1.8250, -0.1455, 0.1690, -0.3963, 0.9024, 
    136.2228, 95.9489, -76.3386, 0.0735, 3.5351, -0.3028, -0.3961, 0.8111, 0.4304;
    return K_matlab;

    for (int ricatti = 2; ricatti < 1000; ricatti++){
        //backwards Ricatti recursion
        for (int i = ricatti; i > 0; i--){
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
    
}

//*************************************
void set_position_servo(const mjModel* m, int actuator_no, double kp)
{
  //gets the leg motors to lock in place and not move
  m->actuator_gainprm[10*actuator_no+0] = kp;
  m->actuator_biasprm[10*actuator_no+1] = -kp;
}
//***********************************

void mycontroller(const mjModel* m, mjData* d)
{
    int bodyid;
    //getting current COM position and velocity
    //COM position is not right, need to add a body at COM
    bodyid = mj_name2id(m, mjOBJ_SITE, "imu");
    //gives current body frame orientation as a quaternion in Cartesian coordinates
    mjtNum com_mat[9];
    mjtNum com_pos[4];
    mju_copy(com_mat, d->site_xmat, 9);
    mju_mat2Quat(com_pos, com_mat);
    cout << "com quat: " << com_pos[0] << " " << com_pos[1] << " " << com_pos[2] << " " << com_pos[3] << endl;
    //adding noise to current quaternion orientation (modeling IMU noise)
    
    double noise;
    //multiplying starting position difference between ref position and current quaternion to find current COM
    mjtNum ref_com[3];
    ref_com[0] = 0;
    ref_com[1] = 0;
    ref_com[2] = 0.3;
    mjtNum delta_x[3];
    mju_rotVecQuat(delta_x, ref_com, com_pos);
    cout << "com est pos: " << delta_x[0] << " " << delta_x[1] << " " << delta_x[2] << endl;

    //finding angle from z axis for x and y
    mjtNum reaction_angles[3];
    mjtNum angles[3];
    angles[0] = atan(delta_x[0]/delta_x[2]);
    angles[1] = atan(delta_x[1]/delta_x[2]);//atan(mju_sqrt(mju_pow(delta_x[0], 2) + mju_pow(delta_x[2], 2))/delta_x[1]);
    angles[2] = 0;//atan(mju_sqrt(mju_pow(delta_x[0], 2) + mju_pow(delta_x[1], 2))/delta_x[2]);
    mju_rotVecQuat(reaction_angles, angles, com_pos);
    cout << "angles: " << reaction_angles[0] << " " << reaction_angles[1] << " " << reaction_angles[2] << endl;
    //cout << "angles: " << angles[0] << " " << angles[1] << " " << angles[2] << endl;

    //transforming into reaction wheel frame from x and y world frame axes
    /*mjtNum reaction_angles[3];
    mjtNum rotation_matrix[9];
    mju_zero(rotation_matrix, 9);
    mjtNum theta_rot = atan(angle_rot[1]/angle_rot[0]);
    //cout << "theta rot: " << theta_rot << endl;
    rotation_matrix[0] = cos(-theta_rot - M_PI/4);
    rotation_matrix[1] = sin(-theta_rot - M_PI/4);
    rotation_matrix[3] = -sin(-theta_rot - M_PI/4);
    rotation_matrix[4] = cos(-theta_rot - M_PI/4);
    rotation_matrix[8] = 1;
    mju_rotVecMat(reaction_angles, angles, rotation_matrix);*/

    //getting body angular velocities
    bodyid = mj_name2id(m, mjOBJ_BODY, "Link 1");
    mjtNum com_vel[6];
    mjtNum trans_vel[3];
    mjtNum vel_angles[3];
    mju_copy(com_vel, d->cvel + bodyid, 6);
    //cout << "com vel: " << com_vel[0] << " " << com_vel[1] << " " << com_vel[2] << " " << com_vel[3] << " " << com_vel[4] << " " << com_vel[5] << endl;
    trans_vel[0] = com_vel[3];
    trans_vel[1] = com_vel[4];
    trans_vel[2] = com_vel[5];
    mju_rotVecQuat(vel_angles, trans_vel, com_pos);
    //mju_rotVecMat(vel_angles, trans_vel, rotation_matrix);

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
    rw_x.push_back(xvel);
    rw_y.push_back(yvel);
    x_theta.push_back(reaction_angles[0]);
    y_theta.push_back(reaction_angles[1]);
    

    //running controller at 200 Hz
    if (counter % 5 == 0) {

        /*dot_thetax.push_back(xtheta_dot);
        dot_thetay.push_back(ytheta_dot);*/
        int actuator_x = mj_name2id(m, mjOBJ_ACTUATOR, "rw0");
        mjtNum xvel = d->actuator_velocity[actuator_x];
        int actuator_y = mj_name2id(m, mjOBJ_ACTUATOR, "rw1");
        mjtNum yvel = d->actuator_velocity[actuator_y];
        int actuator_z = mj_name2id(m, mjOBJ_ACTUATOR, "rwz");
        mjtNum zvel = d->actuator_velocity[actuator_z];

        Matrix<double, 9, 1> state;
        state << reaction_angles[0],reaction_angles[1],reaction_angles[2],vel_angles[0],vel_angles[1],vel_angles[2],-xvel,-yvel,-zvel;

        Matrix<double, 3, 1> ctrl;
        Matrix<double, 3, 9> K_matlab;
        K_matlab << -40.6038026360662, -4.68900462769708, 26.7403636401237, -0.0906376850286823 , 0.224196908063965, 0.162355415373592, 0.0327193698827690,0.0652,-0.0461,
                    62.9683902271361,2.64277184992056,-2.72798446765594,0.142987127227929,-0.642295913097646,-0.0993941905468881,0.00326971995829481,-0.1542,0.1732,
                    9.36444980550883,7.31882606957536,-20.3743982633689,0.0375068252883356,0.298005966853193,-0.0413294610419337,-0.0118339985925592,0.0653,0.00082555;
        ctrl = K_matlab * state;

        cout << ctrl[0] << " " << ctrl[1] << " " << ctrl[2]<< endl;

        //reaction wheel 1 (x)
        /*cout << "x angle: " << state[0] << endl;
        cout << "x angle ctrl: " << -K[0]*state[0] << endl;
        //cout << "x speed: " << state[1] << endl;
        cout << "x speed ctrl: " << -K[1]*state[1] << endl;
        cout << "rw speed (x): " << state[2] << endl;
        cout << "rw speed ctrl (x): " << -K[2]*state[2] << endl;
        cout << "control (x): " << -ctrl_x << endl;*/
        mjtNum ctrl_x = ctrl[0];
        //cout << "ctrl x: " << ctrl_x << endl;
        d->ctrl[actuator_x] = -ctrl_x;

        //reaction wheel 2 (y)
        /*cout << "y angle: " << state[0] << endl;
        cout << "y angle ctrl: " << -K[0]*state[0] << endl;
        //cout << "y speed: " << state[1] << endl;
        cout << "y speed ctrl: " << -K[1]*state[1] << endl;
        cout << "rw speed (y): " << state[2] << endl;
        cout << "rw speed ctrl (y): " << -K[2]*state[2] << endl;
        cout << "control (y): " << -ctrl_y << endl;*/
        mjtNum ctrl_y = ctrl[1];
        //cout << "ctrl y: " << ctrl_y << endl;
        d->ctrl[actuator_y] = -ctrl_y;

        mjtNum ctrl_z = ctrl[2];
        //cout << "ctrl z: " << ctrl_z << endl;
        d->ctrl[actuator_z] = -ctrl_z;//-ctrl_z;
        //cout << " " << endl;

        //cout << K[0] << " " << K[1] << " " << K[2] << endl;
        /*ctrl_rwx.push_back(-ctrl_x);
        ctrl_rwy.push_back(-ctrl_y);
        ctrl_rwz.push_back(-ctrl_z);*/
        //cout << "done with controller" << endl;
    }
    counter += 1;
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

    //d->qpos[0]=1.57; //pi/2
    //getting K matrix for LQR controls
    //K = LQR_controller(m, d);

    //converting K matrix to K double
    //Map<MatrixXd>(K, controls.rows(), controls.cols()) = controls;
    mjcb_control = mycontroller;

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