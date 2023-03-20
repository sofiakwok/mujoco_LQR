#include <stdbool.h> //for bool
#include <iostream>
//#include<unistd.h> //for usleep
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
// #include <lapack>
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
    double mass = 1;
    double rw_mass = 0.5;
    double I_p = 2;
    double I_rw = 1;
    double L = 3;
    double l = 0.75;

    double a = rw_mass*L*L + I_p; 
    //cout << "a: " << a << endl;
    double b = mass*l + rw_mass*L;

    Matrix3d A_cont;
    A_cont << 0, 1, 0, 
        b*g/a, 0, 0,
        -b*g/a, 0, 0;
    Matrix<double, 3, 1> B_cont = {{0}, {-1/a}, {(a + I_rw)/(a*I_rw)}};

    // discretize continuous time model
    MatrixXd A_B(3, 4);
    A_B << A_cont, B_cont;
    MatrixXd discretize(4, 4);
    MatrixXd end_row(1, 4);
    end_row << 0, 0, 0, 0;
    discretize << A_B/60, end_row/60;
    MatrixXd expo;
    expo = discretize.exp();
    //cout << "expo: " << expo << endl;
    //getting 3x3 A matrix
    MatrixXd A;
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

    MatrixXd Q = C_T * C;
    Q(0, 0) = 100;
    Q(1, 1) = 100;
    Q(2, 2) = 100;
    //cout << "Q: " << Q << endl;
    //this is sus
    Matrix<double, 1, 1> R;
    R(0, 0) = 0.001;
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

        if ((P - P2).norm() <= 1e-10){
            cout << "iters: " << ricatti << endl;
            cout << "K: " << K << endl;
            return K; 
        }
    }
    cout << "Did not converge" << endl;
    return K;
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
    int actuator_no;
    int bodyid;

    //fix leg angles 
    int joint_leg0 = mj_name2id(m, mjOBJ_JOINT, "Joint 0");
    int act_leg0 = mj_name2id(m, mjOBJ_ACTUATOR, "q0");
    d->ctrl[act_leg0] = -2000*d->qpos[m->jnt_qposadr[joint_leg0]] - 5*d->qvel[m->jnt_dofadr[joint_leg0]];
    int joint_leg2 = mj_name2id(m, mjOBJ_JOINT, "Joint 2");
    int act_leg2 = mj_name2id(m, mjOBJ_ACTUATOR, "q2");
    d->ctrl[act_leg2] = -2000*d->qpos[m->jnt_qposadr[joint_leg2]] - 5*d->qvel[m->jnt_dofadr[joint_leg2]];

    //getting current COM position and velocity
    //COM position is not right, need to add a body at COM
    //bodyid = mj_name2id(m, mjOBJ_BODY, "Link 0");
    bodyid = mj_name2id(m, mjOBJ_BODY, "COM");
    //gives current body frame orientation as a quaternion in Cartesian coordinates
    mjtNum com_pos[4];
    mju_copy(com_pos, d->xquat + m->jnt_qposadr[m->body_jntadr[bodyid]], 4);
    //defining reference quaternion (pointing straight up, at (0, 0) where foot contacts ground)
    mjtNum quat_ref[4];
    quat_ref[0] = 0;
    quat_ref[1] = 0;
    quat_ref[2] = 1;
    quat_ref[3] = 0;
    //finding difference between reference quaternion and current COM orientation, converting to 3D coordinates
    mjtNum delta_x[3];
    mju_subQuat(delta_x, com_pos, quat_ref);
    //velocity data - gives rotational velocity followed by translational velocity (6x1)
    //finding COM velocity
    mjtNum com_vel[6];
    mju_copy(com_vel, d->cvel + m->jnt_qposadr[m->body_jntadr[bodyid]], 6);

    /*cout << "com (x): " << com_pos[0] << endl;
    cout << "com (y): " << com_pos[1] << endl;
    cout << "com (z): " << com_pos[2] << endl;
    cout << "com (w): " << com_pos[3] << endl;*/
    //cout << "delta (x): " << delta_x[0] << endl;
    //cout << "delta (y): " << delta_x[1] << endl;
    //cout << "delta (z): " << delta_x[2] << endl;
    /*cout << "com vel (x): " << com_vel[3] << endl;
    cout << "com vel (y): " << com_vel[4] << endl;
    cout << "com vel (z): " << com_vel[5] << endl;*/

    //2 = reaction wheel 1 (x)
    actuator_no = mj_name2id(m, mjOBJ_ACTUATOR, "rw0");
    int body_rw0 = mj_name2id(m, mjOBJ_BODY, "rw0");
    mjtNum state[3];
    int xveladr = -1;
    xveladr = m->jnt_dofadr[m->body_jntadr[body_rw0]];
    mjtNum xvel = d->qvel[xveladr];
    //cout << "rw speed (x): " << xvel << endl;
    state[0] = delta_x[0] - delta_x[2];
    state[1] = com_vel[3];
    state[2] = xvel;
    mjtNum ctrl = mju_dot(K, state, 1);
    //cout << "control (x): " << ctrl << endl;
    d->ctrl[actuator_no] = -ctrl;

    //3 = reaction wheel 2 (y)
    actuator_no = mj_name2id(m, mjOBJ_ACTUATOR, "rw1");
    int body_rw1 = mj_name2id(m, mjOBJ_BODY, "rw1");
    int yveladr = -1;
    yveladr = m->jnt_dofadr[m->body_jntadr[body_rw1]];
    mjtNum yvel = d->qvel[yveladr];
    //cout << "rw speed (y): " << yvel << endl;
    state[0] = delta_x[0] + delta_x[2];
    state[1] = com_vel[4];
    state[2] = yvel;
    ctrl = mju_dot(K, state, 1);
    //cout << "control (y): " << ctrl << endl;
    d->ctrl[actuator_no] = -ctrl;
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

    d->qpos[0]=1.57; //pi/2
    //getting K matrix for LQR controls
    MatrixXd controls;
    controls = LQR_controller(m, d);

    //converting K matrix to K double
    Map<MatrixXd>(K, controls.rows(), controls.cols()) = controls;
    mjcb_control = mycontroller;

    // use the first while condition if you want to simulate for a period.
    while( !glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
        {
            mj_step(m, d);
        }

       // get framebuffer viewport
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