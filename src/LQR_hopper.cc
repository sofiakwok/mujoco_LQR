#include <stdbool.h> //for bool
#include <iostream>
//#include<unistd.h> //for usleep
//#include <math.h>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
// #include <lapack>
using namespace std;
using namespace Eigen;

char filename[] = "../model/hopper/hopper_rev10_mjcf.xml";

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
    double rw_mass = 1;
    double I_p = 2;
    double I_rw = 2;
    double L = 1;
    double l = 0.75;

    double a = rw_mass*L*L + I_p; 
    //cout << "a: " << a << endl;
    double b = mass*l + rw_mass*L;

    Matrix<double, 3, 3> A;
    A(0, 1) = 1;
    A(1, 0) = b*g/a;
    A(2, 0) = -b*g/a;
    //cout << "A: " << A << endl; 
    Matrix<double, 3, 3> A_T = A.transpose();
    Matrix<double, 3, 1> B = {{0}, {-1/a}, {(a + I_rw)/(a*I_rw)}};
    //cout << "B: " << B << endl;
    Matrix<double, 1, 3> B_T = B.transpose();
    Matrix<double, 1, 3> C = {1.0, 0, 0};
    Matrix<double, 3, 1> C_T = C.transpose();
    //not used at all
    Matrix<double, 3, 1> D = {0, 0, 0};

    Matrix<double, 3, 3> Q;
    Q = C_T * C;
    //cout << "Q: " << Q << endl;
    //this is sus
    double R = 1;

    //backwards Ricatti recursion
    Matrix3d P = Q;
    MatrixXd K;
    Matrix3d Pn;
    //change arbitary amount of timesteps here at some point
    for (int i = 10; i > 0; i--){
        K = 1/(R + B_T*P*B)*B_T*P*A;
        //cout << "K: " << K << endl;
        Pn = Q + A_T*P*(A - B*K);
        //cout << "Pn: " << Pn << endl;
        P = Pn;
    }
    //cout << "K: " << K << endl;
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

    //0 = first leg joint
    actuator_no = mj_name2id(m, mjOBJ_ACTUATOR, "q0");
    // set_position_servo(m, actuator_no, 1000);
    d->ctrl[actuator_no] = 0;

    //1 = second leg joint
    actuator_no = mj_name2id(m, mjOBJ_ACTUATOR, "q2");
    // set_position_servo(m, actuator_no, 1000);
    d->ctrl[actuator_no] = 0;

    //fix leg angles 
    int joint_leg0 = mj_name2id(m, mjOBJ_JOINT, "Joint 0");
    int act_leg0 = mj_name2id(m, mjOBJ_ACTUATOR, "q0");
    d->ctrl[act_leg0] = -2000*d->qpos[m->jnt_qposadr[joint_leg0]] - 5*d->qvel[m->jnt_dofadr[joint_leg0]];
    int joint_leg2 = mj_name2id(m, mjOBJ_JOINT, "Joint 2");
    int act_leg2 = mj_name2id(m, mjOBJ_ACTUATOR, "q2");
    d->ctrl[act_leg2] = -2000*d->qpos[m->jnt_qposadr[joint_leg2]] - 5*d->qvel[m->jnt_dofadr[joint_leg2]];

    //getting K matrix for LQR controls
    MatrixXd controls;
    controls = LQR_controller(m, d);

    //converting K matrix to K double
    double K[3];
    Map<MatrixXd>(K, controls.rows(), controls.cols()) = controls;

    //getting current COM position and velocity
    //body name is not right, need to add a body at COM
    bodyid = mj_name2id(m, mjOBJ_BODY, "Link 2");
    int qposadr = -1, qveladr = -1;
    //7 number position data: gives current position in 3D followed by a unit quaternion
    qposadr = m->jnt_qposadr[m->body_jntadr[bodyid]];
    //6 number velocity data - gives linear velocity followed by angular velocity
    qveladr = m->jnt_dofadr[m->body_jntadr[bodyid]];


    //2 = reaction wheel 1 (x)
    actuator_no = mj_name2id(m, mjOBJ_ACTUATOR, "rw0");
    bodyid = mj_name2id(m, mjOBJ_BODY, "myfloatingbody");
    mjtNum state[2*m->nq];
    qveladr = m->jnt_dofadr[m->body_jntadr[actuator_no]];
    state[0] -= M_PI_2; // stand-up position
    mjtNum ctrl = mju_dot(K, state, 1);
    cout << "control (x): " << ctrl << endl;
    d->ctrl[actuator_no] = -ctrl;

    //3 = reaction wheel 2 (y)
    actuator_no = mj_name2id(m, mjOBJ_ACTUATOR, "rw1");
    bodyid = mj_name2id(m, mjOBJ_BODY, "myfloatingbody");
    ctrl = mju_dot(K, state, 2*m->nq);
    cout << "control (y): " << ctrl << endl;
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