#include <stdbool.h> //for bool
#include <iostream>
#include <fstream> //for writing data
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

using namespace std;
using namespace Eigen;

char filename[] = "../model/hopper/hopper_ball_joint.xml";

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
Matrix<double, 3, 26> K;
//reference state x0
Matrix<double, 27, 1> x0;

int counter = 0;


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
    //from hopper_lqr_rw_only.jl

    Matrix<double, 3, 26> K0;
    /*K0 << -22.793985345802135, 17.575003406189058, -0.018361663885326045, 856.0918649420217, 847.4563306776349, -18.392043793212252, 0.45624015090188463, -0.5878875588305351, -0.09781386986829045, 2.6291775787154078e-6, -0.00023507800420089612, 0.1967680895180451, 0.3470610889145994, -4.457657291480159, 3.4811960530744805, 0.003908174273440726, 153.0329407274372, 153.0152348408098, -3.289592346096297, 0.08104868095417637, 0.022241280587210994, -0.13626206754646442, 0.00017934949605344556, -0.0002782394021163071, 0.034951941038798305, 0.126464279981237,
    22.487885168625603, 17.28259665812077, -0.6379866439447641, 856.5460079041366, -847.886619985325, -16.319574389295163, 0.47410708499651194, -0.4426355962460131, 2.0655571032067448e-6, -0.09781357075762927, 0.00030897350099205994, 0.16474914386048395, 0.32936966213236, 4.400546627908781, 3.4265480162746784, -0.12201293161324243, 153.02888638773834, -153.00673616414036, -2.9137520654269795, 0.08425826320377912, 0.04835345991234282, 0.0001790710116561888, -0.1362417989582578, 0.00036509096561455814, 0.02916504762473093, 0.12343545517311245, 
    0.08930502549463455, 0.013436828271801302, -0.0032929947132841206, -0.36098652174010576, 2.654319848945262, 0.0039113223628028755, 0.022371420477359325, -0.05851616479117369, -0.00021714712267665667, 0.0002851056299710807, 0.09157184252594307, 0.009446341768909461, -0.006714888446476814, -0.010520100276191791, -0.0010242730369673652, -0.0009665849426769278, -0.05316863264985474, 0.39170294162922803, 0.0005966044639223484, 0.00010795168249360339, -0.000409885110159991, -0.0002635457494057913, 0.00034581565255064695, 0.09214090442680467, 5.974648670213104e-5, -8.625054191780815e-5;
    */
    K0 << -4.90356948913481, 3.8951861579606764, 0.005844270589807783, 170.34276665443002, 167.57800735491963, -3.658312743580898, 0.09078087494946918, -0.11793839698019658, 5.460805529316093e-8, 4.955259358827839e-8, -2.2674186358438432e-10, 0.039129728810876674, 0.06849655633831478, -0.8765671185205933, 0.6856296658428891, 0.05972922947006955, 30.683639072834083, 30.487676887224882, -0.659338466442822, 0.01617161681155407, 0.0046852801599752214, -0.009876957678565172, -5.826996908062301e-5, 0.0006832438792173984, 0.00696762616742327, 0.025381772434397085,  
    4.844413539184518, 3.838757951838518, -0.13779595358173438, 170.4929784350663, -167.72363967855088, -3.2496480048416587, 0.09437974825863807, -0.08933205961801356, -3.317537930987948e-8, 2.7284793774057872e-8, 1.3245883573702266e-7, 0.03282005852585728, 0.06501377811893638, 0.8655051424744692, 0.6751477229962781, -0.09162273986650284, 30.693868551964563, -30.497047866105383, -0.5846639648259668, 0.01699033036965193, 0.009443606542035049, -5.881899836511901e-5, -0.009862079448117237, -0.0007706268057056371, 0.0058952602927408275, 0.024729131550726927,  
    -0.018697300660370664, -0.0020578120374804216, -0.030829612019044033, -0.1084707418065768, 0.7183579902248409, 0.0013148682479951386, 0.0002208795852754276, -0.0007393488270414448, 1.597580214576431e-9, 5.712972513426556e-10, -9.811118583173924e-10, 0.00011258753112437923, -0.00012060484139120766, 0.0033242762811544414, 0.0005922721598341097, -0.7636439185569888, -0.01939357092236926, 0.1298290520075583, 0.00023365917914972454, 0.0010495235888326117, -0.0027612877046800724, -0.0007127252430738774, 0.0008125518069897955, 0.0005096175830820992, 0.0004798991276432091, -0.00037048985559035843;
    
    cout << K0 << endl;

    return K0;
}

MatrixXd ref_pos()
{
    Matrix<double, 27, 1> x0;
    x0 << 0.999948475958916, 0.0006138456069342759, -0.010132552541783801, -0.0, 0.00017582778541125332, -0.0003617632294766306, 0.2946863026769917, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    //x0 << 0.999948475958916, 0.0006138456069342759, -0.010132552541783801, -0.0, -0.00607922, -0.000368288, 0.299938, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    return x0;
}

//*************************************
void set_position_servo(const mjModel* m, int actuator_no, double kp)
{
  //gets the leg motors to lock in place and not move
  m->actuator_gainprm[10*actuator_no+0] = kp;
  m->actuator_biasprm[10*actuator_no+1] = -kp;
}
//***********************************

MatrixXd skew(VectorXd q)
{
    Matrix3d skew;
    double v1 = q(0);
    double v2 = q(1);
    double v3 = q(2);
    skew << 0, -v3, v2, v3, 0, -v1, -v2, v1, 0;
    return skew;
}

MatrixXd L_mult(VectorXd q)
{
    double qs = q(0);
    VectorXd qv(3);
    qv << q(1), q(2), q(3);
    MatrixXd subset(3, 3);
    subset = qs*MatrixXd::Identity(3, 3) + skew(qv);
    MatrixXd stack(4, 4);
    stack.row(0) << q(0), -q(1), -q(2), -q(3);
    stack.col(0).tail(3) << qv(0), qv(1), qv(2);
    stack.block(1, 1, 3, 3) = subset;
    return stack;
}

VectorXd quat_to_axis_angle(VectorXd q)
{    
    double tol = 1e-12;
    double qs = q(0);
    VectorXd qv(3);
    qv << q(1), q(2), q(3);
    double norm_qv = qv.norm();
    
    if (norm_qv >= tol){
        double theta = 2*atan(norm_qv/qs);
        return theta*qv/norm_qv;
    }
    else {
        return VectorXd::Zero(3, 1);
    }
}

MatrixXd state_error(VectorXd x, VectorXd x0)
{
    Matrix<double, 26, 1> err;
    VectorXd x0_q(4);
    x0_q << x0(0), x0(1), x0(2), x0(3);
    VectorXd x_q(4);
    x_q << x(0), x(1), x(2), x(3);
    VectorXd quat_diff = L_mult(x0_q).transpose()*x_q;

    VectorXd x0_pos(23);
    VectorXd x_pos(23);

    x0_pos << x0(4), x0(5), x0(6), x0(7), x0(8), x0(9), x0(10), x0(11), x0(12), x0(13), x0(14), x0(15), x0(16), x0(17), x0(18), x0(19), x0(20), x0(21), x0(22), x0(23), x0(24), x0(25), x0(26);
    x_pos << x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11), x(12), x(13), x(14) , x(15), x(16), x(17), x(18), x(19), x(20), x(21), x(22), x(23), x(24), x(25), x(26);

    VectorXd pos_diff(23);
    pos_diff << x_pos - x0_pos;
    VectorXd axis_diff(3);
    axis_diff << quat_to_axis_angle(quat_diff);
    err << axis_diff, pos_diff;

    return err;
}


void mycontroller(const mjModel* m, mjData* d)
{
    //running controller at 200 Hz
    if (counter % 1 == 0) {
        int actuator_no;
        int bodyid;

        VectorXd x(27);
        //getting q
        bodyid = mj_name2id(m, mjOBJ_SITE, "imu");
        //gives current body frame orientation as a quaternion in Cartesian coordinates
        mjtNum com_mat[9];
        mjtNum com_quat[4];
        mjtNum com_pos[3];
        mju_copy(com_mat, d->site_xmat + 9*bodyid, 9);
        mju_mat2Quat(com_quat, com_mat);
        mju_copy(com_pos, d->site_xpos + 3*bodyid, 3);

        int link0 = mj_name2id(m, mjOBJ_JOINT, "Joint 0");
        mjtNum link_0;
        link_0 = d->qpos[m->jnt_qposadr[link0]];
        int link2 = mj_name2id(m, mjOBJ_JOINT, "Joint 2");
        mjtNum link_2;
        link_2 = d->qpos[m->jnt_qposadr[link2]];
        int rw0 = mj_name2id(m, mjOBJ_JOINT, "joint_rw0");
        mjtNum rwx;
        rwx = d->qpos[m->jnt_qposadr[rw0]];
        int rw1 = mj_name2id(m, mjOBJ_JOINT, "joint_rw1");
        mjtNum rwy;
        rwy = d->qpos[m->jnt_qposadr[rw1]];
        int rw2 = mj_name2id(m, mjOBJ_JOINT, "joint_rwz");
        mjtNum rwz;
        rwz = d->qpos[m->jnt_qposadr[rw2]];
        int link1 = mj_name2id(m, mjOBJ_JOINT, "Joint 1");
        mjtNum link_1;
        link_1 = d->qpos[m->jnt_qposadr[link1]];
        int link3 = mj_name2id(m, mjOBJ_JOINT, "Joint 3");
        mjtNum link_3;
        link_3 = d->qpos[m->jnt_qposadr[link3]];

        //COM velocity data - gives rotational velocity followed by translational velocity (6x1)
        mjtNum com_vel[6];
        mjtNum vel_angles[3];
        bodyid = mj_name2id(m, mjOBJ_JOINT, "base joint");
        mju_copy(com_vel, d->qvel + m->jnt_dofadr[bodyid], 6);
        //cout << com_vel[0] << " " << com_vel[1] << " " << com_vel[2] << " " << com_vel[3] << " " << com_vel[4] << " " << com_vel[5] << endl;
        mjtNum link_0_vel;
        mjtNum link_2_vel;
        mjtNum rwx_vel;
        mjtNum rwy_vel;
        mjtNum rwz_vel;
        mjtNum link_1_vel;
        mjtNum link_3_vel;
        link_0_vel = d->qvel[m->jnt_dofadr[link0]];
        link_2_vel = d->qvel[m->jnt_dofadr[link2]];
        rwx_vel = d->qvel[m->jnt_dofadr[rw0]];
        rwy_vel = d->qvel[m->jnt_dofadr[rw1]];
        rwz_vel = d->qvel[m->jnt_dofadr[rw2]];
        link_1_vel = d->qvel[m->jnt_dofadr[link1]];
        link_3_vel = d->qvel[m->jnt_dofadr[link3]];

        x << com_quat[0], com_quat[1], com_quat[2], com_quat[3], com_pos[0], com_pos[1], com_pos[2], link_0, link_2, rwx, rwy, rwz, link_1, link_3, 
        com_vel[0], com_vel[1], com_vel[2], com_vel[3], com_vel[4], com_vel[5], link_0_vel, link_2_vel, rwx_vel, rwy_vel, rwz_vel, link_1_vel, link_3_vel;

        Matrix<double, 26, 1> delta_x;
        delta_x = state_error(x, x0);

        cout << "x: " << x.transpose() << endl;
        if (counter == 0){
            cout << "x: " << x << endl;
        }

        Matrix<double, 3, 1> ctrl;
        ctrl = -K * delta_x;
        cout << "ctrl: " << ctrl << endl;
        if (counter == 0){
            cout << "ctrl: " << ctrl << endl;
        }

        //reaction wheel 1 (x)
        int actuator_x = mj_name2id(m, mjOBJ_ACTUATOR, "rw0");
        mjtNum ctrl_x = ctrl[0];
        d->ctrl[actuator_x] = ctrl_x;

        //reaction wheel 2 (y)
        int actuator_y = mj_name2id(m, mjOBJ_ACTUATOR, "rw1");
        mjtNum ctrl_y = ctrl[1];
        d->ctrl[actuator_y] = ctrl_y;

        int actuator_z = mj_name2id(m, mjOBJ_ACTUATOR, "rwz");
        mjtNum ctrl_z = ctrl[2];
        d->ctrl[actuator_z] = ctrl_z;
    }

    //fix leg angles 
    int joint_leg0 = mj_name2id(m, mjOBJ_JOINT, "Joint 0");
    int act_leg0 = mj_name2id(m, mjOBJ_ACTUATOR, "q0");
    d->ctrl[act_leg0] = -2000*d->qpos[m->jnt_qposadr[joint_leg0]] - 5*d->qvel[m->jnt_dofadr[joint_leg0]];
    int joint_leg2 = mj_name2id(m, mjOBJ_JOINT, "Joint 2");
    int act_leg2 = mj_name2id(m, mjOBJ_ACTUATOR, "q2");
    d->ctrl[act_leg2] = -2000*d->qpos[m->jnt_qposadr[joint_leg2]] - 5*d->qvel[m->jnt_dofadr[joint_leg2]];

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
    
    //change first 7 values of d to change starting position of hopper
    //changing quaternion 
    d->qpos[0] = 0.999948475958916;
    d->qpos[1] = 0.0006138456069342759;
    d->qpos[2] = -0.010132552541783801;
    d->qpos[3] = -0.0;

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
    K = LQR_controller(m, d);
    x0 = ref_pos();

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

    /*std::ofstream myfile;
    myfile.open ("rwdata.csv");
    for (int i = 0; i < rw_x.size(); i++){
        myfile << to_string(rw_x[i]) + "," + to_string(rw_y[i]) + ", " + to_string(rw_z[i]) + "\n";
    }
    myfile.close();

    std::ofstream ctrlfile;
    ctrlfile.open ("ctrldata.csv");
    for (int i = 0; i < ctrl_rwx.size(); i++){
        ctrlfile << to_string(ctrl_rwx[i]) + "," + to_string(ctrl_rwy[i]) + ", " + to_string(ctrl_rwz[i]) + "\n";
    }
    ctrlfile.close();

    std::ofstream anglefile;
    anglefile.open ("angles.csv");
    for (int i = 0; i < x_theta.size(); i++){
        anglefile << to_string(x_theta[i]) + "," + to_string(y_theta[i]) + ", " + to_string(z_theta[i]) + "\n";
    }
    anglefile.close();

    std::ofstream dot_angle;
    dot_angle.open ("dotangles.csv");
    for (int i = 0; i < dot_thetax.size(); i++){
        dot_angle << to_string(dot_thetax[i]) + "," + to_string(dot_thetay[i]) + ", " + to_string(dot_thetaz[i]) + "\n";
    }
    dot_angle.close();
    */

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