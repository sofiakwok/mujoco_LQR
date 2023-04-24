# set(BIN_NAME planar_LQR)
# set(EXEC_NAME LQR_hop)

# find_package(glfw3 REQUIRED)
# #include_directories(${GLFW_INCLUDE_DIRS})
# #link_libraries(${GLFW_LIBRARY_DIRS})

# #find_package(GLEW REQUIRED)
# #include_directories(${GLEW_INCLUDE_DIRS})

# add_executable(${EXEC_NAME} ${BIN_NAME}.cc)
# target_link_libraries(${EXEC_NAME} ${LIB_MUJOCO})

# #Standard libraries for GL
# target_link_libraries(${EXEC_NAME} GL GLU glut )

# #Additional libraries from mujoco package
# target_link_libraries(${EXEC_NAME} ${GLEW_LIBRARIES} libglfw.so)


find_package(glfw3 REQUIRED)
include_directories(${glfw_INCLUDE_DIRS})

find_package(Eigen3 3.3 REQUIRED)
INCLUDE_DIRECTORIES ($ENV{EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen: " ${EIGEN3_INCLUDE_DIRS})

#find_package(BLAS REQUIRED)
#find_package(LAPACK)

# file(GLOB LAPACK_INCLUDES /usr/lib/x86_64-linux-gnu/lapack/*)
# file(GLOB LIB_ARMADILLO /usr/lib/x86_64-linux-gnu/libarmadillo.so)

# include_directories(planar_LQR ${LAPACK_INCLUDES})
# include_directories(planar_LQR ${LIB_ARMADILLO})
set(SOURCE_FILES kalman.cpp planar_LQR.cc)
add_executable(planar_LQR ${SOURCE_FILES})
target_link_libraries(planar_LQR ${LIB_MUJOCO})
target_link_libraries(planar_LQR glfw)
target_link_libraries(planar_LQR Eigen3::Eigen)

#target_link_libraries(planar_LQR lapack)