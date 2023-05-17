# set(BIN_NAME LQR_ball)
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

# include_directories(LQR_ball ${LAPACK_INCLUDES})
# include_directories(LQR_ball ${LIB_ARMADILLO})
set(SOURCE_FILES kalman.cpp LQR_ball.cc)
add_executable(LQR_ball ${SOURCE_FILES})
target_link_libraries(LQR_ball ${LIB_MUJOCO})
target_link_libraries(LQR_ball glfw)
target_link_libraries(LQR_ball Eigen3::Eigen)

#target_link_libraries(LQR_ball lapack)