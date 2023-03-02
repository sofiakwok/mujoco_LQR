# set(BIN_NAME LQR_hopper)
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


add_executable(LQR_hopper LQR_hopper.cc)
target_link_libraries(LQR_hopper ${LIB_MUJOCO})
target_link_libraries(LQR_hopper glfw)