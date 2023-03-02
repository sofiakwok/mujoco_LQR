cmake_minimum_required(VERSION 3.0)
project(LQR_hopper)
set(USE_GL true)

#Mujoco paths
set(mujoco_PATH ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "MuJoCo path: " ${mujoco_PATH})
include_directories(${mujoco_PATH}/include)
include_directories(${mujoco_PATH}/src)
link_directories(${mujoco_PATH}/lib)

set(${CMAKE_CURRENT_SOURCE_DIR}_SOURCES
    src/LQR_hopper.cpp
)

#Finding main mujoco library
if(${USE_GL})
    message("using GL")
    file(GLOB LIB_MUJOCO ${mujoco_PATH}/lib/libmujoco.so.[0-9].[0-9].[0-9])
else()
    file(GLOB LIB_MUJOCO ${mujoco_PATH}/lib/libmujoco[0-9][0-9][0-9]nogl.so)
endif()
#Showing mujoco library found
message(STATUS "MuJoCo lib: " ${LIB_MUJOCO})

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_subdirectory(src)

#target_link_libraries(LQR_hopper PUBLIC mujoco)