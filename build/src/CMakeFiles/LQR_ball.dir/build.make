# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sofia/mujoco_LQR

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sofia/mujoco_LQR/build

# Include any dependencies generated for this target.
include src/CMakeFiles/LQR_ball.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/LQR_ball.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/LQR_ball.dir/flags.make

src/CMakeFiles/LQR_ball.dir/kalman.cpp.o: src/CMakeFiles/LQR_ball.dir/flags.make
src/CMakeFiles/LQR_ball.dir/kalman.cpp.o: ../src/kalman.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sofia/mujoco_LQR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/LQR_ball.dir/kalman.cpp.o"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LQR_ball.dir/kalman.cpp.o -c /home/sofia/mujoco_LQR/src/kalman.cpp

src/CMakeFiles/LQR_ball.dir/kalman.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LQR_ball.dir/kalman.cpp.i"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sofia/mujoco_LQR/src/kalman.cpp > CMakeFiles/LQR_ball.dir/kalman.cpp.i

src/CMakeFiles/LQR_ball.dir/kalman.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LQR_ball.dir/kalman.cpp.s"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sofia/mujoco_LQR/src/kalman.cpp -o CMakeFiles/LQR_ball.dir/kalman.cpp.s

src/CMakeFiles/LQR_ball.dir/LQR_ball.cc.o: src/CMakeFiles/LQR_ball.dir/flags.make
src/CMakeFiles/LQR_ball.dir/LQR_ball.cc.o: ../src/LQR_ball.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sofia/mujoco_LQR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/LQR_ball.dir/LQR_ball.cc.o"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LQR_ball.dir/LQR_ball.cc.o -c /home/sofia/mujoco_LQR/src/LQR_ball.cc

src/CMakeFiles/LQR_ball.dir/LQR_ball.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LQR_ball.dir/LQR_ball.cc.i"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sofia/mujoco_LQR/src/LQR_ball.cc > CMakeFiles/LQR_ball.dir/LQR_ball.cc.i

src/CMakeFiles/LQR_ball.dir/LQR_ball.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LQR_ball.dir/LQR_ball.cc.s"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sofia/mujoco_LQR/src/LQR_ball.cc -o CMakeFiles/LQR_ball.dir/LQR_ball.cc.s

# Object files for target LQR_ball
LQR_ball_OBJECTS = \
"CMakeFiles/LQR_ball.dir/kalman.cpp.o" \
"CMakeFiles/LQR_ball.dir/LQR_ball.cc.o"

# External object files for target LQR_ball
LQR_ball_EXTERNAL_OBJECTS =

../bin/LQR_ball: src/CMakeFiles/LQR_ball.dir/kalman.cpp.o
../bin/LQR_ball: src/CMakeFiles/LQR_ball.dir/LQR_ball.cc.o
../bin/LQR_ball: src/CMakeFiles/LQR_ball.dir/build.make
../bin/LQR_ball: ../lib/libmujoco.so.2.3.2
../bin/LQR_ball: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
../bin/LQR_ball: src/CMakeFiles/LQR_ball.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sofia/mujoco_LQR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../bin/LQR_ball"
	cd /home/sofia/mujoco_LQR/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LQR_ball.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/LQR_ball.dir/build: ../bin/LQR_ball

.PHONY : src/CMakeFiles/LQR_ball.dir/build

src/CMakeFiles/LQR_ball.dir/clean:
	cd /home/sofia/mujoco_LQR/build/src && $(CMAKE_COMMAND) -P CMakeFiles/LQR_ball.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/LQR_ball.dir/clean

src/CMakeFiles/LQR_ball.dir/depend:
	cd /home/sofia/mujoco_LQR/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sofia/mujoco_LQR /home/sofia/mujoco_LQR/src /home/sofia/mujoco_LQR/build /home/sofia/mujoco_LQR/build/src /home/sofia/mujoco_LQR/build/src/CMakeFiles/LQR_ball.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/LQR_ball.dir/depend
