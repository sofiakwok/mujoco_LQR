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
include src/CMakeFiles/planar_LQR.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/planar_LQR.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/planar_LQR.dir/flags.make

src/CMakeFiles/planar_LQR.dir/planar_LQR.cc.o: src/CMakeFiles/planar_LQR.dir/flags.make
src/CMakeFiles/planar_LQR.dir/planar_LQR.cc.o: ../src/planar_LQR.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sofia/mujoco_LQR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/planar_LQR.dir/planar_LQR.cc.o"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/planar_LQR.dir/planar_LQR.cc.o -c /home/sofia/mujoco_LQR/src/planar_LQR.cc

src/CMakeFiles/planar_LQR.dir/planar_LQR.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/planar_LQR.dir/planar_LQR.cc.i"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sofia/mujoco_LQR/src/planar_LQR.cc > CMakeFiles/planar_LQR.dir/planar_LQR.cc.i

src/CMakeFiles/planar_LQR.dir/planar_LQR.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/planar_LQR.dir/planar_LQR.cc.s"
	cd /home/sofia/mujoco_LQR/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sofia/mujoco_LQR/src/planar_LQR.cc -o CMakeFiles/planar_LQR.dir/planar_LQR.cc.s

# Object files for target planar_LQR
planar_LQR_OBJECTS = \
"CMakeFiles/planar_LQR.dir/planar_LQR.cc.o"

# External object files for target planar_LQR
planar_LQR_EXTERNAL_OBJECTS =

../bin/planar_LQR: src/CMakeFiles/planar_LQR.dir/planar_LQR.cc.o
../bin/planar_LQR: src/CMakeFiles/planar_LQR.dir/build.make
../bin/planar_LQR: ../lib/libmujoco.so.2.3.2
../bin/planar_LQR: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
../bin/planar_LQR: src/CMakeFiles/planar_LQR.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sofia/mujoco_LQR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/planar_LQR"
	cd /home/sofia/mujoco_LQR/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/planar_LQR.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/planar_LQR.dir/build: ../bin/planar_LQR

.PHONY : src/CMakeFiles/planar_LQR.dir/build

src/CMakeFiles/planar_LQR.dir/clean:
	cd /home/sofia/mujoco_LQR/build/src && $(CMAKE_COMMAND) -P CMakeFiles/planar_LQR.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/planar_LQR.dir/clean

src/CMakeFiles/planar_LQR.dir/depend:
	cd /home/sofia/mujoco_LQR/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sofia/mujoco_LQR /home/sofia/mujoco_LQR/src /home/sofia/mujoco_LQR/build /home/sofia/mujoco_LQR/build/src /home/sofia/mujoco_LQR/build/src/CMakeFiles/planar_LQR.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/planar_LQR.dir/depend

