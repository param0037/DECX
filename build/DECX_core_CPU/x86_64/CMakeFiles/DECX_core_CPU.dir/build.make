# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/wayne/Disk/DECX_world/srcs/modules/core

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64

# Include any dependencies generated for this target.
include CMakeFiles/DECX_core_CPU.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/DECX_core_CPU.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/DECX_core_CPU.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DECX_core_CPU.dir/flags.make

CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/global_vars.cxx
CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/global_vars.cxx

CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/global_vars.cxx > CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.i

CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/global_vars.cxx -o CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.s

CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/pre_post_proc_CPUcoreModule.cxx
CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/pre_post_proc_CPUcoreModule.cxx

CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/pre_post_proc_CPUcoreModule.cxx > CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.i

CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/pre_post_proc_CPUcoreModule.cxx -o CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.s

CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Matrix.cxx
CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Matrix.cxx

CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Matrix.cxx > CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.i

CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Matrix.cxx -o CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.s

CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/classes/MatrixArray.cxx
CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/classes/MatrixArray.cxx

CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/classes/MatrixArray.cxx > CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.i

CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/classes/MatrixArray.cxx -o CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.s

CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Tensor.cxx
CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Tensor.cxx

CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Tensor.cxx > CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.i

CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Tensor.cxx -o CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.s

CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/classes/TensorArray.cxx
CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/classes/TensorArray.cxx

CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/classes/TensorArray.cxx > CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.i

CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/classes/TensorArray.cxx -o CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.s

CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Vector.cxx
CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Vector.cxx

CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Vector.cxx > CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.i

CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/classes/Vector.cxx -o CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.s

CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/ResMgr.cxx
CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/ResMgr.cxx

CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/ResMgr.cxx > CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.i

CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/ResMgr.cxx -o CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.s

CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/decx_resource.cxx
CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/decx_resource.cxx

CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/decx_resource.cxx > CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.i

CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/resources_manager/decx_resource.cxx -o CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.s

CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/task_queue.cxx
CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/task_queue.cxx

CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/task_queue.cxx > CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.i

CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/task_queue.cxx -o CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.s

CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/thread_pool.cxx
CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/thread_pool.cxx

CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/thread_pool.cxx > CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.i

CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/thread_management/thread_pool.cxx -o CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.s

CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o: CMakeFiles/DECX_core_CPU.dir/flags.make
CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/core/configs/config.cxx
CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o: CMakeFiles/DECX_core_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o -MF CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o.d -o CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/core/configs/config.cxx

CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/core/configs/config.cxx > CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.i

CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/core/configs/config.cxx -o CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.s

# Object files for target DECX_core_CPU
DECX_core_CPU_OBJECTS = \
"CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o" \
"CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o"

# External object files for target DECX_core_CPU
DECX_core_CPU_EXTERNAL_OBJECTS =

/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/global_vars.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/pre_post_proc_CPUcoreModule.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/classes/Matrix.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/classes/MatrixArray.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/classes/Tensor.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/classes/TensorArray.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/classes/Vector.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/resources_manager/ResMgr.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/resources_manager/decx_resource.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/thread_management/task_queue.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/thread_management/thread_pool.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/configs/config.cxx.o
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/build.make
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: GNU/x86_64/liballocators_host.a
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: GNU/x86_64/libconfigs_x86_64_cpu.a
/media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so: CMakeFiles/DECX_core_CPU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX shared library /media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DECX_core_CPU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DECX_core_CPU.dir/build: /media/wayne/Disk/DECX_world/build/bin/x86_64/libDECX_core_CPU.so
.PHONY : CMakeFiles/DECX_core_CPU.dir/build

CMakeFiles/DECX_core_CPU.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DECX_core_CPU.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DECX_core_CPU.dir/clean

CMakeFiles/DECX_core_CPU.dir/depend:
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wayne/Disk/DECX_world/srcs/modules/core /media/wayne/Disk/DECX_world/srcs/modules/core /media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64 /media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64 /media/wayne/Disk/DECX_world/build/DECX_core_CPU/x86_64/CMakeFiles/DECX_core_CPU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DECX_core_CPU.dir/depend

