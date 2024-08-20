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
CMAKE_SOURCE_DIR = /media/wayne/Disk/DECX_world/DECX_CV_CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/wayne/Disk/DECX_world/DECX_CV_CUDA/build

# Include any dependencies generated for this target.
include CMakeFiles/DECX_CV_CUDA.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/DECX_CV_CUDA.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/DECX_CV_CUDA.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DECX_CV_CUDA.dir/flags.make

CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o: CMakeFiles/DECX_CV_CUDA.dir/flags.make
CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o: /media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu
CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o: CMakeFiles/DECX_CV_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/DECX_CV_CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o -MF CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o.d -x cu -rdc=true -c /media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu -o CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o

CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target DECX_CV_CUDA
DECX_CV_CUDA_OBJECTS = \
"CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o"

# External object files for target DECX_CV_CUDA
DECX_CV_CUDA_EXTERNAL_OBJECTS =

CMakeFiles/DECX_CV_CUDA.dir/cmake_device_link.o: CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o
CMakeFiles/DECX_CV_CUDA.dir/cmake_device_link.o: CMakeFiles/DECX_CV_CUDA.dir/build.make
CMakeFiles/DECX_CV_CUDA.dir/cmake_device_link.o: CMakeFiles/DECX_CV_CUDA.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wayne/Disk/DECX_world/DECX_CV_CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/DECX_CV_CUDA.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DECX_CV_CUDA.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DECX_CV_CUDA.dir/build: CMakeFiles/DECX_CV_CUDA.dir/cmake_device_link.o
.PHONY : CMakeFiles/DECX_CV_CUDA.dir/build

# Object files for target DECX_CV_CUDA
DECX_CV_CUDA_OBJECTS = \
"CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o"

# External object files for target DECX_CV_CUDA
DECX_CV_CUDA_EXTERNAL_OBJECTS =

/media/wayne/Disk/DECX_world/bin/x64/libDECX_CV_CUDA.so: CMakeFiles/DECX_CV_CUDA.dir/media/wayne/Disk/DECX_world/srcs/modules/CV/NLM/NLM.cu.o
/media/wayne/Disk/DECX_world/bin/x64/libDECX_CV_CUDA.so: CMakeFiles/DECX_CV_CUDA.dir/build.make
/media/wayne/Disk/DECX_world/bin/x64/libDECX_CV_CUDA.so: CMakeFiles/DECX_CV_CUDA.dir/cmake_device_link.o
/media/wayne/Disk/DECX_world/bin/x64/libDECX_CV_CUDA.so: CMakeFiles/DECX_CV_CUDA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wayne/Disk/DECX_world/DECX_CV_CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA shared library /media/wayne/Disk/DECX_world/bin/x64/libDECX_CV_CUDA.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DECX_CV_CUDA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DECX_CV_CUDA.dir/build: /media/wayne/Disk/DECX_world/bin/x64/libDECX_CV_CUDA.so
.PHONY : CMakeFiles/DECX_CV_CUDA.dir/build

CMakeFiles/DECX_CV_CUDA.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DECX_CV_CUDA.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DECX_CV_CUDA.dir/clean

CMakeFiles/DECX_CV_CUDA.dir/depend:
	cd /media/wayne/Disk/DECX_world/DECX_CV_CUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wayne/Disk/DECX_world/DECX_CV_CUDA /media/wayne/Disk/DECX_world/DECX_CV_CUDA /media/wayne/Disk/DECX_world/DECX_CV_CUDA/build /media/wayne/Disk/DECX_world/DECX_CV_CUDA/build /media/wayne/Disk/DECX_world/DECX_CV_CUDA/build/CMakeFiles/DECX_CV_CUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DECX_CV_CUDA.dir/depend
