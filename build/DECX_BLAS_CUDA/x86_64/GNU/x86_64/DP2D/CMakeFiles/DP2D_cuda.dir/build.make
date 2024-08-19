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
CMAKE_SOURCE_DIR = /media/wayne/Disk/DECX_world/srcs/modules/BLAS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64

# Include any dependencies generated for this target.
include GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/compiler_depend.make

# Include the progress variables for this target.
include GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/flags.make

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/flags.make
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o: /media/wayne/Disk/DECX_world/srcs/modules/BLAS/Dot\ product/CUDA/2D/DP2D_1way_callers.cu
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o -MF CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o.d -x cu -rdc=true -c "/media/wayne/Disk/DECX_world/srcs/modules/BLAS/Dot product/CUDA/2D/DP2D_1way_callers.cu" -o CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/flags.make
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o: /media/wayne/Disk/DECX_world/srcs/modules/BLAS/Dot\ product/CUDA/2D/DP2D_1way_config.cu
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o -MF CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o.d -x cu -rdc=true -c "/media/wayne/Disk/DECX_world/srcs/modules/BLAS/Dot product/CUDA/2D/DP2D_1way_config.cu" -o CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/flags.make
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o: /media/wayne/Disk/DECX_world/srcs/modules/BLAS/Dot\ product/CUDA/2D/DP2D_kernels.cu
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o -MF CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o.d -x cu -rdc=true -c "/media/wayne/Disk/DECX_world/srcs/modules/BLAS/Dot product/CUDA/2D/DP2D_kernels.cu" -o CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target DP2D_cuda
DP2D_cuda_OBJECTS = \
"CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o" \
"CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o" \
"CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o"

# External object files for target DP2D_cuda
DP2D_cuda_EXTERNAL_OBJECTS =

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/build.make
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/deviceLinkLibs.rsp
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/deviceObjects1
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA device code CMakeFiles/DP2D_cuda.dir/cmake_device_link.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DP2D_cuda.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/build: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o
.PHONY : GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/build

# Object files for target DP2D_cuda
DP2D_cuda_OBJECTS = \
"CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o" \
"CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o" \
"CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o"

# External object files for target DP2D_cuda
DP2D_cuda_EXTERNAL_OBJECTS =

GNU/x86_64/libDP2D_cuda.a: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_callers.cu.o
GNU/x86_64/libDP2D_cuda.a: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_1way_config.cu.o
GNU/x86_64/libDP2D_cuda.a: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DP2D_kernels.cu.o
GNU/x86_64/libDP2D_cuda.a: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/build.make
GNU/x86_64/libDP2D_cuda.a: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/cmake_device_link.o
GNU/x86_64/libDP2D_cuda.a: GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CUDA static library ../libDP2D_cuda.a"
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D && $(CMAKE_COMMAND) -P CMakeFiles/DP2D_cuda.dir/cmake_clean_target.cmake
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DP2D_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/build: GNU/x86_64/libDP2D_cuda.a
.PHONY : GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/build

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/clean:
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D && $(CMAKE_COMMAND) -P CMakeFiles/DP2D_cuda.dir/cmake_clean.cmake
.PHONY : GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/clean

GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/depend:
	cd /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wayne/Disk/DECX_world/srcs/modules/BLAS "/media/wayne/Disk/DECX_world/srcs/modules/BLAS/Dot product/CUDA/2D" /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64 /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D /media/wayne/Disk/DECX_world/build/DECX_BLAS_CUDA/x86_64/GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : GNU/x86_64/DP2D/CMakeFiles/DP2D_cuda.dir/depend

