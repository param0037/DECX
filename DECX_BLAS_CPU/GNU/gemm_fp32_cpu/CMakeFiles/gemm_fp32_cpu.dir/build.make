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
CMAKE_SOURCE_DIR = /media/wayne/Disk/DECX_world/DECX_BLAS_CPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/build

# Include any dependencies generated for this target.
include /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/compiler_depend.make

# Include the progress variables for this target.
include /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/flags.make

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o: /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/flags.make
/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/GEMM_fp32_caller.cxx
/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o: /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o"
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o -MF CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o.d -o CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/GEMM_fp32_caller.cxx

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.i"
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/GEMM_fp32_caller.cxx > CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.i

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.s"
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/GEMM_fp32_caller.cxx -o CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.s

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o: /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/flags.make
/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o: /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/matrix_B_arrange_fp32.cxx
/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o: /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o"
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o -MF CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o.d -o CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o -c /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/matrix_B_arrange_fp32.cxx

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.i"
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/matrix_B_arrange_fp32.cxx > CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.i

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.s"
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32/matrix_B_arrange_fp32.cxx -o CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.s

gemm_fp32_cpu: /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/GEMM_fp32_caller.cxx.o
gemm_fp32_cpu: /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/matrix_B_arrange_fp32.cxx.o
gemm_fp32_cpu: /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/build.make
.PHONY : gemm_fp32_cpu

# Rule to build all files generated by this target.
/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/build: gemm_fp32_cpu
.PHONY : /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/build

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/clean:
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu && $(CMAKE_COMMAND) -P CMakeFiles/gemm_fp32_cpu.dir/cmake_clean.cmake
.PHONY : /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/clean

/media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/depend:
	cd /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wayne/Disk/DECX_world/DECX_BLAS_CPU /media/wayne/Disk/DECX_world/srcs/modules/BLAS/GEMM/CPU/fp32 /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/build /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : /media/wayne/Disk/DECX_world/DECX_BLAS_CPU/GNU/gemm_fp32_cpu/CMakeFiles/gemm_fp32_cpu.dir/depend

