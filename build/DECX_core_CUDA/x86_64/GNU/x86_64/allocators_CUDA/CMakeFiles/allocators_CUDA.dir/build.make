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
CMAKE_BINARY_DIR = /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64

# Include any dependencies generated for this target.
include GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/compiler_depend.make

# Include the progress variables for this target.
include GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/progress.make

# Include the compile flags for this target's objects.
include GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/flags.make

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/flags.make
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o: /media/wayne/Disk/DECX_world/srcs/modules/core/allocators/_allocator.cu
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o -MF CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o.d -x cu -rdc=true -c /media/wayne/Disk/DECX_world/srcs/modules/core/allocators/_allocator.cu -o CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/flags.make
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o: /media/wayne/Disk/DECX_world/srcs/modules/core/allocators/_deallocator.cu
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o -MF CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o.d -x cu -rdc=true -c /media/wayne/Disk/DECX_world/srcs/modules/core/allocators/_deallocator.cu -o CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/flags.make
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o: /media/wayne/Disk/DECX_world/srcs/modules/core/allocators/_memset.cu
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o -MF CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o.d -x cu -rdc=true -c /media/wayne/Disk/DECX_world/srcs/modules/core/allocators/_memset.cu -o CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/allocators_CUDA.dir/__/_memset.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/allocators_CUDA.dir/__/_memset.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target allocators_CUDA
allocators_CUDA_OBJECTS = \
"CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o" \
"CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o" \
"CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o"

# External object files for target allocators_CUDA
allocators_CUDA_EXTERNAL_OBJECTS = \
"/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemBlock.cu.o" \
"/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemChunk_D.cu.o" \
"/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemoryPool_D.cu.o"

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemBlock.cu.o
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemChunk_D.cu.o
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemoryPool_D.cu.o
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/build.make
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/deviceLinkLibs.rsp
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/deviceObjects1
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA device code CMakeFiles/allocators_CUDA.dir/cmake_device_link.o"
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/allocators_CUDA.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/build: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o
.PHONY : GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/build

# Object files for target allocators_CUDA
allocators_CUDA_OBJECTS = \
"CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o" \
"CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o" \
"CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o"

# External object files for target allocators_CUDA
allocators_CUDA_EXTERNAL_OBJECTS = \
"/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemBlock.cu.o" \
"/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemChunk_D.cu.o" \
"/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemoryPool_D.cu.o"

GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_allocator.cu.o
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_deallocator.cu.o
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/__/_memset.cu.o
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemBlock.cu.o
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemChunk_D.cu.o
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/mempool_CUDA/CMakeFiles/mempool_CUDA.dir/__/MemoryPool_D.cu.o
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/build.make
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/cmake_device_link.o
GNU/x86_64/liballocators_CUDA.a: GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CUDA static library ../liballocators_CUDA.a"
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA && $(CMAKE_COMMAND) -P CMakeFiles/allocators_CUDA.dir/cmake_clean_target.cmake
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/allocators_CUDA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/build: GNU/x86_64/liballocators_CUDA.a
.PHONY : GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/build

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/clean:
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA && $(CMAKE_COMMAND) -P CMakeFiles/allocators_CUDA.dir/cmake_clean.cmake
.PHONY : GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/clean

GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/depend:
	cd /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wayne/Disk/DECX_world/srcs/modules/core /media/wayne/Disk/DECX_world/srcs/modules/core/allocators/CUDA /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64 /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA /media/wayne/Disk/DECX_world/build/DECX_core_CUDA/x86_64/GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : GNU/x86_64/allocators_CUDA/CMakeFiles/allocators_CUDA.dir/depend

