# Samples
Samples are provided here for users who develop applications based on DECX. The samples provide the usage of majority of APIs. The entire samples are in C++ language currently.

## 1. Visual Studio
To build and run the samples using Microsoft Visual Studio, please put the include headers of DECX in /project_dir/includes/. For Windows, the .lib files should be put in /project_dir/bin/x64/. Otherwise, the projects of the samples should all be re-configure, following the steps below:

(1) Open /project_dir/samples/DECX_samples.sln
(2) Target to the projects of the sample, configure them according to the directory where DECX libraries are installed:
(3) Right click on project, click "properties" button on the menu, open the project properties window.
(4) In C/C++->General->Additional Include Directories, fill in the include headers of DECX libraries.
(5) In Linker->General->Additional Library Directories, fill in the path where the library files, e.g. DECX_core_CPU.lib are located.
(6) In Linker->Input->Additional Dependencies, fill in the names of the libraries that are depended by the sample application.


## 2. Build with CMake (For x64 targets)
(1) Target to the project directory to be built.
(a) On Windows, run cmd.exe or powershell.exe under the project path or cd into the project path, enter "build" to run the script. 
(b) On Linux, run terminal under the project path or cd into the project path. Enter "./build.sh" to run the script.
(2) The built applications are installed in /project_dir/samples/x64.