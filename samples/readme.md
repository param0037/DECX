# Samples
To build and run the samples, please put the include headers of DECX in /project_dir/includes/. For Windows, the .lib files should be put in /project_dir/bin/x64/. Otherwise, the projects of the samples should all be re-configure, following the steps below:

## 1. Visual Studio
(1) Open /project_dir/samples/DECX_samples.sln
(2) Target to the projects of the sample, configure them according to the directory where DECX libraries are installed:
(3) Right click on project, click "properties" button on the menu, open the project properties window.
(4) In C/C++->General->Additional Include Directories, fill in the include headers of DECX libraries.
(5) In Linker->General->Additional Library Directories, fill in the path where the library files, e.g. DECX_core_CPU.lib are located.
(6) In Linker->Input->Additional Dependencies, fill in the names of the libraries that are depended by the sample application.


## 2. Build with CMake