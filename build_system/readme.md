# Build System
This is the document for the new building system of DECX. When users compile DECX, there are always tens of combinations regarding to some major indices: Compiler, exported language support, targeted OS, targeted architecture, etc. Thus, a new building system is established to adapt with various of choices.

## 1. On x64 Linux
Make sure your are located in the root of Linux building system: /${Project\_Path}/building_system/Linux/. Before starting, make ./build_configs.sh and ./build.sh executable by:
```bash 
sudo chmod u+x ./build_configs.sh
sudo chmod u+x ./build.sh
```
Then source the file ./build_configs.sh. The standard way of passing build configuration parameters is
```bash
./build_config.sh [$1=host_arch] [$2=exported_lang] [$3=CXX_version] [$4=cmake_toolchain_file]
```
These parameters have their own default values, so for simplicity, the script can be sourced without any parameter. The default values of the parameters are:
```bash
export host_arch=x64                    # Targeted architecture is x86-64
export exported_lang=c,cxx              # Exported C++ and C
export CXX_version=14                   # C++14
export cmake_toolchain_file=${empty}    # Not specified, empty value
```

Except for the parameters mentioned, there are lots of critical parameters for this project, which will be introduced in the following sections.

### a. Key Functions
#### (1) 

## On x64 Windows