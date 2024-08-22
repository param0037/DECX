# Build System
This is the document for the new building system of DECX. When users compile DECX, there are always tens of combinations regarding to some major indices: Compiler, exported language support, targeted OS, targeted architecture, etc. Thus, a new building system is established to adapt with various of choices.

## Structure
<font color="cyan">./build_system<br></font>
├── <font color="cyan">Linux</font><br>
│&emsp;├── build_configs.sh<br>
│&emsp;├── build.sh<br>
│&emsp;├── <font color="cyan">params</font><br>
│&emsp;│&emsp;├── <font color="green">cuda_sm.sh</font><br>
│&emsp;│&emsp;├── <font color="green">cxx_ver.sh</font><br>
│&emsp;│&emsp;├── <font color="green">exp_lang.sh</font><br>
│&emsp;│&emsp;├── <font color="green">host_arch.sh</font><br>
│&emsp;│&emsp;├── <font color="green">param_lists.sh</font><br>
│&emsp;│&emsp;├── <font color="green">set_module.sh</font><br>
│&emsp;│&emsp;└── <font color="green">toolchain.sh</font><br>
│&emsp;└── utils.sh<br>
├── readme.md<br>
└── Windows<br>
&emsp;├── CMake<br>
&emsp;└── VS<br>
&emsp;&emsp;├── DECX_BLAS_CPU.vcxproj<br>
&emsp;&emsp;├── DECX_BLAS_CPU.vcxproj.filters<br>
&emsp;&emsp;├── DECX_BLAS_CPU.vcxproj.user<br>
&emsp;&emsp;├── DECX_BLAS_CUDA.vcxproj<br>
&emsp;&emsp;├── DECX_BLAS_CUDA.vcxproj.filters<br>
&emsp;&emsp;├── DECX_BLAS_CUDA.vcxproj.user<br>
&emsp;&emsp;├── DECX_core_CPU.vcxproj<br>
&emsp;&emsp;├── DECX_core_CPU.vcxproj.filters<br>
&emsp;&emsp;├── DECX_core_CPU.vcxproj.user<br>
&emsp;&emsp;├── DECX_core_CUDA.vcxproj<br>
&emsp;&emsp;├── DECX_core_CUDA.vcxproj.filters<br>
&emsp;&emsp;├── DECX_core_CUDA.vcxproj.user<br>
&emsp;&emsp;└── DECX_world.sln<br>


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
#### (1) Functions to configure
(a) Set the targeted host architecture. The architectures supported by DECX are x86-64 and aarch64. Enter one of the architectures as the parameter, the parameter is <font color="yellow">not case sensitive</font>.
```bash
host_arch [$1=<x86-64><x86_64><x64><aarch64><arm64>]
```

(b) Set the exported language(s)



## On x64 Windows