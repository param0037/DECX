/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#include "../configs/config.h"


static PyObject* InitCPUInfo(PyObject* self, PyObject* Py_UNUSED(args))
{
    De_InitCPUInfo();
    Py_RETURN_NONE;
}


static PyObject* InitCUDA(PyObject* self, PyObject* Py_UNUSED(args))
{
    De_InitCUDA();
    Py_RETURN_NONE;
}


static PyMethodDef method_table[] = {
    {"InitCPUInfo", (PyCFunction)InitCPUInfo, METH_NOARGS, PyDoc_STR("Initialize CPU information for DECX")},
    {"InitCUDA", (PyCFunction)InitCUDA, METH_NOARGS, PyDoc_STR("Initialize CUDA information for DECX")},
    {NULL, NULL, 0, NULL}
};


static PyModuleDef _core = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "Core of DECX",
    -1,
    method_table,
    NULL,
    NULL,
    NULL,
    NULL
};


/*
* PyInit_modulename
* The module name must be consistent with the real module name,
* since Python interpreter will try to find this function by name
*/
PyMODINIT_FUNC PyInit__core()
{
    PyObject* mod = PyModule_Create(&_core);
    if (!mod) {
        printf("Internal_Error\n");
        return NULL;
    }
    PyModule_AddStringConstant(mod, "__author__", "Wayne Anderson");
    PyModule_AddStringConstant(mod, "__version__", "1.0.0");
    return mod;
}
