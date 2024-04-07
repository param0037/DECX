/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
