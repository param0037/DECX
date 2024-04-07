///**
//*   ---------------------------------------------------------------------
//*   Author : Wayne Anderson
//*   Date   : 2021.04.16
//*   ---------------------------------------------------------------------
//*   This is a part of the open source program named "DECX", copyright c Wayne,
//*   2021.04.16, all right reserved.
//*   More information please visit https://github.com/param0037/DECX
//*/
//
//#include "../modules/core/configs/config.h"
//#include <Python.h>
//
//#define PY_SSIZE_T_CLEAN
//
//
//static PyObject* InitCUDA(PyObject* self, PyObject* Py_UNUSED(args))
//{
//    De_InitCUDA();
//    Py_RETURN_NONE;
//}
//
//
//static PyMethodDef method_table_cuda[] = {
//    {"InitCPUInfo", (PyCFunction)InitCUDA, METH_NOARGS, PyDoc_STR("Initialize CUDA information for DECX")},
//        {NULL, NULL, 0, NULL}
//};
//
//
//static PyModuleDef core_CUDA = {
//    PyModuleDef_HEAD_INIT,
//    "cuda",
//    "Core of DECX runing on device",
//    -1,
//    method_table_cuda,
//    NULL,
//    NULL,
//    NULL,
//    NULL
//};
//
///*
//* PyInit_modulename
//* The module name must be consistent with the real module name,
//* since Python interpreter will try to find this function by name
//*/
//PyMODINIT_FUNC PyInit_cuda()
//{
//    PyObject* mod = PyModule_Create(&core_CUDA);
//    if (!mod) {
//        Print_Error_Message(4, "Internal_Error\n");
//        return NULL;
//    }
//    PyModule_AddStringConstant(mod, "__author__", "Wayne Anderson");
//    PyModule_AddStringConstant(mod, "__version__", "1.0.0");
//    return mod;
//}