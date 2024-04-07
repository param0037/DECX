#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject* InitCPUInfo(PyObject* self, PyObject* Py_UNUSED(args)) 
{
    printf("Hello World");
    Py_RETURN_NONE;
}


static PyMethodDef method_table[] = {
    {"InitCPUInfo", (PyCFunction)InitCPUInfo, METH_NOARGS, PyDoc_STR("Initialize CPU information for DECX")},
        {NULL, NULL, 0, NULL}
};


static PyModuleDef core = {
    PyModuleDef_HEAD_INIT,
    "core",
    "Core of DECX runing on host",
    -1,
    method_table,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC PyInit_core()
{
    PyObject* mod = PyModule_Create(&core);
    if (!mod) {
        printf("Internal_Error\n");
        return NULL;
    }
    PyModule_AddStringConstant(mod, "__author__", "Wayne Anderson");
    PyModule_AddStringConstant(mod, "__version__", "1.0.0");
    return mod;
}