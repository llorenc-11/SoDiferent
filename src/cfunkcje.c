#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

// pip3 install .

double kwadrat(double x) {
    return x*x;
}

double suma(const double* x, Py_ssize_t n)
{
    double s = 0.0;
    for (Py_ssize_t i=0; i<n; ++i) s += x[i];
    return s;
}


PyObject* pysuma(PyObject* self, PyObject* args)
{
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs == -1) return NULL;
    else if (nargs != 1) {
        return PyErr_Format(PyExc_RuntimeError, "expected 1 argument");
    }

    PyObject* args0 = PyTuple_GetItem(args, 0);
    if (!args0) return NULL;

    if (!PyArray_Check(args0)) return PyErr_Format(PyExc_RuntimeError, "expected a numpy array");

    if (PyArray_TYPE((PyArrayObject*)args0) != NPY_DOUBLE ||
        !PyArray_IS_C_CONTIGUOUS((const PyArrayObject*)args0))
        return PyErr_Format(PyExc_RuntimeError, "expected a numpy array[double]");

    const double* x = PyArray_DATA((PyArrayObject*)args0);
    Py_ssize_t n = PyArray_SIZE((PyArrayObject*)args0);


    double s = suma(x, n);
    return PyFloat_FromDouble(s);  // Py_INCREF
}

PyObject* pykwadrat(PyObject* self, PyObject* args)
{
    // args jest tuplem
    double x;

    // if (!PyArg_ParseTuple(args, "d", &x)) {
    //     return NULL;
    // }

    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs == -1) return NULL;
    else if (nargs != 1) {
        return PyErr_Format(PyExc_RuntimeError, "expected 1 argument");
    }

    PyObject* args0 = PyTuple_GetItem(args, 0);
    if (!args0) return NULL;

    x = PyFloat_AsDouble(args0);
    if (x == -1.0 && PyErr_Occurred())
        return NULL;

    double s = kwadrat(x);

    return PyFloat_FromDouble(s);  // Py_INCREF
}

static PyMethodDef my_methods[] = {
    {"pykwadrat", pykwadrat, METH_VARARGS, "kwadrat liczy (docstring)"},
    {"pysuma", pysuma, METH_VARARGS, "sume liczy"},
    {NULL, NULL, 0, NULL} // wartownik
};

static struct PyModuleDef my_module = {
    PyModuleDef_HEAD_INIT,
    "cfunkcje",
    "moj modul (docstring)",
    -1,
    my_methods
};

PyMODINIT_FUNC/* PyObject* */ PyInit_cfunkcje()
{
    PyObject* m = PyModule_Create(&my_module);
    if (!m) return NULL;
    import_array();
    return m;
}

