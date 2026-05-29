#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <utility>
#include <array>
#include <stdexcept>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>
using namespace std;



static constexpr std::array<double, 7> weights{5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0};
static constexpr std::array<double, 6> nodes{1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0};
static constexpr std::array<double, 21> table{
1.0,
1.0/4.0, 3.0/4.0,
11.0/9.0, -14.0/3.0, 40.0/9.0,
4843.0/1458.0, -3170.0/243.0, 8056.0/729.0, -53.0/162.0,
9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0,
35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0
};

pair<vector<double>, vector<double>> Rk(const function <double(double,double)> &func,double min_arg_val, double max_arg_val, double initial_val, double tolerance){
    int const dim = 7;
    double const min_h_val = 0.001;

    double curr_slope;
    double curr_arg = min_arg_val;
    double order4;
    double order5;
    double a;

    vector<double> s(dim,0);
    s[0] = func(min_arg_val,initial_val);
    double h = max((0.5 - s[0]/20),min_h_val);

    vector<double> y;
    y.reserve(ceil((max_arg_val - min_arg_val)/h));
    y.push_back(initial_val);

    vector<double> args;
    args.reserve(ceil((max_arg_val - min_arg_val)/h));
    args.push_back(min_arg_val);
     
    while(curr_arg < max_arg_val){
        if (curr_arg + h > max_arg_val) {
            if(max_arg_val - curr_arg > min_h_val){
                h = max_arg_val - curr_arg;
            }else{
                h = min_h_val;
            }

        }

        for(int i =0 ; i<dim-1;i++){
           curr_slope=0;
            for(int j =0 ; j<=i; j++){
               curr_slope+= table[i*(i+1)/2 + j]*s[j];
            }
            order5 = y.back() + h*nodes[i]*curr_slope;
            s[i+1]= func(curr_arg + h*nodes[i],order5);
        }

        curr_slope =0;
        for(int i =0 ; i<dim;i++){
            curr_slope += weights[i]*s[i];
        }
        order4 = y.back() + h*curr_slope;

        a = pow(tolerance*h/(2*abs(order5-order4)),0.25);

        if( a>1 || h == min_h_val){
            y.push_back(order5);
            args.push_back(curr_arg+h);
            s[0] = s[6];
            curr_arg += h;
            h = max(min_h_val, h * a * 0.9);
        }else{
            h = max(min_h_val, h * a * 0.9);
        } 
    }

    return pair(y,args);
}



static PyObject* py_rk(PyObject* self, PyObject* args,PyObject* kwargs) {
    PyObject* py_func;
    double x_min, x_max, y0, tol;
    

    static char* kwlist[] = {(char*)"function", 
        (char*)"x_min", 
        (char*)"x_max", 
        (char*)"initial_y", 
        (char*)"tolerance", 
        NULL};

    if (!PyArg_ParseTupleAndKeywords(args,kwargs, "Odddd",kwlist, &py_func, &x_min, &x_max, &y0, &tol)) {
        return NULL; 
    }

    if (!PyCallable_Check(py_func)) {
        PyErr_SetString(PyExc_TypeError, "Function argument must be a callable python object.");
        return NULL;
    }

    function<double(double, double)> cpp_wrapper_func = [py_func](double t, double y) -> double {
        //sprawdzenie czy ta funkcja bierze dwa argumenty 
        PyObject* arglist = Py_BuildValue("(dd)", t, y);
        
        PyObject* py_result = PyObject_CallObject(py_func, arglist);

        Py_DECREF(arglist); 

        if (!py_result) {
            throw runtime_error("Python function failed");
        }

        double c_result = PyFloat_AsDouble(py_result);
        
        Py_DECREF(py_result); 
        
        return c_result;
    };


    pair<std::vector<double>, std::vector<double>> result;
    try{
        result = Rk(cpp_wrapper_func, x_min, x_max, y0, tol);
    }catch(const runtime_error& e){
        return NULL;
    }

    Py_ssize_t num_points = result.first.size();
    PyObject* py_y_list = PyList_New(num_points);
    PyObject* py_t_list = PyList_New(num_points);

    for (Py_ssize_t i = 0; i < num_points; ++i) {
        PyList_SET_ITEM(py_y_list, i, PyFloat_FromDouble(result.first[i]));
        PyList_SET_ITEM(py_t_list, i, PyFloat_FromDouble(result.second[i]));
    }

    return Py_BuildValue("(OO)", py_y_list, py_t_list);
}





static PyMethodDef my_methods[] = {
    {"adaptiveRk", (PyCFunction)py_rk, METH_VARARGS | METH_KEYWORDS, "Runge-Kutta adaptive solver using the Dormand-Prince algorythm"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef my_module = {
    PyModuleDef_HEAD_INIT,
    "Numeric",
    "moj modul (docstring)",
    -1,
    my_methods
};

extern "C" {

PyMODINIT_FUNC PyInit_Numeric()
{
    PyObject* m = PyModule_Create(&my_module);
    if (!m) return NULL;
    import_array();
    return m;
}
}