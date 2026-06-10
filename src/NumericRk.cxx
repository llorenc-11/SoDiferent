#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <utility>
#include <array>
#include <stdexcept>
#include <cstring>
#include <string>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>
using namespace std;



static constexpr array<double, 7> weights{5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0};
static constexpr array<double, 6> nodes{1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0};
static constexpr array<double, 21> table{
1.0,
1.0/4.0, 3.0/4.0,
11.0/9.0, -14.0/3.0, 40.0/9.0,
4843.0/1458.0, -3170.0/243.0, 8056.0/729.0, -53.0/162.0,
9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0,
35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0
};

pair<vector<double>, vector<double>> Rk(const function <double(double,double)> &func,double min_arg, double max_arg, double initial_val, double tolerance=1e-5,double min_step= 1e-4){
    int const dim = 7;
    double curr_slope;
    double curr_arg;
    double order4;
    double order5;
    double error_est;

    min_step = max(min_step,1e-6);
    tolerance = max(tolerance,1e-15);

    if (min_arg > max_arg) {
        swap(min_arg, max_arg);
    }
    curr_arg = min_arg;

    vector<double> s(dim,0);
    s[0] = func(min_arg,initial_val);
    double h = max((0.5 - s[0]/20),min_step); // rough estimate for initial step size 

    pair<vector<double>,vector<double>> result;

    result.first.reserve(ceil((max_arg - min_arg)/((h+min_step)/2))); // rough estimate of result size , division by mean of first step and minimal h value 
    result.first.push_back(initial_val);


    result.second.reserve(ceil((max_arg - min_arg)/(h+min_step)/2));
    result.second.push_back(min_arg);
     
    while(curr_arg < max_arg){

        if (curr_arg + h > max_arg) { // decrease step size if the maximum argument value is going to be execeeded
            h = max(max_arg - curr_arg, min_step);
        }

        for(int i =0 ; i<dim-1;i++){
           curr_slope=0;
            for(int j =0 ; j<=i; j++){
               curr_slope+= table[i*(i+1)/2 + j]*s[j];
            }
            order5 = result.first.back() + h*nodes[i]*curr_slope;
            s[i+1]= func(curr_arg + h*nodes[i],order5);
        }

        curr_slope =0;
        for(int i =0 ; i<dim;i++){
            curr_slope += weights[i]*s[i];
        }
        order4 = result.first.back() + h*curr_slope;

        error_est = pow(tolerance*h/(2*abs(order5-order4)),0.25);

        if( error_est>1 || abs(h -  min_step) < 1e-12 ){
            curr_arg += h;
            result.first.push_back(order5);
            result.second.push_back(curr_arg);
            s[0] = s[6];
            h = max(min_step, h * error_est * 0.9);
        }else{
            h = max(min_step, h * error_est * 0.9);
        } 
    }

    return result;  // modern compilers should use here named return value optimization, although older ones are not certain to do so,
                    // it could be changed so the result is passed as reference to the function that would guarantee no copy on all compilers and systems 
}


pair<vector<double>, vector<double>> RkVectorised(const function <vector<double>(double,const vector<double>&)> &func,double min_arg,double max_arg, const vector<double> &initial_val, double tolerance=1e-5,double min_step= 1e-4){
    int const dim = 7;
    double h; 
    double curr_slope;
    double curr_arg;
    size_t num_equations = initial_val.size();

    vector<double> order4(num_equations,0);
    vector<double> order5(num_equations,0);
    vector<double> s(dim*num_equations,0);

    min_step = max(min_step,1e-6);
    tolerance = max(tolerance,1e-15);

    if (min_arg > max_arg) {
        swap(min_arg, max_arg);
    }
    curr_arg = min_arg;

    vector<double> initial_slopes = func(min_arg,initial_val);
    double max_slope0 = abs(initial_slopes[0]);

    for(size_t i =0 ; i<num_equations;i++){
        s[i] = initial_slopes[i];
        max_slope0 = max(max_slope0,abs(initial_slopes[i]));
    }
    h = max(min_step, 0.5 - max_slope0 / 20.0); // rough estimate for initial step size 


    pair<vector<double>,vector<double>> result;
    size_t estimated_size = ceil((max_arg - min_arg)/((h+min_step)/2));

    result.first.reserve(num_equations *estimated_size);
    for(size_t i =0 ; i<num_equations;i++){
        result.first.push_back(initial_val[i]);
    }

    result.second.reserve(estimated_size);
    result.second.push_back(min_arg);


    vector<double> curr_state = initial_val; 
     
    while(curr_arg < max_arg){
        if(curr_arg + h > max_arg){
            h = max(max_arg-curr_arg,min_step);
        }

        for (int i = 0; i < dim - 1; i++) {
            for (size_t n = 0; n < num_equations; n++) {
                curr_slope = 0.0;
                for (int j = 0; j <= i; j++) {
                    curr_slope += table[i*(i+1)/2 + j] * s[j * num_equations + n];
                }
                order5[n] = curr_state[n] + h * nodes[i] * curr_slope; 
            }
            
            vector<double> next_slope = func(curr_arg + h * nodes[i], order5); 
            
            for (size_t n = 0; n < num_equations; n++) {
                s[(i + 1) * num_equations + n] = next_slope[n];
            }
        }

        for (size_t n = 0; n < num_equations; n++) {
            curr_slope = 0.0;
            for (int i = 0; i < dim; i++) {
                curr_slope += weights[i] * s[i * num_equations + n];
            }
            order4[n] = curr_state[n] + h * curr_slope;
        }


        double error_est = 1e-15; 
        for(size_t n = 0; n < num_equations; n++) {
            error_est = max(error_est, abs(order5[n] - order4[n]));
        }
        double a = pow(tolerance * h / (2.0 * error_est), 0.25);


        if (a > 1.0 || h <= min_step) {
            curr_arg += h;
            swap(curr_state,order5); // order5 is overwriten in each loop before it is used
            
            result.second.push_back(curr_arg);
            for (size_t n = 0; n < num_equations; n++) {
                result.first.push_back(curr_state[n]);
            }
            
            for (size_t n = 0; n < num_equations; n++) {
                s[n] = s[6 * num_equations + n];
            }

            h = max(min_step, h * a * 0.9);
        } else {
            h = max(min_step, h * a * 0.9);
        } 
    }

    return result;
}






static PyObject* py_rk(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* py_func;
    PyObject* py_y0; 
    PyObject* extra_parameters = NULL;
    double t_min, t_max, tol, min_step;
    
    static char* kwlist[] = {
        (char*)"function", 
        (char*)"t_min", 
        (char*)"t_max", 
        (char*)"initial_y", 
        (char*)"tolerance",
        (char*)"min_step",
        (char*)"extra_parameters",
        NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OddOddO", kwlist, 
        &py_func, &t_min, &t_max, &py_y0, &tol, &min_step, &extra_parameters)){
        return NULL; 
    }

    if (extra_parameters == Py_None){
        extra_parameters = NULL;
    }


    if (PyFloat_Check(py_y0)){
        
        double y0 = PyFloat_AsDouble(py_y0);
        function<double(double, double)> cpp_wrapper_func;

        if (PyLong_Check(py_func)) {
            void* ptr_address = PyLong_AsVoidPtr(py_func);
            typedef double (*ptrCFunc)(double, double,const double*);
            ptrCFunc compiled_c_func = reinterpret_cast<ptrCFunc>(ptr_address);

            vector<double> cpp_params;
            if (extra_parameters){
                Py_ssize_t plen = PyTuple_Size(extra_parameters);
                for (Py_ssize_t i = 0; i < plen; ++i){
                    cpp_params.push_back(PyFloat_AsDouble(PyTuple_GetItem(extra_parameters, i)));
                }
            }

            cpp_wrapper_func = [compiled_c_func, cpp_params](double x, double y) -> double {
                const double* temp;
                if(cpp_params.empty()){
                    temp = nullptr;
                }else{
                    temp = cpp_params.data();
                }
                return compiled_c_func(x, y, temp);
            };

        }else{
            cpp_wrapper_func = [py_func, extra_parameters](double x, double y) -> double {
                Py_ssize_t extra_len;
                if(extra_parameters){
                    extra_len = 1;
                }else{
                    extra_len =0;
                }
                PyObject* paramlist = PyTuple_New(2 + extra_len);

                PyTuple_SetItem(paramlist, 0, PyFloat_FromDouble(x));
                PyTuple_SetItem(paramlist, 1, PyFloat_FromDouble(y));
                if(extra_parameters){
                    Py_INCREF(extra_parameters);
                    PyTuple_SetItem(paramlist,2,extra_parameters);
                }
                
                PyObject* py_result = PyObject_CallObject(py_func, paramlist);
                Py_DECREF(paramlist); 

                if(!py_result) throw runtime_error("Python function failed.");

                double c_result = PyFloat_AsDouble(py_result);
                Py_DECREF(py_result); 
                return c_result;
            };
        }

        pair<vector<double>, vector<double>> result;
        try{
            result = Rk(cpp_wrapper_func, t_min, t_max, y0, tol, min_step);
        }catch(const runtime_error& e) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }

        Py_ssize_t size_result = result.first.size();
        Py_ssize_t dims[1] = { size_result };

        PyObject* py_y_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        PyObject* py_x_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

        if (!py_y_array || !py_x_array) return NULL; 

        memcpy(PyArray_DATA((PyArrayObject*)py_y_array), result.first.data(), size_result * sizeof(double));
        memcpy(PyArray_DATA((PyArrayObject*)py_x_array), result.second.data(), size_result * sizeof(double));

        return Py_BuildValue("(NN)", py_y_array, py_x_array);

    }else{

        PyArrayObject* vector_y0 = (PyArrayObject*)PyArray_FROM_OTF(py_y0, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!vector_y0) return NULL;
        
        Py_ssize_t num_equations = PyArray_SIZE(vector_y0);
        double* y0_data = (double*)PyArray_DATA(vector_y0);
        vector<double> y0_vec(y0_data, y0_data + num_equations);
        Py_DECREF(vector_y0);

        function<vector<double>(double, const vector<double>&)> cpp_vec_func_wrapper;

        if (PyLong_Check(py_func)) {
            void* ptr_address = PyLong_AsVoidPtr(py_func);
            typedef void (*ptrCVectorFunc)(double, const double*, double*, const double*);
            ptrCVectorFunc compiled_c_func = reinterpret_cast<ptrCVectorFunc>(ptr_address);

            vector<double> cpp_params;
            if (extra_parameters) {
                Py_ssize_t plen = PyTuple_Size(extra_parameters);
                for (Py_ssize_t i = 0; i < plen; ++i) {
                    cpp_params.push_back(PyFloat_AsDouble(PyTuple_GetItem(extra_parameters, i)));
                }
            }

            cpp_vec_func_wrapper = [compiled_c_func, num_equations, cpp_params](double x, const vector<double>& y) -> vector<double> {
                vector<double> out_slope(num_equations); 
                const double* temp;
                if(cpp_params.empty()){
                    temp = nullptr;
                }else{
                    temp = cpp_params.data();
                }
                compiled_c_func(x, y.data(), out_slope.data(), temp);
                return out_slope;
            };
        } else {

            cpp_vec_func_wrapper = [py_func, extra_parameters, num_equations](double x, const vector<double>& y) -> vector<double> {
                
                Py_ssize_t extra_len;
                if(extra_parameters){
                    extra_len = 1;
                }else{
                    extra_len =0;
                }
                PyObject* paramlist = PyTuple_New(2 + extra_len);

                PyTuple_SetItem(paramlist, 0, PyFloat_FromDouble(x));
                
                Py_ssize_t dims[1] = { num_equations };
                PyObject* temp_y_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)y.data());
                PyTuple_SetItem(paramlist, 1, temp_y_array); 

                if(extra_parameters){
                    Py_INCREF(extra_parameters);
                    PyTuple_SetItem(paramlist,2,extra_parameters);
                }
                
                PyObject* py_result = PyObject_CallObject(py_func, paramlist);
                Py_DECREF(paramlist); 

                if(!py_result) throw runtime_error("Python function failed.");

                PyArrayObject* res_arr = (PyArrayObject*)PyArray_FROM_OTF(py_result, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
                if (!res_arr) {
                    Py_DECREF(py_result);
                    throw runtime_error("Function must return a numpy array or list of floats.");
                }

                if (PyArray_SIZE(res_arr) != num_equations) {
                    Py_DECREF(res_arr);
                    Py_DECREF(py_result);
                    throw runtime_error("Python function returned an object of wrong size.Expected size: " + to_string(num_equations) + ".");
                }
                
                double* res_data = (double*)PyArray_DATA(res_arr);
                vector<double> c_result(res_data, res_data + num_equations);
                
                Py_DECREF(res_arr);
                Py_DECREF(py_result);
                return c_result;
            };
        }

        pair<vector<double>, vector<double>> result;
        try {
            result = RkVectorised(cpp_vec_func_wrapper, t_min, t_max, y0_vec, tol, min_step);
        } catch(const runtime_error& e) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }

        Py_ssize_t num_steps = result.second.size();
        Py_ssize_t y_dims[2] = { num_steps, num_equations };
        Py_ssize_t x_dims[1] = { num_steps };

        PyObject* py_y_array = PyArray_SimpleNew(2, y_dims, NPY_DOUBLE);
        PyObject* py_x_array = PyArray_SimpleNew(1, x_dims, NPY_DOUBLE);

        if (!py_y_array || !py_x_array) return NULL; 

        memcpy(PyArray_DATA((PyArrayObject*)py_y_array), result.first.data(), num_steps * num_equations * sizeof(double));
        memcpy(PyArray_DATA((PyArrayObject*)py_x_array), result.second.data(), num_steps * sizeof(double));

        return Py_BuildValue("(NN)", py_y_array, py_x_array);
    } 
}





static PyMethodDef my_methods[] = {
    {"adaptiveRk", (PyCFunction)py_rk, METH_VARARGS | METH_KEYWORDS, "Solves an initial value problem for a system of ordinary differential equations using the explicit Dormand-Prince from adaptive Runge-Kutta methods family."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef my_module = {
    PyModuleDef_HEAD_INIT,
    "NumericRk",
    "Module implementing numeric Runga-Kutta solvers for ordinary/partial differential equations",
    -1,
    my_methods
};

extern "C" {

PyMODINIT_FUNC PyInit_NumericRk()
{
    PyObject* m = PyModule_Create(&my_module);
    if (!m) return NULL;
    import_array();
    return m;
}
}