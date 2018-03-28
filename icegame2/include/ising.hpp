#pragma once
//#define DEBUG
// cpp standard 
#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <string>

// monte carlo libraries
#include "sample.hpp"
#include "hamiltonian.hpp"
#include "lattice.hpp"
#include "timer.hpp"

// boost.python intefaces
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <numpy/ndarrayobject.h> // ensure you include this header

using namespace boost::python; 

// wrap c++ array as numpy array
static boost::python::object float_wrap(const std::vector<double> &vec) {
    npy_intp size = vec.size();
    double *data = const_cast<double *>(&vec[0]);
    npy_intp shape[1] = { size }; // array size
    PyObject* obj = PyArray_New(&PyArray_Type, 1, shape, NPY_DOUBLE, // data type
                                NULL, data, // data pointer
                                0, NPY_ARRAY_CARRAY, // NPY_ARRAY_CARRAY_RO for readonly
                                NULL);
    handle<> array( obj );
    return object(array);
}

// Converter for std::vector to python list
template <class T>
struct Vec2List {
    static PyObject* convert (const std::vector<T> &vec) {
        boost::python::list *l = new boost::python::list();
        for (size_t i = 0; i < vec.size(); ++i)
            (*l).append(vec[i]);
        return l->ptr();
    }
};

class IsingGame {
    public:
        IsingGame(INFO info);

        void InitModel();
        void SetTemperature(double T);
        void MCRun(int mcSteps);

        // Operations
        void Reset();
        vector<int> GetState();

        // Passing python list to function in cpp
        // https://stackoverflow.com/questions/4819707/passing-python-list-to-c-vector-using-boost-python

    private:

        INFO sim_info;
        Square latt;
        Ising  model;
        Sample ising_config;

        unsigned int L, N;
        double kT;
        double h1_t, h2_t, h3_t;
        double J1;
        vector<double> mag_fields;

        // MAPS
        /* Current states */ 
        vector<int> state_0;
        vector<int> state_t;
        vector<int> state_tp1;
        // utilities
        Timer tt;
};

BOOST_PYTHON_MODULE(isinggame)
{
    class_<INFO>("INFO", init<int, int, int, int, int, int, int, int>())
    ;

    import_array();

    to_python_converter<std::vector<int, class std::allocator<int> >, Vec2List<int> >();
    to_python_converter<std::vector<double, class std::allocator<double> >, Vec2List<double> >();

    class_<IsingGame>("IsingGame", init<INFO>())
        .def("init_model", &IsingGame::InitModel)
        .def("set_temperature", &IsingGame::SetTemperature)
        .def("mc_run", &IsingGame::MCRun)
    ;
}