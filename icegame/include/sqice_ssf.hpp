/*
 * This program serves as new icegame, simplified versnio of environment for learning loop algorithm.
    TODO: Use site coordinate action, single spin flip action.
 */
#pragma once

// Switch of the debug mode.
// #define DEBUG

// cpp standard libs
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

//// Constants used in Icegame SSF ////
const int LSIZE = 32;
const int NUM_OF_SPINS = LSIZE * LSIZE; // number of spins
const int NUM_OF_ACTIONS = NUM_OF_SPINS + 1; // SSF + Metropolis Proposal
const int METROPOLIS_PROPOSAL = 0;
const int NULL_SITE = -1;
const int NULL_COORD= -1;
const int SPIN_UP   = +1;
const int SPIN_DOWN = -1;
const double DEFECT_MAP_DEFAULT_VALUE = 1.0;
const double ENERGY_MAP_DEFAULT_VALUE = 0.0;
const double EMPTY_MAP_VALUE = 0.0;
const double OCCUPIED_MAP_VALUE = 1.0;
const double AVERAGE_GORUND_STATE_ENERGY = -1.0;
const double AVERAGE_GORUND_STATE_DEFECT_DENSITY = 0.0;
const double ACCEPT_VALUE = 1.0;
const double REJECT_VALUE = -1.0;
//// End of Constants ////


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

// avoid using template
static boost::python::object int_wrap(const std::vector<int> &vec) {
    npy_intp size = vec.size();
    int *data = const_cast<int*>(&vec[0]);
    npy_intp shape[1] = { size }; // array size
    PyObject* obj = PyArray_New(&PyArray_Type, 1, shape, NPY_INT, // data type
                                NULL, data, // data pointer
                                0, NPY_ARRAY_CARRAY, // NPY_ARRAY_CARRAY_RO for readonly
                                NULL);
    handle<> array( obj );
    return object(array);
}

// conversion for std::vector to python list
template <class T>
struct Vec2List {
    static PyObject* convert (const std::vector<T> &vec) {
        boost::python::list *l = new boost::python::list();
        for (size_t i = 0; i < vec.size(); ++i)
            (*l).append(vec[i]);
        return l->ptr();
    }
};

/* Function Naming Convention:
    Upper Case function used for python interface
    lower case used as member functions
*/

enum DIR {RIGHT, DOWN, LEFT, UP, LOWER_NEXT, UPPER_NEXT};

class SQIceGameSSF {
    public:
        // Init Constructor
        SQIceGameSSF (INFO info);
        // Init all physical parameters
        void InitModel();
        void SetTemperature(double T);

        // Equalibirum
        void MCRun(int mcSteps);

        // Two kinds of action operation
        //    * directional action (up/down/left/right/upper_next/lower_next)
        //    * TODO coordinate action (depends on system size)
        //    return:
        //        vector of [accept, dE, dd, dC]
        //

        vector<double> Draw(int dir_idx);
        /*
            Note: coord is labeled from 1 to N.
                    site is labeled from 0 to N-1.
                    action is labeled from 0 to N. (N+1 in total)
        */
        vector<double> SSF(int coord);

        // Metropolis operation
        vector<double> Metropolis();

        inline void FlipTrajectory() {flip_along_traj(agent_site_trajectory);};

        int Start(int init_site);
        int Restart(int init_site);

        void ClearBuffer();
        void UpdateConfig();

        inline int GetAgentSite() {return get_agent_site();};
        object GetStateTMap();
        object GetCanvasMap(); 
        object GetEnergyMap();
        object GetDefectMap();

        // Statistical Informations
        int GetStartPoint();
        inline unsigned long GetTotalSteps() {return num_total_steps;};
        inline unsigned long GetEpisode() {return num_episode;};
        inline int GetEpStepCounter() {return ep_step_counter;};
        // Number of successfully update the configuration
        inline unsigned long GetUpdatedCounter() {return updated_counter;};
        // Number of calling Metropolis
        inline unsigned long GetUpdatingCounter() {return num_updates;};
        inline vector<int> GetAcceptedLen() {return accepted_looplength;};
        inline vector<int> GetTrajectory() {return agent_site_trajectory;};
        inline vector<int> GetActionStatistics() {return action_statistics;};
        inline vector<int> GetEpActionCounters() {return ep_action_counters;};
        inline vector<int> GetEpActionList() {return ep_action_list;};

        bool TimeOut();

        void update_ice_config();
        /// this function used fliiping ice config according to states?

        void flip_site(int site);
        void flip_agent_site();
        void flip_along_traj(const vector<int> &traj);

        // return new agent site
        // these functions will change state_tp1 which is used for calculating dE.
        int go(int dir);
        int goflip(int site);
        int how_to_go(int site);

        int coord2site(int coord);
        int site2coord(int site);

        // propose a move satisfying the ice-rule
        // return a site
        int icemove(bool);

        void update_state_to_config();
        void restore_config_to_state();

        void clear_all();
        void clear_maps();
        void clear_counters();
        void clear_lists();

        int get_neighbor_site_by_direction(int dir);
        int get_direction_by_sites(int site, int next_site);
        
        vector<int> get_neighbor_sites();
        vector<int> get_neighbor_spins();
        vector<int> get_neighbor_candidates(bool same_spin);

        void set_agent_site(int site);

        /* get funcs */
        int  get_agent_site();
        int  get_agent_spin();
        int  get_spin(int site);

    private:
        // private functions
        double _cal_energy_of_state(const vector<int> &s);
        double _cal_energy_of_site(const vector<int> &s, int site);
        double _cal_defect_density_of_state(const vector<int> &s);
        int _cal_config_t_difference();
        int _count_config_difference(const vector<int> &c1, const vector<int> &c2);
        // magic function compute periodic boundary condition
        int inline _pdb(int site, int d, int l) {return ((site + d) % l + l) % l;};

        void long_loop_algorithm();
        void short_loop_algorithm();

        void _print_vector(const vector<int> &v);
        double _cal_mean(const vector<int> &s);
        double _cal_stdev(const vector<int> &s);
        bool _is_visited(int site);
        bool _is_traj_continuous();
        bool _is_traj_intersect();
        bool _is_start_end_meets(int site);

        // Physical System
        INFO sim_info;
        Square latt;
        Square_ice model;
        Sample ice_config;

        double config_mean; 
        double config_stdev;

        unsigned int L, N;
        double kT;
        double h1_t, h2_t, h3_t;
        double J1;
        vector<double> mag_fields;

        // RL intefaces
        int agent_site;
        vector<int> agent_site_trajectory;
        vector<int> agent_spin_trajectory;

        /* Statistics of Game */

        unsigned long num_total_steps;
        unsigned long num_episode; // number of resets
        unsigned long num_updates; // number of calling Metropolis
        unsigned long updated_counter; // number of successfully updated
        int same_ep_counter; // records for playing the same game

        int update_interval; // 

        int init_agent_site;
        unsigned int ep_step_counter; // each episode
        vector<int> ep_site_counters;
        vector<int> ep_action_counters;
        vector<int> ep_action_list;
        vector<int> action_statistics;   // not reset
        vector<int> accepted_looplength; // not reset

        // MAPS
        /* Current states */ 
        vector<int> state_0;
        vector<int> state_t;
        vector<int> state_tp1;

        // maps always use double
        vector<double> agent_map;
        vector<double> canvas_traj_map;
        vector<double> canvas_spin_map;
        vector<double> energy_map;
        vector<double> defect_map;
        vector<int> diff_map;

        // utilities
        Timer tt;

};

BOOST_PYTHON_MODULE(icegame_ssf)
{
    // Register Twice, but dont wanna import INFO from icegame, that is wired.
    class_<INFO>("INFO", init<int, int, int, int, int, int, int, int>())
    ;
    // Maybe can solve from: https://github.com/esa/pagmo/issues/1

    import_array();

    to_python_converter<std::vector<int, class std::allocator<int> >, Vec2List<int> >();
    to_python_converter<std::vector<double, class std::allocator<double> >, Vec2List<double> >();

    class_<SQIceGameSSF>("SQIceGameSSF", init<INFO>())
        .def("init_model", &SQIceGameSSF::InitModel)
        .def("set_temperature", &SQIceGameSSF::SetTemperature)
        .def("mc_run", &SQIceGameSSF::MCRun)

        .def("start", &SQIceGameSSF::Start)
        .def("restart", &SQIceGameSSF::Restart)
        .def("timeout", &SQIceGameSSF::TimeOut)

        .def("clear_buffer", &SQIceGameSSF::ClearBuffer)

        // REVISE
        .def("draw", &SQIceGameSSF::Draw)
        .def("ssf", &SQIceGameSSF::SSF)
        .def("metropolis", &SQIceGameSSF::Metropolis)
        .def("flip_trajectory", &SQIceGameSSF::FlipTrajectory)
        .def("update_config", &SQIceGameSSF::UpdateConfig)

        .def("get_agent_site", &SQIceGameSSF::GetAgentSite)
        .def("get_canvas_map", &SQIceGameSSF::GetCanvasMap)
        .def("get_state_t_map", &SQIceGameSSF::GetStateTMap)
        .def("get_energy_map", &SQIceGameSSF::GetEnergyMap)
        .def("get_defect_map", &SQIceGameSSF::GetDefectMap)

        .def("get_start_point", &SQIceGameSSF::GetStartPoint)
        .def("get_total_steps", &SQIceGameSSF::GetTotalSteps)
        .def("get_episode", &SQIceGameSSF::GetEpisode)
        .def("get_ep_step_counter", &SQIceGameSSF::GetEpStepCounter)
        .def("get_action_statistics", &SQIceGameSSF::GetActionStatistics)
        .def("get_ep_action_counters", &SQIceGameSSF::GetEpActionCounters)
        .def("get_ep_action_list", &SQIceGameSSF::GetEpActionList)
        .def("get_updated_counter", &SQIceGameSSF::GetUpdatedCounter)
        .def("get_updating_counter", &SQIceGameSSF::GetUpdatingCounter)
        .def("get_accepted_length", &SQIceGameSSF::GetAcceptedLen)
        .def("get_trajectory", &SQIceGameSSF::GetTrajectory)
    ;
}
