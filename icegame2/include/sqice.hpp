/* SQIceGame: Square Ice Model
    Backend of the environment and provide python interface for gym.
 */
#pragma once

// Switch of the debug mode.
#define DEBUG

/* TODOs
    NEW DESIGN!
    * Use checkerboard as new lattice class.
    * Add loop algorithm as function in this class.

    * Need more flexibility!
    * Go back to old convention: only Neighbor

    Questions:
    * Do we need to maintain 'mode'? (game/algo)
*/

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

//// Constants used in Icegame ////
const int NUM_OF_ACTIONS = 6;
const int METROPOLIS_PROPOSAL = 6;
const int NULL_SITE = -1;
const int SPIN_UP = 1;
const int SPIN_DOWN = -1;
const int NULL_SPIN = 0;

// LEGACY CODES
const double SPIN_UP_SUBLATT_A = +0.75;
const double SPIN_UP_SUBLATT_B = +0.25;
const double SPIN_DOWN_SUBLATT_A = -SPIN_UP_SUBLATT_A;
const double SPIN_DOWN_SUBLATT_B = -SPIN_UP_SUBLATT_B;

const double DEFECT_MAP_DEFAULT_VALUE = 0.0;
const double ENERGY_MAP_DEFAULT_VALUE = 0.0;
const double EMPTY_MAP_VALUE = 0.0;
const double OCCUPIED_MAP_VALUE = 1.0;
const double AVERAGE_GORUND_STATE_ENERGY = -1.0;
const double AVERAGE_GORUND_STATE_DEFECT_DENSITY = 0.0;
const double ACCEPT_VALUE = 1.0;
const double REJECT_VALUE = -1.0;
const double DEFECT_DENSITY_THRESHOLD = 0.2;

const double AGENT_OCCUPIED_VALUE = 1.0;
//const double AGENT_OCCUPIED_SUBLATT_A = +1.0;
//const double AGENT_OCCUPIED_SUBLATT_B = -1.0;
//const double AGENT_FORESEE_VALUE = 0.75;
//const double AGENT_FORESEE_SUBLATT_A = +0.75;
//const double AGENT_FORESEE_SUBLATT_B = -0.75;

const int MCSTEPS_TO_EQUILIBRIUM = 2000;

const std::string empty_str = std::string();


// NOTICE: Have to define this carefully (insightfully)
enum class ActDir {
    Head_0=0, // NN[0]
    Head_1,   // NN[1]
    Head_2,   // NN[2]
    Tail_0,
    Tail_1,
    Tail_2,
    NOOP
};

//// End of Constants ////

using namespace boost::python; 

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
typedef std::vector<int> IntList;

// wrap c++ array as numpy array
// this function may cause some numeric issue? (probably)
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

class SQIceGame {
    public:
        // Init Constructor
        SQIceGame (INFO info);
        // Init all physical parameters
        void InitModel();
        void SetTemperature(double T);

        // Equalibirum
        void MCRun(int mcSteps);

        // Two kinds of action operation
        //    * directional action (up/down/left/right/upper_next/lower_next)
        //    * TODO spatital action (depends on system size)
        //    return:
        //        vector of [accept, dE, dd, dC]

        // Use Move, instead of Draw
        vector<double> Move(int dir_dix);
        vector<double> Draw(int dir_idx);
        vector<double> Flip();

        vector<int> GuideAction();

        // Read the configuration from python and set to ice_config
        void SetIce(const boost::python::object &iter);

        // Metropolis action
        vector<double> Metropolis();

        // Loop algorithms
        vector<int> LongLoopAlgorithm();

        inline void FlipTrajectory() {flip_along_trajectory(agent_site_trajectory);};
        inline void InitAgentSite(int site) {put_agent(site);};

        int Start(int init_site);

        // Two functions below are repeated.
        int Reset(int init_site = 0);
        void ResetConfiguration();

        void ClearBuffer();
        void UpdateConfig();
        inline void ShowInfo() {show_information();};

        inline int GetAgentSite() {return get_agent_site();};
        inline int GetAgentSpin() {return get_agent_spin();};

        // Configurations (integer array)
        // NOTICE: They are not ordered!
        inline vector<int> GetStateT() {return state_t;};
        inline vector<int> GetStateTp1() {return state_tp1;};
        vector<int> GetStateDiff();

        inline object GetSublatt() {return int_wrap(latt.sub);};

        // Mini maps
        /*
          TODO: IMPORTANT!
          Rearrange the state_t (from pure configuration) to ordered map according to coordinates.

          Note: To prevent numerical bugs, object -> vector<int>/ vector<double>
                which is handled by Vec2List.
        */
        vector<int> GetStateTMap();
        vector<int> GetStateTp1Map();
        vector<int> GetStateDiffMap();

        // LEGACY
        object GetCanvasMap(); 
        object GetEnergyMap();
        object GetDefectMap();
        object GetSublattMap();
        object GetLocalMap(); // WARNNING: empty function now.


        // * Local --> Local Energy
        // * Neighbor --> Adjacency

        void PrintLattice();

        // These two are now wrapper functions
        vector<int> GetNeighborSpins();
        vector<int> GetNeighborSites();

        vector<double> GetLocalSpins();
        vector<int> GetLocalSites();

        vector<double> GetPhyObservables();

        // New state tricks.
        object GetStateTMapColor();
        object GetStateTp1MapColor();
        object GetStateDifferenceMap();
        object GetValidActionMap();
        object GetAgentMap();

        // Statistical Informations
        inline unsigned long GetTotalSteps() {return num_total_steps;};
        inline unsigned long GetEpisode() {return num_episode;};
        inline int GetEpStepCounter() {return ep_step_counter;};
        // Number of successfully update the configuration
        inline unsigned long GetUpdatedCounter() {return updated_counter;};
        // Number of calling Metropolis
        inline unsigned long GetUpdatingCounter() {return num_updates;};
        inline double GetTotalAcceptanceRate() 
                {return updated_counter/(double) num_updates;};

        inline vector<int> GetAcceptedLen() {return accepted_looplength;};
        inline vector<int> GetTrajectory() {return agent_site_trajectory;};
        inline vector<int> GetActionStatistics() {return action_statistics;};
        inline vector<int> GetEpActionCounters() {return ep_action_counters;};
        inline vector<int> GetEpActionList() {return ep_action_list;};

        bool TimeOut();

        void update_ice_config();
        /// this function used fliiping ice config according to states?

        // legacy
        void flip_agent_state_tp1();
        void flip_along_trajectory(const vector<int> &traj);

        // SO THE FOLLOWING SHOULD BE MODIFIED.
        // return new agent site
        // legacy codes
        int go (int dir);
        // NOTICE: this function is replaced by ask_guide in python.
        ActDir how_to_go(int site);

        // Agent Operations
        int put_agent(int site); //Purely put it on.
        int flip_agent();
        int put_and_flip_agent(int site); // put + flip

        // propose a move satisfying the ice-rule
        // return a site

        // Transform between state and config
        void update_state_to_config();
        void restore_config_to_state();

        void clear_all();
        void clear_maps();
        void clear_counters();
        void clear_lists();

        //TODO: UPDATE
        int get_site_by_direction(int dir);
        ActDir get_direction_by_sites(int site, int next_site);
        
        // Neighbor is adjacent 8 sites
        vector<int> get_neighbor_sites();
        vector<int> get_neighbor_spins();
        // maybe we need a candidate for counter partner

        // Local contributes to energy
        // TODO: update to new version
        vector<int> get_local_sites();
        vector<int> get_local_spins();
        vector<int> get_local_candidates(bool same_spin);

        void push_site_to_trajectory(int site);

        /* get funcs */
        int  get_agent_site();
        int  get_agent_spin();
        int  get_spin(int site);
        void show_information();

    private:
        // private functions

        // fundamental flipping functions
        int _flip_state_t_site (int site); // use this less and carefully
        int _flip_state_tp1_site(int site); // return spin

        double _cal_energy_of_state(const vector<int> &s);
        double _cal_energy_of_site(const vector<int> &s, int site);
        double _cal_defect_density_of_state(const vector<int> &s);
        int _cal_config_t_difference();
        int _count_config_difference(const vector<int> &c1, const vector<int> &c2);
        // magic function compute periodic boundary condition
        int inline _pdb(int site, int d, int l) {return ((site + d) % l + l) % l;};

        vector<int> _indices_to_sites1d(const vector<int> indices);
        vector<int> _sites1d_to_indices(const vector<int> sites);
        inline int _index_to_site1d(int index) {return latt.site1d[index];};
        inline int _site1d_to_index(int site) {return latt.indices[site];};

        template <class T>
        void _print_vector(const vector<T> &v, const std::string &s=empty_str);
        double _cal_mean(const vector<int> &s);
        double _cal_stdev(const vector<int> &s);
        bool _is_visited(int site);
        bool _is_traj_continuous();
        bool _is_traj_intersect();
        bool _is_start_end_meets(int site);

        // Used for loop algorithm
        // Note: site is different from index!
        // OOOOOOOOO! Change index <-> site !!
        vector<int> _get_neighbor_of_site(int site);
        vector<int> _get_neighbor_of_index(int index);

        vector<int> _end_sites(int site);

        int _loop_extention(int site);
        int _icerule_head_check(int site);
        int _icerule_tail_check(int site);

        // Physical System
        INFO sim_info;
        Checkerboard latt;

        // Square_ice -->
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
        int start_spin; // used for loop
        int start_site;
        int init_agent_site; // is it used?
        vector<int> agent_site_trajectory;
        vector<int> agent_spin_trajectory;

        /* Statistics of Game */

        unsigned long num_total_steps;
        unsigned long num_episode; // number of resets, TODO: Change point of views!

        unsigned long num_restarts;
        unsigned long num_resets; // number of reset configurations

        unsigned long num_updates; // number of calling Metropolis
        unsigned long updated_counter; // number of successfully updated
        int same_ep_counter; // records for playing the same game

        int update_interval; // not used, actually

        unsigned int ep_step_counter; // counts steps each episode
        // statistics information
        vector<int> ep_site_counters;
        vector<int> ep_action_counters;
        vector<int> ep_action_list;
        vector<int> action_statistics;   // not reset
        vector<int> accepted_looplength; // not reset

        /* Current states */ 
        vector<int> state_0;
        vector<int> state_t;
        vector<int> state_tp1;

        // Maps always use double for ml computation
        // TODO: change these maps back to int
        //      We convert them into np.float32 in python. (or make them colorful.)
        vector<double> agent_map;
        vector<double> canvas_traj_map;
        vector<double> canvas_spin_map;
        vector<double> energy_map;
        vector<double> defect_map;
        //vector<int> diff_map;

        // utilities
        Timer tt;

};

// icegame --> icegame2 ?, well it's okay
BOOST_PYTHON_MODULE(icegame)
{
    class_<INFO>("INFO", init<int, int, int, int, int, int, int, int>())
    ;

    import_array();

    to_python_converter<std::vector<int, class std::allocator<int> >, Vec2List<int> >();
    to_python_converter<std::vector<double, class std::allocator<double> >, Vec2List<double> >();

    //class_<IntList>("IntList")
    //    .def(vector_indexing_suite<IntList>());

    class_<SQIceGame>("SQIceGame", init<INFO>())
        // Monte Carlo related
        .def("init_model", &SQIceGame::InitModel)
        .def("set_temperature", &SQIceGame::SetTemperature)
        .def("mc_run", &SQIceGame::MCRun)
        .def("long_loop_algorithm", &SQIceGame::LongLoopAlgorithm)

        // System commands
        .def("start", &SQIceGame::Start)
        .def("reset", &SQIceGame::Reset)
        .def("reset_config", &SQIceGame::ResetConfiguration)
        .def("timeout", &SQIceGame::TimeOut)
        .def("print_lattice", &SQIceGame::PrintLattice)
        .def("clear_buffer", &SQIceGame::ClearBuffer)

        // REVISE, change the state
        .def("draw", &SQIceGame::Draw)
        .def("move", &SQIceGame::Move)
        .def("flip", &SQIceGame::Flip)
        .def("guide_action", &SQIceGame::GuideAction)

        .def("metropolis", &SQIceGame::Metropolis)
        .def("init_agent_site", &SQIceGame::InitAgentSite)
        .def("flip_trajectory", &SQIceGame::FlipTrajectory)
        .def("update_config", &SQIceGame::UpdateConfig)
        .def("set_ice", &SQIceGame::SetIce)

        // State Raw data
        .def("get_state_t", &SQIceGame::GetStateT)
        .def("get_state_tp1", &SQIceGame::GetStateTp1)
        .def("get_state_diff", &SQIceGame::GetStateDiff)
        .def("get_sublatt", &SQIceGame::GetSublatt)

        // Observations

        // Updated
        .def("get_agent_site", &SQIceGame::GetAgentSite)
        .def("get_agent_spin", &SQIceGame::GetAgentSpin)
        .def("get_agent_map", &SQIceGame::GetAgentMap)
        .def("get_canvas_map", &SQIceGame::GetCanvasMap)
        .def("get_state_t_map", &SQIceGame::GetStateTMap)
        .def("get_state_tp1_map", &SQIceGame::GetStateTp1Map)
        .def("get_state_diff_map", &SQIceGame::GetStateDiffMap)

        .def("get_neighbor_spins", &SQIceGame::GetNeighborSpins)
        .def("get_neighbor_sites", &SQIceGame::GetNeighborSites)
        .def("get_phy_observables", &SQIceGame::GetPhyObservables)

        // Waiting list
        .def("get_energy_map", &SQIceGame::GetEnergyMap)
        .def("get_defect_map", &SQIceGame::GetDefectMap)
        .def("get_valid_action_map", &SQIceGame::GetValidActionMap)
        .def("get_sublatt_map", &SQIceGame::GetSublattMap)

        // LEGACY or REMOVE
        .def("get_local_spins", &SQIceGame::GetLocalSpins)
        .def("get_local_sites", &SQIceGame::GetLocalSites)


        // Game information
        .def("show_info", &SQIceGame::ShowInfo)
        .def("get_total_steps", &SQIceGame::GetTotalSteps)
        .def("get_episode", &SQIceGame::GetEpisode)
        .def("get_ep_step_counter", &SQIceGame::GetEpStepCounter)
        .def("get_total_acceptance_rate", &SQIceGame::GetTotalAcceptanceRate)
        .def("get_action_statistics", &SQIceGame::GetActionStatistics)
        .def("get_ep_action_counters", &SQIceGame::GetEpActionCounters)
        .def("get_ep_action_list", &SQIceGame::GetEpActionList)
        .def("get_updated_counter", &SQIceGame::GetUpdatedCounter)
        .def("get_updating_counter", &SQIceGame::GetUpdatingCounter)
        .def("get_accepted_length", &SQIceGame::GetAcceptedLen)
        .def("get_trajectory", &SQIceGame::GetTrajectory)
    ;
}