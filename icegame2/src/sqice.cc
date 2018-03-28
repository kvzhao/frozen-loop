/* Icegame2
*/
#include "sqice.hpp"
#include <math.h>

SQIceGame::SQIceGame (INFO info) : sim_info(info) {
    model.init(sim_info);
    latt.init(sim_info);
    ice_config.init (sim_info);

    // pre-determined parameters
    N = sim_info.Num_sites;
    L = sim_info.lattice_size;

    std::cout << "[GAME] N: " << N << ", L: " << L << "\n"; 

    h1_t = 0.0;
    h2_t = 0.0;
    h3_t = 0.0;
    J1 = 1.0;

    // Agent information
    agent_site = NULL_SITE;
    init_agent_site = NULL_SITE;

    // Set initial episode to 1, avoid division by zero
    num_episode = 1; 
    num_total_steps = 0;
    same_ep_counter = 0;
    updated_counter = 0;
    num_updates = 0;

    mag_fields.push_back(h1_t);
    mag_fields.push_back(h2_t);
    mag_fields.push_back(h3_t);

    // initialze all the member data
    state_0.resize(N, 0);
    state_t.resize(N, 0);
    state_tp1.resize(N, 0);

    // counters initialization
    ep_step_counter = 0;
    ep_site_counters.resize(N, 0);
    ep_action_counters.resize(NUM_OF_ACTIONS, 0);
    action_statistics.resize(NUM_OF_ACTIONS, 0); // not reset

    // maps initilization
    agent_map.resize(N, EMPTY_MAP_VALUE);
    canvas_traj_map.resize(N, EMPTY_MAP_VALUE);
    canvas_spin_map.resize(N, EMPTY_MAP_VALUE);
    energy_map.resize(N, ENERGY_MAP_DEFAULT_VALUE);
    defect_map.resize(N, DEFECT_MAP_DEFAULT_VALUE);
    diff_map.resize(N, EMPTY_MAP_VALUE);

    std::cout << "[GAME] SQIceGame is created.\n";
}

bool SQIceGame::TimeOut() {
    /* TimeOut: The function is used for terminate the episode
        1. steps Timeout
        (TODO) 2. Defect Density "Timeout"
    */
    bool is_timeout = false;
    if (ep_step_counter >= N) {
        is_timeout = true;
    }
    return is_timeout;
}

void SQIceGame::clear_maps() {
    std::fill(canvas_traj_map.begin(), canvas_traj_map.end(), EMPTY_MAP_VALUE);
    std::fill(agent_map.begin(), agent_map.end(), EMPTY_MAP_VALUE);
    std::fill(canvas_spin_map.begin(), canvas_spin_map.end(), EMPTY_MAP_VALUE);
    std::fill(energy_map.begin(), energy_map.end(), ENERGY_MAP_DEFAULT_VALUE);
    std::fill(defect_map.begin(), defect_map.end(), DEFECT_MAP_DEFAULT_VALUE);
    std::fill(diff_map.begin(), diff_map.end(), EMPTY_MAP_VALUE);
}

void SQIceGame::clear_counters() {
    std::fill(ep_site_counters.begin(), ep_site_counters.end(), 0);
    std::fill(ep_action_counters.begin(), ep_action_counters.end(), 0);
    ep_step_counter = 0; 
}

void SQIceGame::clear_lists() {
    agent_site_trajectory.clear();
    agent_spin_trajectory.clear();
    ep_action_list.clear();
}

void SQIceGame::update_state_to_config() {
    int diff = _cal_config_t_difference();
    vector<int> backup = ice_config.Ising;
    ice_config.Ising = state_t;
    state_0 = state_t;
    state_tp1 = state_t;
    // state_0 = state_t 
    // ... Sanity check!
    if ( _cal_defect_density_of_state(ice_config.Ising) == AVERAGE_GORUND_STATE_DEFECT_DENSITY \
        && _cal_energy_of_state(ice_config.Ising) == AVERAGE_GORUND_STATE_ENERGY) {
        std::cout << "[GAME] Updated Succesfully!\n";
        updated_counter++; 
        accepted_looplength.push_back(diff);
        // Avoid periodic timeout mechanism rule out preferable results
    } else {
        std::cout << "[GAME] Ice Config is RUINED. Restore.\n";
        ice_config.Ising = backup;
        restore_config_to_state();
    }
    // reset maps
}

void SQIceGame::UpdateConfig() {
    update_state_to_config();
}

void SQIceGame::restore_config_to_state() {
    // any check?
    state_t = ice_config.Ising;
    state_tp1 = ice_config.Ising;
    state_0 = ice_config.Ising;
}

void SQIceGame::InitModel() {
    model.set_J1(J1);
	model.initialization(&ice_config, &latt, 1);
	model.initialize_observable(&ice_config, &latt, kT, mag_fields);
    std::cout << "[GAME] All physical parameters are initialized\n";
}

void SQIceGame::SetTemperature(double T) {
    if (T >= 0.0) kT = T;
    std::cout << "[GAME] Set temperature kT = " << kT << "\n";
}

void SQIceGame::ResetConfiguration() {
    std::fill(ice_config.Ising.begin(), ice_config.Ising.end(), 1);
    MCRun(2000);
}

void SQIceGame::SetIce(const boost::python::object &iter) {
    // (DANGER) Set the ice configuration from given python configuration.
    std::vector< double > dcfg = std::vector< double > (boost::python::stl_input_iterator< double > (iter),
                                                        boost::python::stl_input_iterator< double > ());
    // ! Conversion is needed here! (double -> int)
    std::vector<int> cfg(dcfg.begin(), dcfg.end());
    #ifdef DEBUG
      std::cout << "SQIceGame::SetIce {\\";
      for(auto i : cfg) {
          std::cout << " " << i << " ";
      }
      std::cout << "\n";
    #endif

    // Should do the sanity check and assign to spins and config, then update
    // refer to the updating function
    double defect_density = _cal_defect_density_of_state(cfg);
    double energy_density = _cal_energy_of_state(cfg);
    if (  defect_density == AVERAGE_GORUND_STATE_DEFECT_DENSITY \
        && energy_density == AVERAGE_GORUND_STATE_ENERGY) {
        // We should carefully assign the configuration to all of the state, and clear the buffer.
        // almost restart!
        clear_all();
        // Set the cfg to state_t
        state_t = std::move(cfg); // now, cfg is empty
        ice_config.Ising = state_t;
        state_0 = state_t;
        state_tp1 = state_t;
        // but in this way, update counter++, which is wrong.
        // update_state_to_config();
        // Here, no sanity check is needed.
        // ...
        // Or, we just lazily do this and call from python?
        std::cout << "[GAME] SetIce from python legally.\n";
    } else {
        std::cout << "[GAME] The Configuration is not in ice state! Store in buffer.\n";
        std::cout << " Energy: " << energy_density << " Defect: " << defect_density << ".\n";
        //ice_config.Ising = backup;
        //restore_config_to_state();
    }
} 

void SQIceGame::MCRun(int mcSteps) {
    /*
        Prepare an ice state for game
    */
    tt.timer_begin();
    for (int i = 0; i < mcSteps; ++i) {
        model.MCstep(&ice_config, &latt);
    }
    tt.timer_end();

    std::cout << "[GAME] Monte Carlo runs " 
                << mcSteps << " steps with "
                << tt.timer_duration() << " seconds.\n"; 
    
    // Check whether it is icestates
    double Etot = model.total_energy(&ice_config, &latt);
    std::cout << "[GAME] Total Energy E = " << Etot << "\n";


    // Get the ising variables
    state_0 = ice_config.Ising;
    state_t = ice_config.Ising;
    state_tp1 = ice_config.Ising;

    std::cout << "[GAME] Average Energy E = " << _cal_energy_of_state(state_0) << "\n";
    //std::cout << "[GAME] Defect Density D = " << _cal_defect_density_of_state(state_0) << "\n";
    //std::cout << "[GAME] Config mean = " << config_mean << " , and std = " << config_stdev << "\n";
    // ======== FAILS BELOW ====== //
    /*
    config_mean = _cal_mean(state_0);
    config_stdev = _cal_stdev(state_0);

    std::cout << "[GAME] Average Energy E = " << _cal_energy_of_state(state_0) << "\n";
    std::cout << "[GAME] Defect Density D = " << _cal_defect_density_of_state(state_0) << "\n";
    std::cout << "[GAME] Config mean = " << config_mean << " , and std = " << config_stdev << "\n";
    */

}

// Start: Just put the agent on the site
int SQIceGame::Start(int init_site) {
    int ret = put_agent(init_site);
    if (ret != agent_site) { 
        std::cout << "[GAME] WARNING: Start() get wrong init_site!\n";
    }
    return agent_site;
}

void SQIceGame::ClearBuffer() {
    clear_all();
    same_ep_counter++;
}

// Reset: Generate a new initial state
int SQIceGame::Reset(int init_site) {
    ClearBuffer();

    //TODO: Create the new starting state
    MCRun(2000);

    // push_agent_site(init_site);
    // Not sure what it is doing

    same_ep_counter = 0;
    num_episode++;
    return agent_site;
}

// Restart: Back to the same initail state
int SQIceGame::Restart(int init_site) {
    ClearBuffer();
    Start(init_site);
    same_ep_counter = 0;
    num_episode++;
    return agent_site;
}

void SQIceGame::clear_all() {
    clear_maps();
    clear_lists();
    clear_counters();
    restore_config_to_state();
    init_agent_site = agent_site;
}

int SQIceGame::GetStartPoint() {
    if (agent_site_trajectory.size() != 0) {
        if (init_agent_site != agent_site_trajectory[0]) {
            std::cout << "[Game] Sanity check fails! init_agent_site != trajectory[0]!\n";
        }
    }
    return init_agent_site;
}

// member functions 
vector<double> SQIceGame::Metropolis() {
    /* Metropolis operation called by agent.
        Returns: 
            results = []
    */
    vector<double> rets(4);
    bool is_accept = false;
    double E0 = _cal_energy_of_state(state_0);
    double Et = _cal_energy_of_state(state_t);
    double dE = Et - E0;
    double dd = _cal_defect_density_of_state(state_t);
    int diff_counts = _cal_config_t_difference();
    double diff_ratio = diff_counts / double(N);

    // calculates returns
    if (dE == 0.0) {
        if (dd != 0.0) {
            std::cout << "[Game]: State has no energy changes but contains defects! Sanity checking fails!\n";
        }
        is_accept = true;
        rets[0] = ACCEPT_VALUE;
    } else {
        rets[0] = REJECT_VALUE;
    }
    rets[1] = dE;
    rets[2] = dd;
    rets[3] = diff_ratio;

    // update counters
    action_statistics[METROPOLIS_PROPOSAL]++;
    ep_action_counters[METROPOLIS_PROPOSAL]++;
    num_total_steps++;
    // Not an Episode
    ep_step_counter++;
    num_updates++;

    return rets;
}

int SQIceGame::_flip_state_tp1_site(int site) {
    // maybe we need cal some info when flipping
    int spin = 0;
    if(site >= 0 && site < N) {
        state_tp1[site] *= -1;
        spin = state_tp1[site];
    } else {
        std::cout << "[GAME] WARNING: You try to flip #"
                  << site 
                  << " spin out of range.\n";
    }
    return spin;
}

int SQIceGame::_flip_state_t_site(int site) {
    // DANGER!, we shoud carefully use this function.
    int spin = 0;
    if(site >= 0 && site < N) {
        state_tp1[site] *= -1;
        spin = state_tp1[site];
    } else {
        std::cout << "[GAME] WARNING: You try to flip #"
                  << site 
                  << " spin out of range.\n";
    }
    return spin;
}

// LEGACY
void SQIceGame::flip_agent_state_tp1() {
    // helper function flip site on current state tp1
    _flip_state_tp1_site(agent_site);
    // also need to draw on canvas
    // draw on canvas TODO: no repeats!
    if (canvas_traj_map[agent_site] == EMPTY_MAP_VALUE) {
        // TODO: distinguish AB sublattice
        if (latt.sub[agent_site] == 1) {
            canvas_traj_map[agent_site] = OCCUPIED_MAP_VALUE;
        } else {
            canvas_traj_map[agent_site] = - OCCUPIED_MAP_VALUE;
        }
    }
    canvas_spin_map[agent_site] = double(state_t[agent_site]);
}

// We should clearify the function name.
void SQIceGame::flip_along_traj(const vector<int>& traj) {
    // check traj is not empty
    // check traj is cont. or not
    // flip along traj on state_t
    // done, return ?
    if(!traj.empty()) {
        for (auto const & site : traj) {
            _flip_state_t_site(site);
            #ifdef DEBUG
            std::cout << "Flip site " << site 
                    << " and dE = " << _cal_energy_of_state(state_t) - _cal_energy_of_state(state_0)
                    << ", dd = " << _cal_defect_density_of_state(state_t) << endl;
            #endif
        }
    }
}

void SQIceGame::push_site_to_trajectory(int site) {
    // Just used for taking care of trajectory.
    if (site >= 0 && site < N) {
        // If starting point
        if (agent_site_trajectory.size() == 0) {
            agent_site_trajectory.push_back(site);
            agent_spin_trajectory.push_back(get_spin(site));
            init_agent_site = site;
        } else if (!_is_visited(site)) {
            agent_site_trajectory.push_back(site);
            agent_spin_trajectory.push_back(get_spin(site));
        } else {
            // visited site and do nothing
        }
    } else {
        std::cout << "[GAME] WORNING, Set Agent on Illegal Site!\n";
    }
}
// how about write the pop mechanism?

int SQIceGame::get_agent_site() {
    return agent_site;
}

int SQIceGame::get_spin(int site) {
    // NOTE: We get the state_tp1 spin!
    int spin = 0;
    if(site >= 0 && site < N) {
        spin = state_tp1[site];
    }
    return spin;
}

int SQIceGame::get_agent_spin() {
    return get_spin(agent_site);
}

// GRAY ZONE.


vector<double> SQIceGame::Draw(int dir_idx) {
    // The function handles canvas and calculates step-wise returns
    // TODO Extend action to 8 dir
    int curt_spin = get_agent_spin();
    vector<double> rets(4);
    // get where to go
    // NOTICE: get_spin returns state_tp1 spin
    int next_spin = get_spin(get_local_site_by_direction(dir_idx));
    // move agent
    int site = go(dir_idx);

    // draw on canvas TODO: no repeats!
    if (canvas_traj_map[site] == EMPTY_MAP_VALUE) {
        // TODO: distinguish AB sublattice
        if (latt.sub[site] == 1) {
            canvas_traj_map[site] = OCCUPIED_MAP_VALUE;
        } else {
            canvas_traj_map[site] = - OCCUPIED_MAP_VALUE;
        }
    }
    canvas_spin_map[site] = double(state_t[site]);

    double dE = _cal_energy_of_state(state_tp1) - _cal_energy_of_state(state_t);
    double dd = _cal_defect_density_of_state(state_tp1);

    // TODO: compare t and tp1
    double dC = _count_config_difference(state_t, state_tp1) / double(N);

    if (curt_spin == next_spin) {
        rets[0] = REJECT_VALUE;
    } else {
        rets[0] = ACCEPT_VALUE;
    }
    rets[1] = dE; 
    rets[2] = dd;
    rets[3] = dC;

    #ifdef DEBUG 
    std::cout << "  current spin " << curt_spin << " , next spin " << next_spin << "\n";
    std::cout << " Draw dE = " << dE
                << ", dd = " << dd << "\n";
    #endif

    return rets;
}

vector<double> SQIceGame::Flip() {
    /* !!! This function is DEPRECIATED. !!!*/
    vector<double> rets(4);
    // Flip the current agent site.
    int site = agent_site;
    agent_map[site] = OCCUPIED_MAP_VALUE; 

    // Flip no matter how.
    state_tp1[site] *= -1;

    num_total_steps++;
    ep_step_counter++;
    ep_site_counters[site]++;

    // draw on canvas TODO: no repeats
    if (canvas_traj_map[site] == EMPTY_MAP_VALUE) {
        // TODO: distinguish AB sublattice
        if (latt.sub[site] == 1) {
            canvas_traj_map[site] = OCCUPIED_MAP_VALUE;
        } else {
            canvas_traj_map[site] = - OCCUPIED_MAP_VALUE;
        }
    }

    canvas_spin_map[site] = double(state_t[site]);

    double dE = _cal_energy_of_state(state_tp1) - _cal_energy_of_state(state_t);
    double dd = _cal_defect_density_of_state(state_tp1);

    // TODO: compare t and tp1
    double dC = _count_config_difference(state_t, state_tp1) / double(N);

    rets[0] = REJECT_VALUE;
    rets[1] = dE; 
    rets[2] = dd;
    rets[3] = dC;

    return rets;
}

int SQIceGame::go(int dir) {
    // One of the core function.
    // this function handles moveing, so the ep_action should be counted.
    int new_site = get_local_site_by_direction(dir);

    int agent_site = put_and_flip_agent(new_site);

    // action statistics only be counted when called by icegame_env.
    ep_action_counters[dir]++;
    ep_action_list.push_back(dir);
    action_statistics[dir]++;

    return agent_site;
}

int SQIceGame::put_agent(int new_site) {
    // Purely put it on site, we can find agent on map but no other records.
    int old_site = agent_site;
    // agent map TODO: use trajectory 
    if (old_site >=0 && old_site < N) {
        agent_map[old_site] = EMPTY_MAP_VALUE;
    } else {
        std::cout << "[GAME] minor warning: put_agent with illegal starting site\n";
    }
    // new site should be checked, it's important to make sure agent_site is corrent.
    if (new_site >=0 && new_site < N) {
        // chane the agent site to new_site
        agent_site = new_site;
        agent_map[new_site] = OCCUPIED_MAP_VALUE; 
    } else {
        std::cout << "[GAME] WARNING! You assign agent to go insane: "
                  << new_site <<  "\n";
    }
    return agent_site;
}

int SQIceGame::flip_agent() {
    // This function can clip the state_tp1.
    _flip_state_tp1_site(agent_site);

    // Flipping is the formal action, so we need to count in ep. counters.
    num_total_steps++;
    ep_step_counter++;
    ep_site_counters[agent_site]++;

    // save it into records (traj)
    push_site_to_trajectory(agent_site);
    return agent_site;
}

int SQIceGame::put_and_flip_agent(int site) {
    // 1. Put the agent on the site
    int ret_site = put_agent(site);
    // check the operation is okay.
    if (ret_site == site) {
        flip_agent();
    } else {
        std::cout << "[GAME] WARNING! Try the put and flip agent out of range!\n";
    }
    return ret_site;
}


int SQIceGame::how_to_go(int site) {
    int dir = get_direction_by_sites(agent_site, site);
    return dir;
}

vector<double> SQIceGame::GetNeighborSpins() {
    // TODO:
    vector<double> neighbor_spins(9);
}

vector<double> SQIceGame::GetLocalSpins() {
    // Return [agent_spin, neighbor spins (with sublattice convention)], 7 elements
    vector<double> local_spins(7);
    vector<int> nspins = get_local_spins();
    vector<int> nsites = get_local_sites();
    if (latt.sub[agent_site] == 1) {
        local_spins[0] = get_agent_spin() > 0 ? SPIN_UP_SUBLATT_A : SPIN_DOWN_SUBLATT_A;
    } else{
        local_spins[0] = get_agent_spin() > 0 ? SPIN_UP_SUBLATT_B : SPIN_DOWN_SUBLATT_B;
    }
    // number of neighbor is fixed to be 6, hard-coded.
    for (int i=0; i<6; i++) {
        if (latt.sub[nsites[i]] == 1) {
            local_spins[i+1] = nspins[i] > 0 ? SPIN_UP_SUBLATT_A : SPIN_DOWN_SUBLATT_A;
        } else {
            local_spins[i+1] = nspins[i] > 0 ? SPIN_UP_SUBLATT_B : SPIN_DOWN_SUBLATT_B;
        }
    }
    return local_spins;
}

vector<int> SQIceGame::GetLocalSites() {
    // Wrapper of the function 'get_neighbor_sites'
    return get_local_sites();
}

vector<int> SQIceGame::GetNeighborSites() {
    return get_neighbor_sites();
}

vector<double> SQIceGame::GetPhyObservables() {
    // Get defect density and energy density of the system
    /*
        How many phyiscal qunatities are relevant?
            * Energy Density
            * Defect Density
            * Configuration Defference Ratio
        Is loop length important here?
    */

    vector<double> obs(2);
    double eng_density = _cal_energy_of_state(state_tp1); // this is already the density
    //double def_density = _cal_defect_density_of_state(state_tp1);
    double config_diff_ratio = _count_config_difference(state_t, state_tp1) / double(N);
    obs[0] = eng_density;
    obs[1] = config_diff_ratio;
    //obs[1] = def_density;
    return obs;
}

object SQIceGame::GetLocalMap() {
    // return the patch of the local mini-map
}


vector<int> SQIceGame::get_neighbor_spins() {
    vector<int> nsites = get_neighbor_sites();
    vector<int> nspins;
    for(const auto& s : nsites) {
        nspins.push_back(state_t[s]);
    }
    return nspins;
}

vector<int> SQIceGame::_get_neighbor_of_site(int _site) {
    vector<int> locals(6);
    locals[0] = latt.NN[_site][0];
    locals[1] = latt.NN[_site][1];
    locals[2] = latt.NN[_site][2];
    locals[3] = latt.NN[_site][3];
    locals[4] = latt.NN[_site][4];
    locals[5] = latt.NN[_site][5];
    return locals;
}


vector<int> SQIceGame::get_local_sites() {
    // Fool around first.
    return get_neighbor_sites();
}

vector<int> SQIceGame::get_neighbor_sites() {
    return _get_neighbor_of_site(agent_site);
}

vector<int> SQIceGame::get_local_spins() {
    vector<int> locals(6);
    locals[0] = state_t[latt.NN[agent_site][0]];
    locals[1] = state_t[latt.NN[agent_site][1]];
    locals[2] = state_t[latt.NN[agent_site][2]];
    locals[3] = state_t[latt.NN[agent_site][3]];
    if (latt.sub[agent_site] == 1) {
        locals[4] = state_t[latt.NNN[agent_site][0]];
        locals[5] = state_t[latt.NNN[agent_site][2]];
    } else {
        locals[4] = state_t[latt.NNN[agent_site][1]];
        locals[5] = state_t[latt.NNN[agent_site][3]];
    }
    return locals;
}

vector<int> SQIceGame::get_local_candidates(bool same_spin) {
    // This function returns candidates agent can go
    // TODO: Should be revised!
    int inverse_agent_spin = 0;
    if (same_spin) {
        inverse_agent_spin = get_agent_spin();
    } else {
        inverse_agent_spin = -1 * get_agent_spin();
    }
    vector<int> nspins = get_local_spins();
    vector<int> nsites = get_local_sites();
    vector<int> idxes;
    for (std::vector<int>::size_type i =0; i < nspins.size(); i++) {
        if (inverse_agent_spin == nspins[i]) {
            idxes.push_back(i);
        }
    }
    vector<int> candidates;
    for (auto const & idx : idxes) {
        candidates.push_back(nsites[idx]);
    }
    std::random_shuffle(candidates.begin(), candidates.end());
    // should i avoid repeating?
    return candidates;
}

// TODO::WARNING: OLD CONVENTIONS!
int SQIceGame::get_local_site_by_direction(int dir_idx) {
    int site = agent_site;
    // c++11 method for casting
    auto dir = static_cast<ActDir>(dir_idx);
    // TODO: 6 action -> 8 actions
    // TODO: Checkerboard lattice!
    switch (dir) {
        case ActDir::RIGHT:
            site = latt.NN[site][0];
            break;
        case ActDir::DOWN:
            site = latt.NN[site][1];
            break;
        case ActDir::LEFT:
            site = latt.NN[site][2];
            break;
        case ActDir::UP:
            site = latt.NN[site][3];
            break;
        case ActDir::LOWER_RIGHT:
            site = latt.NNN[site][0];
            break;
        case ActDir::LOWER_LEFT:
            site = latt.NNN[site][1];
            break;
        case ActDir::UPPER_LEFT:
            site = latt.NNN[site][2];
            break;
        case ActDir::UPPER_RIGHT:
            site = latt.NNN[site][3];
            break;
    }

    // others recognized as no-op (just stay the same site)
    //#ifdef DEBUG
    std::cout << "get_neighbor_site_by_direction(dir=" << dir_idx << ") = " 
                << site << " with agent site = " << agent_site << " \n";
    //#endif
    return site;
}

// Maybe we can implement a ask_guide here.
int SQIceGame::get_direction_by_sites(int site, int next_site) {
    int right_site = latt.NN[site][0];
    int left_site = latt.NN[site][2];
    int up_site = latt.NN[site][3];
    int down_site = latt.NN[site][1];
    int upper_next_site = NULL_SITE;
    int lower_next_site = NULL_SITE;
    if (latt.sub[site] == 1) {
        lower_next_site = latt.NNN[site][0];
        upper_next_site = latt.NNN[site][2];
    } else {
        lower_next_site = latt.NNN[site][1];
        upper_next_site = latt.NNN[site][3];
    }

    int dir = -1;
    // check next state is in its neighbots
    // BRUTAL FORCE
    if (next_site == right_site) {
        dir = 0;
    } else if (next_site == left_site) {
        dir = 2;
    } else if (next_site == up_site) {
        dir = 3;
    } else if (next_site == down_site) {
        dir  = 1;
    } else if (next_site == upper_next_site) {
        dir = 5;
    } else if (next_site == lower_next_site) {
        dir = 4;
    } else {
        dir = 7; // index 7 is a no-operation
    }

    #ifdef DEBUG
    std::cout << "get_direction_by_sites(site=" << site << ", next=" << next_site << " ): its neighbors are "
                << "right site = " << right_site << "\n"
                << "left site = " << left_site << "\n"
                << "up site = " << up_site << "\n"
                << "down site = " << down_site << "\n"
                << "upper next site = " << upper_next_site << "\n"
                << "lower next site = " << lower_next_site << "\n";
    #endif

    return dir;
}

/* GetMaps:
    TODO: add more map tricks
*/

object SQIceGame::GetEnergyMap() {
    for (int i = 0; i < N; i++) {
        double se = 0.0;
        // TODO: Assign different values on AB sublattice
        // if (latt.sub[i] == 1
        se = _cal_energy_of_site(state_tp1, i);
        energy_map[i] = se / 6.0;
    }
    return float_wrap(energy_map);
}

object SQIceGame::GetStateTMap() {
    vector<double> map(state_t.begin(), state_t.end());
    return float_wrap(map);
}

object SQIceGame::GetStateTp1Map() {
    vector<double> map(state_tp1.begin(), state_tp1.end());
    return float_wrap(map);
}

object SQIceGame::GetStateTMapColor() {
    //TODO: Review this codes
    // parsing the map and return.
    vector<double> map_(state_t.begin(), state_t.end());
    // Do things here.
    for (int i=0; i < N; i++) {
        double spin = map_[i];
        if (latt.sub[i] == 1) {
            map_[i] = spin > 0.0 ? SPIN_UP_SUBLATT_A : SPIN_DOWN_SUBLATT_A;
        } else {
            // B sublatt
            map_[i] = spin > 0.0 ? SPIN_UP_SUBLATT_B : SPIN_DOWN_SUBLATT_B;
        }
    }
    return float_wrap(map_);
}

object SQIceGame::GetStateTp1MapColor() {
    //TODO: Review this codes
    // parsing the map and return.
    vector<double> map_(state_tp1.begin(), state_tp1.end());
    // Do things here.
    for (int i=0; i < N; i++) {
        double spin = map_[i];
        if (latt.sub[i] == 1) {
            map_[i] = spin > 0.0 ? SPIN_UP_SUBLATT_A : SPIN_DOWN_SUBLATT_A;
        } else {
            // B sublatt
            map_[i] = spin > 0.0 ? SPIN_UP_SUBLATT_B : SPIN_DOWN_SUBLATT_B;
        }
    }
    return float_wrap(map_);
}

object SQIceGame::GetSublattMap() {
    //TODO: Should the value changes to sublatt AB convention?
    // Or identity just means to be valid.
    vector<double> map_(state_t.begin(), state_t.end());
    for (int i = 0; i < N; i++) {
        if (latt.sub[i] == 1) {
            map_[i] = 1.0;
        } else {
            map_[i] = 0.0;
        }
    }
    return float_wrap(map_);
}

object SQIceGame::GetStateDifferenceMap() {
    // TODO: Compute the state difference of state_t and state_t-1
    vector<double> map_(state_t.begin(), state_t.end());
    return float_wrap(map_);
}

object SQIceGame::GetCanvasMap() {
    return float_wrap(canvas_traj_map);
}

object SQIceGame::GetAgentMap() {
    vector<double> map_(N, 0.0);
    map_[get_agent_site()] = AGENT_OCCUPIED_VALUE;
    return float_wrap(map_);
}

object SQIceGame::GetValidActionMap() {
    vector<double> map_(N, 0.0);
    vector<int> nsites = get_neighbor_sites();
    vector<int> nspins = get_neighbor_spins();

    map_[get_agent_site()] = AGENT_OCCUPIED_VALUE;
    /* 
        Start from here, considering the forsee map
    if (latt.sub[get_agent_site()] == 1) {
        map_[get_agent_site()] = get_agent_spin() > 0 ? SPIN_UP_SUBLATT_A : SPIN_DOWN_SUBLATT_A;
    } else {
        map_[get_agent_site()] = get_agent_spin() > 0 ? SPIN_UP_SUBLATT_B : SPIN_DOWN_SUBLATT_B;
    }
    */

    for (std::vector<int>::size_type i =0; i < nsites.size(); i++) {
        if (latt.sub[nsites[i]] == 1) {
            // sub latt A
            map_[nsites[i]] = nspins[i] > 0 ? SPIN_UP_SUBLATT_A : SPIN_DOWN_SUBLATT_A;
        } else {
            // sub latt B
            map_[nsites[i]] = nspins[i] > 0 ? SPIN_UP_SUBLATT_B : SPIN_DOWN_SUBLATT_B;
        }
    }

    return float_wrap(map_);
}

object SQIceGame::GetDefectMap() {
    for (uint i =0; i < N; i++) {
        double dd = 0.0;
        if (latt.sub[i] == 1) {
            dd = abs(state_tp1[i] + state_tp1[latt.NN[i][0]] + state_tp1[latt.NN[i][1]] + state_tp1[latt.NNN[i][0]]);
            dd /= 4.0;
        }
        //TODO: Make difference of AB sublattice
        //defect_map[i] = DEFECT_MAP_DEFAULT_VALUE - dd;
        defect_map[i] = dd;
    }
    return float_wrap(defect_map);
}

void SQIceGame::PrintLattice() {
    for (int p = 0; p < N; ++p) {
        std::cout << "Site " << p
                  << " is sublatt: " << latt.sub[p]
                  << ", with neigohor: ";
        for (auto nb: latt.NN[p]) {
            std::cout << nb << ", ";
        }
        std::cout << "\n";
    }
    // latt.show_lattice();
}

// ###### private function ######
double SQIceGame::_cal_energy_of_state(const vector<int> & s) {
    double eng = 0.0;

    for (int i = 0; i < N; i++) {
        double se = 0.0;
        for (auto nb : latt.NN[i]) {
            se += s[nb];
        }
        se *= s[i];
        eng += J1 * se;
    }
    eng /= 2.0;
    eng /= N;
    return eng;
}

double SQIceGame::_cal_energy_of_site(const vector<int> &s, int site) {
    // TODO: FIX
    double se = 0.0;
    int i = site;
    if (latt.sub[i] == 1) {
        se = s[i] * (s[latt.NN[i][0]]
                        +s[latt.NN[i][1]]
                        +s[latt.NN[i][2]]
                        +s[latt.NN[i][3]]
                        +s[latt.NNN[i][0]]
                        +s[latt.NNN[i][2]]
                        );
    } else {
        se = s[i] * (s[latt.NN[i][0]]
                        +s[latt.NN[i][1]]
                        +s[latt.NN[i][2]]
                        +s[latt.NN[i][3]]
                        +s[latt.NNN[i][1]]
                        +s[latt.NNN[i][3]]
                        );
    }
    se = J1 * se;
    return se;
}

double SQIceGame::_cal_defect_density_of_state(const vector<int> & s) {
    double dd = 0.0;
    //for (int i = 0; i < N; ++i) {
        //dd += s[latt.NN[i]];
    //}
    dd /= 2.0;
    dd /= N;
    return abs(dd);
}

double SQIceGame::_cal_mean(const vector<int> &s) {
    return 1.0* std::accumulate(s.begin(), s.end(), 0LL) / s.size();
}

double SQIceGame::_cal_stdev(const vector<int> &s) {
    double mean = _cal_mean(s);
    vector<double> diff(s.size());
    std::transform(s.begin(), s.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum/ s.size());
    return stdev;
}

void SQIceGame::_print_vector(const vector<double> &v) {
    std::cout << "[";
    for (const auto &i: v)
        std::cout << i << ", ";
    std::cout << "]\n";
}

bool SQIceGame::_is_visited(int site) {
    bool visited = false;
    if (std::find(agent_site_trajectory.begin(), 
                    agent_site_trajectory.end(), site) != agent_site_trajectory.end()) {
        visited = true;
    }
    return visited;
}

bool SQIceGame::_is_start_end_meets(int site) {
    return (site == init_agent_site) ? true : false;
}

bool SQIceGame::_is_traj_continuous() {
    // LEGACY CODE
    bool cont = true;
    // try walk through traj with operation
    // check next site is in its neighbors
    for (std::vector<int>::size_type i = 1;  i < agent_site_trajectory.size(); i++) {
        std::cout << "step " << i << endl;
        int dir = get_direction_by_sites(agent_site_trajectory[i-1], agent_site_trajectory[i]); 
        std::cout << "From " << agent_site_trajectory[i-1] << " to " << agent_site_trajectory[i] 
                << " is done by action " << dir << "\n"; 
        if (dir == -1) {
            cont = false;
            break;
        }
    }
    return cont;
}

bool SQIceGame::_is_traj_intersect() {
    /*
        Not complete
    */
    bool meet = false;
    std::vector<int>::iterator p = std::find(ep_site_counters.begin(), ep_site_counters.end(), 2);
    if (p != ep_site_counters.end()) {
        meet = true;
    }
    return meet;
}

int SQIceGame::_cal_config_t_difference() {
    int diff_counter = 0;
    for (size_t i = 0 ; i < N; i++) {
        if (state_0[i] != state_t[i]) {
            diff_counter++;
            diff_map[i] = OCCUPIED_MAP_VALUE;
        }
    }
    return diff_counter;
}
int SQIceGame::_count_config_difference(const vector<int> &c1, const vector<int> &c2) {
    // Navie method compute the difference of two configurations
    int counter = 0;
    for (size_t i = 0 ; i < N; i++) {
        if (c1[i] != c2[i]) {
            counter++;
        }
    }
    return counter;
}

vector<int> SQIceGame::LongLoopAlgorithm() {
    // Return the list of proposed sites according to ice rule.
    // Think more deeperly about the usage of this function.
    std::cout << "[GAME] Execute the Long Loop Algorithm.\n";

    vector<int> segments;

    int start_site = get_agent_site();
    vector<int> neighbors = get_neighbor_sites();
    // We update the state_tp1, right?
    int head_sum = _icerule_head_check(start_site);
    int tail_sum = _icerule_tail_check(start_site);

    // Check the starting point is legal
    bool is_safe = false;
    if ((tail_sum == 0) && (head_sum == 0)) {
        is_safe = true;
        #ifdef DEBUG
            std::cout << "IceRule starting point: " << start_site << "\n";
        #endif
    } else {
        std::cout << "[GAME] WARNING!, starting point breaks ice rule, head = "
                << head_sum
                << ", tail= "
                << tail_sum << "\n";
    }

    // Save the ending condition
    vector<int> endsites = _end_sites(start_site);
    #ifdef DEBUG
        std::cout << "Ending sites are " 
                  << endsites[0] << ", "
                  << endsites[1] << ".\n";
    #endif

    // Flip on site and start Loop Algorithm.
    put_and_flip_agent(start_site);

    // === LOOP ALGORITHM === //
    bool stop = false;
    int status = 0; // NEED MORE CAREFUL HANDLINGS
    /* status:
        0: every thing is fine
        1: loop accept
        2: loop reject
        3: timeout
    */
    unsigned int lcter = 0; // loop counter
    int start_spin = get_agent_spin();
    segments.emplace_back(start_site);
    int curr_site = start_site;
    int new_site = 0;
    do {
        lcter++;

        new_site = _loop_extention(curr_site, start_spin);

        // Several nested conditions 
        //  * Stop conditions: termination, defect
        //  * Loop completion,
        //  * Keep growing,
        if (new_site == -1) {
            // Fail to grow the loop, meet defect.
            stop = true;
            status = 2;
        } else if (lcter >= N) {
            // Timeout when walks more than num of sites.
            stop = true;
            status = 3;
        } else if ((new_site == endsites[0]) || (new_site == endsites[1])) {
            // Long loop is created.
            status = 1;
            segments.emplace_back(new_site);
            put_and_flip_agent(new_site);
            stop = true;
        } else {
            // Grow the loop as usual.
            segments.emplace_back(new_site);
            put_and_flip_agent(new_site);
        }

        // update to new_site 
        curr_site = new_site;

    } while (!stop);

    #ifdef DEBUG
    std::cout << "Leave the Loop algorithm safely, at least.\n";
    switch (status) {
        case 0 :
            std::cout << "[status " << status << "]: loop growing\n";
        break;
        case 1 :
            std::cout << "[status " << status << "]: Loop Accepted!\n";
        break;
        case 2 :
            std::cout << "[status " << status << "]: Loop Rejected.\n";
        break;
        case 3 :
            std::cout << "[status " << status << "]: Timeout.\n";
        break;
    }
    #endif

    return segments;
}  


int SQIceGame::_loop_extention(int curr_site, int start_spin){
    int new_site = -1;

    int icerule_sum;
    int s1, s2, s3, n1, n2, n3;
    if (get_spin(curr_site) == start_spin) {
        // spin up
        n1 = latt.NN[curr_site][3];
        n2 = latt.NN[curr_site][4];
        n3 = latt.NN[curr_site][5];
    } else {
        // spin down
        n1 = latt.NN[curr_site][0];
        n2 = latt.NN[curr_site][1];
        n3 = latt.NN[curr_site][2];
    }
    int s0 = state_tp1[curr_site];
    s1 = state_tp1[n1];
    s2 = state_tp1[n2];
    s3 = state_tp1[n3];
    // handy way to check rule
    icerule_sum = -1*s0 + s1 + s2 + s3;

    #ifdef DEBUG
        std::cout << "site: " << curr_site 
            << ", icerule sum = " << icerule_sum << "\n";
    #endif

    if (icerule_sum == 0) {
        double dice = uni01_sampler();
        if (s1 == s2) {
            new_site = dice > 0.5 ? n1 : n2;
        } else if (s2 == s3) {
            new_site = dice > 0.5 ? n2 : n3;
        } else if (s3 == s1) {
            new_site = dice > 0.5 ? n3 : n1;
        }
    } else {
        std::cout << "[GAME] __extention fails!\n";
        new_site = -1;
    }

    return new_site;
}

int SQIceGame::_icerule_head_check(int site) {
    vector<int> neighbors = _get_neighbor_of_site(site);
    int sum = state_tp1[site] + 
                state_tp1[neighbors[0]] + 
                state_tp1[neighbors[1]] + 
                state_tp1[neighbors[2]];
    return sum;
}


int SQIceGame::_icerule_tail_check(int site) {
    vector<int> neighbors = _get_neighbor_of_site(site);
    int sum = state_tp1[site] + 
                state_tp1[neighbors[3]] + 
                state_tp1[neighbors[4]] + 
                state_tp1[neighbors[5]];
    return sum;
}

vector<int> SQIceGame::_end_sites(int site) {
    // Get the ending sites of the loop
    int NN0 = latt.NN[site][0];
    int NN1 = latt.NN[site][1];
    int NN2 = latt.NN[site][2];
    vector<int> ends(2);
    if (state_tp1[NN0] == state_tp1[NN1]) {
        ends[0] = NN0;
        ends[1] = NN1;
    } else if (state_tp1[NN1] == state_tp1[NN2]) {
        ends[0] = NN1;
        ends[1] = NN2;
    } else if (state_tp1[NN2] == state_tp1[NN0]) {
        ends[0] = NN2;
        ends[1] = NN0;
    } else {
        ends[0] = -1;
        ends[1] = -1;
        std::cout << "[GAME] WARNING: Ending condition fails!\n";
    }
    return ends;
}