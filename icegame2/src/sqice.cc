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

    start_site = NULL_SITE;
    start_spin = NULL_SPIN;

    // Set initial episode to 1, avoid division by zero
    num_episode = 1; 
    num_total_steps = 0;
    same_ep_counter = 0;
    updated_counter = 0;
    num_updates = 0;
    num_config_resets = 0;

    mag_fields.emplace_back(h1_t);
    mag_fields.emplace_back(h2_t);
    mag_fields.emplace_back(h3_t);

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
    //diff_map.resize(N, EMPTY_MAP_VALUE);

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
    //std::fill(diff_map.begin(), diff_map.end(), EMPTY_MAP_VALUE);
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
    vector<int> backup = ice_config.Ising;
    ice_config.Ising = state_tp1;
    state_0 = state_tp1;
    state_t = state_tp1;
    // ... Sanity check!
    // NOTE: Now, there is no defect density check!
    if ( _cal_defect_density_of_state(ice_config.Ising) == AVERAGE_GORUND_STATE_DEFECT_DENSITY \
        && _cal_energy_density_of_state(ice_config.Ising) == AVERAGE_GORUND_STATE_ENERGY) {
        std::cout << "[GAME] Updated Succesfully!\n";
        updated_counter++; 
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
    // refresh the configuration
    std::fill(ice_config.Ising.begin(), ice_config.Ising.end(), 1);
    // Run SSF Update again.
    MCRun(MCSTEPS_TO_EQUILIBRIUM);
    Reset();
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
    double energy_density = _cal_energy_density_of_state(cfg);
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
        std::cout << "[GAME] Set Ice configuration from python successfully.\n";
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

    std::cout << "[GAME] Average Energy E = " << _cal_energy_density_of_state(state_t) << "\n";
    //std::cout << "[GAME] Defect Density D = " << _cal_defect_density_of_state(state_0) << "\n";
    //std::cout << "[GAME] Config mean = " << config_mean << " , and std = " << config_stdev << "\n";
    // ======== FAILS BELOW ====== //
    /*
    config_mean = _cal_mean(state_0);
    config_stdev = _cal_stdev(state_0);

    std::cout << "[GAME] Average Energy E = " << _cal_energy_density_of_state(state_0) << "\n";
    std::cout << "[GAME] Defect Density D = " << _cal_defect_density_of_state(state_0) << "\n";
    std::cout << "[GAME] Config mean = " << config_mean << " , and std = " << config_stdev << "\n";
    */

}

// Start: Just put the agent on the site
// Or: We use put and flip?
int SQIceGame::Start(int init_site) {
    int ret = put_agent(init_site);
    init_agent_site = init_site;
    // Do we need to flip?, no
    start_spin = get_agent_spin();
    // But we would call loop algo
    if (ret != agent_site) { 
        std::cout << "[GAME] WARNING: Start() get wrong init_site!\n";
    }
    return agent_site;
}

void SQIceGame::ClearBuffer() {
    clear_all();
    same_ep_counter++;
}

// Reset: Clear buffer and 
int SQIceGame::Reset(int init_site) {
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
    start_site = NULL_SITE;
    start_spin = NULL_SPIN;
}

// member functions 
vector<double> SQIceGame::Metropolis() {
    /* Metropolis operation called by agent.
        Returns: 
            results = []
    */
   // Note: Why we use state_tp1 rather than state_t?
    vector<double> rets;
    bool is_accept = false;
    double E0 = _cal_energy_density_of_state(state_t); // check this
    double Et = _cal_energy_density_of_state(state_tp1);
    double dE = Et - E0;
    // defect function now is not fully supported!
    double dd = _cal_defect_density_of_state(state_tp1);
    // explicit function call
    int diff_counts = _count_config_difference(state_t, state_tp1);
    double diff_ratio = diff_counts / double(N);

    // calculates returns
    if (dE == 0.0) {
        if (dd != 0.0) {
            // this condition is used as sanity check
            std::cout << "[Game]: State has no energy changes but contains defects! Sanity checking fails!\n";
        }
        is_accept = true;
        rets.emplace_back(ACCEPT_VALUE);
    } else {
        rets.emplace_back(REJECT_VALUE);
    }

    rets.emplace_back(dE);
    rets.emplace_back(diff_ratio);

    // update counters
    action_statistics[METROPOLIS_PROPOSAL]++;
    ep_action_counters[METROPOLIS_PROPOSAL]++;
    num_total_steps++;
    // Not an Episode
    same_ep_counter++;
    ep_step_counter++;
    num_updates++;
    return rets;
}

int SQIceGame::_flip_state_tp1_site(int site) {
    // maybe we need cal some info when flipping
    int spin = 0;
    if(site >= 0 && site < N) {
        int index = _site1d_to_index(site);
        state_tp1[index] *= -1;
        spin = state_tp1[index];
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
        int index = _site1d_to_index(site);
        state_t[index] *= -1;
        spin = state_t[index];
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
void SQIceGame::flip_along_trajectory(const vector<int>& traj) {
    // check traj is not empty
    // check traj is cont. or not
    // flip along traj on state_t
    // done, return what?
    #ifdef DEBUG
        std::cout << "Start flip along trajectory inside Game.\n";
    #endif 
    if(!traj.empty()) {
        for (auto const & site : traj) {
            _flip_state_t_site(site);
            #ifdef DEBUG
            std::cout << "--> Flip site " << site 
                    << " and dE = " << _cal_energy_density_of_state(state_t) - _cal_energy_density_of_state(state_t)
                    << ", dn = " << _cal_defect_number_of_state(state_t) 
                    << ", dd = " << _cal_defect_density_of_state(state_t) << endl;
            #endif
        }
    } else {
        std::cout << "[GAME] WARNING: Attempt to flip empty trajectory list!\n";
    }
}

void SQIceGame::FollowTrajectory(const boost::python::object &iter) {
    std::vector<int> traj= std::vector<int> (boost::python::stl_input_iterator<int> (iter),
                                                    boost::python::stl_input_iterator<int> ());
    // Notice: this function would not increase counters.
    if(!traj.empty()) {
        for (auto const & site : traj) {
            put_and_flip_agent(site);
        }
    } else {
        std::cout << "[GAME] WARNING: Attempt to follow empty trajectory list!\n";
    }
}

void SQIceGame::push_site_to_trajectory(int site) {
    // Just used for taking care of trajectory.
    if (site >= 0 && site < N) {
        // If starting point
        if (agent_site_trajectory.size() == 0) {
            agent_site_trajectory.emplace_back(site);
            agent_spin_trajectory.emplace_back(get_spin(site));
            init_agent_site = site;
        } else {
            // TODO: REMOVE THIS CONSTRAINTS.
            agent_site_trajectory.emplace_back(site);
            agent_spin_trajectory.emplace_back(get_spin(site));
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
        int idx = _site1d_to_index(site);
        spin = state_tp1[idx];
    }
    return spin;
}

int SQIceGame::get_agent_spin() {
    return get_spin(agent_site);
}

// GRAY ZONE.

vector<double> SQIceGame::Move(int dir_idx) {
    // Move and Flip?
    vector<double> rets;
    double prev_eng = _cal_energy_density_of_state(state_tp1);
    int new_site = get_site_by_direction(dir_idx);
    int new_agent_site = put_and_flip_agent(new_site);
    if (new_site != new_agent_site) {
        std::cout << "[GAME] Warning! Put agent on wrong site!\n";
    }
    double curr_eng = _cal_energy_density_of_state(state_tp1);
    double dE = curr_eng - prev_eng;
    // diff ratio is also different from metropolis.
    int diff_counts = _count_config_difference(state_t, state_tp1);
    double diff_ratio = diff_counts / double(N);
    // Add information to rets. What do we need?
    if (dE == 0.0) {
        // this move follow ice-rule
        rets.emplace_back(ACCEPT_VALUE);
    } else {
        rets.emplace_back(REJECT_VALUE);
    }
    rets.emplace_back(dE);
    rets.emplace_back(diff_ratio);

    // counter increments
    ep_action_counters[dir_idx]++;
    ep_action_list.emplace_back(dir_idx);
    action_statistics[dir_idx]++;

    return rets;
}

// TO BE REMOVED.
vector<double> SQIceGame::Draw(int dir_idx) {
    // The function handles canvas and calculates step-wise returns
    // TODO Extend action to 8 dir
    int curt_spin = get_agent_spin();
    vector<double> rets(4);
    // get where to go
    // NOTICE: get_spin returns state_tp1 spin
    int next_spin = get_spin(get_site_by_direction(dir_idx));
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

    double dE = _cal_energy_density_of_state(state_tp1) - _cal_energy_density_of_state(state_t);
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
    // Flip: usually used as initialization, defect creation.
    vector<double> rets;

    // Flip the current agent site.
    flip_agent();

    double dE = _cal_energy_density_of_state(state_tp1) - _cal_energy_density_of_state(state_t);
    //double dd = _cal_defect_density_of_state(state_tp1);
    double dC = _count_config_difference(state_t, state_tp1) / double(N);

    rets.emplace_back(dE);
    rets.emplace_back(dC);

    return rets;
}

/// LEGACY CODES, TO BE REMOVED.
int SQIceGame::go(int dir) {
    // One of the core function.
    // this function handles moveing, so the ep_action should be counted.
    int new_site = get_site_by_direction(dir);

    int agent_site = put_and_flip_agent(new_site);

    // action statistics only be counted when called by icegame_env.
    ep_action_counters[dir]++;
    ep_action_list.emplace_back(dir);
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
    same_ep_counter++;
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

ActDir SQIceGame::how_to_go(int site) {
    // Direction --> Site Mapping
    ActDir dir = get_direction_by_sites(agent_site, site);
    return dir;
}

vector<double> SQIceGame::GetPhyObservables() {
    // Get defect density and energy density of the system
    /*
        How many phyiscal qunatities are relevant?
            * Energy Density
            * Configuration Defference Ratio
        Is loop length important here?
    */

    vector<double> obs;
    double eng_density = _cal_energy_density_of_state(state_tp1); // this is already the density
    double diff_eng_density = eng_density - _cal_energy_density_of_state(state_t);
    //double def_density = _cal_defect_density_of_state(state_tp1); // well, it seems no need for this.
    double config_diff_ratio = _count_config_difference(state_t, state_tp1) / double(N);
    obs.emplace_back(eng_density);
    obs.emplace_back(diff_eng_density);
    obs.emplace_back(config_diff_ratio);

    return obs;
}

vector<int> SQIceGame::get_neighbor_spins() {
    vector<int> nsites = get_neighbor_sites();
    vector<int> nspins;
    for(const auto& s : nsites) {
        int idx = _site1d_to_index(s);
        // NOTE: Use tp1 rather than t?
        nspins.emplace_back(state_tp1[idx]);
    }
    return nspins;
}

vector<int> SQIceGame::_get_neighbor_of_site(int _site) {
    // Get neighboring sites by given site
    int index = _site1d_to_index(_site);
    vector<int> locals(6);
    locals[0] = _index_to_site1d(latt.NN[index][0]);
    locals[1] = _index_to_site1d(latt.NN[index][1]);
    locals[2] = _index_to_site1d(latt.NN[index][2]);
    locals[3] = _index_to_site1d(latt.NN[index][3]);
    locals[4] = _index_to_site1d(latt.NN[index][4]);
    locals[5] = _index_to_site1d(latt.NN[index][5]);
    return locals;
}

vector<int> SQIceGame::_get_neighbor_of_index(int index) {
    // Get neighboring indices by given indices
    vector<int> locals(6);
    locals[0] = latt.NN[index][0];
    locals[1] = latt.NN[index][1];
    locals[2] = latt.NN[index][2];
    locals[3] = latt.NN[index][3];
    locals[4] = latt.NN[index][4];
    locals[5] = latt.NN[index][5];
    return locals;
}

vector<int> SQIceGame::get_local_sites() {
    vector<int> lsites = get_neighbor_sites();
    lsites.emplace_back(get_agent_site());
    return lsites;
}

vector<int> SQIceGame::get_local_spins() {
    vector<int> lspins = get_neighbor_spins();
    lspins.emplace_back(get_agent_spin());
    return lspins;
}

vector<int> SQIceGame::get_neighbor_sites() {
    return _get_neighbor_of_site(agent_site);
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
            idxes.emplace_back(i);
        }
    }
    vector<int> candidates;
    for (auto const & idx : idxes) {
        candidates.emplace_back(nsites[idx]);
    }
    std::random_shuffle(candidates.begin(), candidates.end());
    // should i avoid repeating?
    return candidates;
}

// TODO::WARNING: OLD CONVENTIONS!
int SQIceGame::get_site_by_direction(int dir_idx) {
    int site = agent_site;
    // c++11 method for casting
    auto dir = static_cast<ActDir>(dir_idx);

    // Get the neighboring sites
    vector<int> neighbors = get_neighbor_sites();
    // print for checking
    int new_site;
    switch (dir) {
        case ActDir::Head_0:
            new_site = neighbors[0];
            break;
        case ActDir::Head_1:
            new_site = neighbors[1];
            break;
        case ActDir::Head_2:
            new_site = neighbors[2];
            break;
        case ActDir::Tail_0:
            new_site = neighbors[3];
            break;
        case ActDir::Tail_1:
            new_site = neighbors[4];
            break;
        case ActDir::Tail_2:
            new_site = neighbors[5];
            break;

        default:
            new_site = site; // not move
            break;
    }

    // Find new site according to the sublattice
    #ifdef DEBUG
      std::cout << "get_neighbor_site_by_direction(dir=" << dir_idx << ") = " 
                << new_site << " with agent site = " << agent_site << " \n";
    #endif
    return new_site;
}

// Maybe we can implement a ask_guide here.
/*
    Move: directional index is designed as
      * head dir
      * tail dir
*/
ActDir SQIceGame::get_direction_by_sites(int site, int next_site) {
    // What do we get?, site should be site
    ActDir dir;
    return dir;
}

/* GetMaps:
    TODO: add more map tricks
*/

vector<int> SQIceGame::GetStateTMap() {
    // WARING: BUGS!, Head of values would go crazy.
    // It is safer for python to do the type conversion.
    vector<int> ordered_state_t;
    for (int i = 0; i < N; i++) {
        int spin = state_t[latt.indices[i]];
        ordered_state_t.emplace_back(spin);
        #ifdef false
          std::cout << "index =  " << i 
            << ", 1D: " << latt.site1d[i] 
            << ", state_t = " << spin
            << "; order_state = " << ordered_state_t[i]
        << "\n";
        #endif
    }
    return ordered_state_t;
}

vector<int> SQIceGame::GetStateTp1Map() {
    vector<int> ordered_state_tp1;
    for (int i = 0; i < N; i++) {
        int spin = state_tp1[latt.indices[i]];
        ordered_state_tp1.emplace_back(spin);
    }
    return ordered_state_tp1;
}

vector<int> SQIceGame::GetStateDiffMap() {
    vector<int> ordered_diff;
    for (int i =0; i < N; i++) {
        int spin = (state_tp1[latt.indices[i]] - state_t[latt.indices[i]])/2;
        //int spin = state_tp1[latt.site1d[i]] - state_t[latt.site1d[i]];
        // Should we normalize from [-2, 2] back to [-1, 1]?
        ordered_diff.emplace_back(spin);
    }
    return ordered_diff;
}

vector<int> SQIceGame::GetStateDiff() {
    // get state_tp1 - state_t
    vector<int> diff;
    std::transform(state_tp1.begin(), state_tp1.end(), state_t.begin(),
        std::back_inserter(diff), [&](int tp1, int t) {return tp1-t;});
    return diff;
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

int SQIceGame::_cal_energy_of_state(const vector<int> & s) {
    int eng = 0;
    for (int i = 0; i < N; i++) {
        int se = 0;
        for (auto nb : latt.NN[i]) {
            se += s[nb];
        }
        se *= s[i];
        eng += se; // J1 is double?, ignore first
    }
    eng = eng >> 2;
    return eng;
}

double SQIceGame::_cal_energy_density_of_state(const vector<int> & s) {
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

int SQIceGame::_cal_defect_number_of_state(const vector<int> &s) {
    int num_defects = 0;
    for (int i = 0 ; i < N; i++) {
        //if (0 == latt.sub[i]) {
            int n0 = latt.NN[i][0];
            int n1 = latt.NN[i][1];
            int n2 = latt.NN[i][2];
            num_defects += (s[i] + s[n0] + s[n1] + s[n2]);
        //}
    }
    return num_defects;
}

double SQIceGame::_cal_defect_density_of_state(const vector<int> & s) {
    double dd = 0.0;
    int num_defects = _cal_defect_number_of_state(s);
    dd = static_cast<double>(num_defects)/N;
    return dd;
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

template <class T>
void SQIceGame::_print_vector(const vector<T> &v, const std::string &prefix) {
    std::cout << prefix << ": [";
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
        ActDir dir = get_direction_by_sites(agent_site_trajectory[i-1], agent_site_trajectory[i]); 
        std::cout << "From " << agent_site_trajectory[i-1] << " to " << agent_site_trajectory[i] << "\n";
        if (dir == ActDir::NOOP) {
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

// Clear, explicitly compuate the difference.
int SQIceGame::_count_config_difference(const vector<int> &c1, const vector<int> &c2) {
    // Navie method compute the difference of two configurations
    int counter = 0;
    for (int i = 0 ; i < N; i++) {
        if (c1[i] != c2[i]) {
            counter++;
        }
    }
    return counter;
}

vector<int> SQIceGame::LongLoopAlgorithm() {
    // Return the list of proposed sites according to ice rule.
    // Think more deeperly about the usage of this function.
    // Use sites here!
    std::cout << "[GAME] Execute the Long Loop Algorithm.\n";

    vector<int> segments;

    start_site = get_agent_site();
    vector<int> neighbor_sites = get_neighbor_sites();
    // We update the state_tp1, right?
    int head_sum = _icerule_head_check(start_site);
    int tail_sum = _icerule_tail_check(start_site);

    #ifdef DEBUG
      std::cout << "Stating spin (before): " << get_agent_spin() << "\n";
      std::cout << "\t head = " << head_sum << ", tail = " << tail_sum << "\n";
    #endif

    // Check the starting point is legal
    bool is_safe = false;
    if ((tail_sum == 0) && (head_sum == 0)) {
        is_safe = true;
        #ifdef DEBUG
            std::cout << "IceRule starting point: " << start_site << "\n";
        #endif
    } else {
        // False alarm occurs
        std::cout << "[GAME] WARNING!, starting point breaks ice rule, head = "
                << head_sum
                << ", tail= "
                << tail_sum 
                << ", E = " 
                << _cal_energy_density_of_state(state_tp1)
                << "\n";
    }

    // Save the ending condition
    vector<int> end_sites = _end_sites(start_site);
    #ifdef DEBUG
        std::cout << "Starting site: "
                  << start_site
                  << ", Ending sites are " 
                  << end_sites[0] << ", "
                  << end_sites[1] << ".\n";
        _print_vector(get_neighbor_sites(), "Neighbor Sites");
        _print_vector(get_neighbor_spins(), "Neighbor Spins (Before flipping)");
    #endif

    // Flip on site and start Loop Algorithm.
    put_and_flip_agent(start_site);
    #ifdef DEBUG
      std::cout << "Stating spin (after): " << get_agent_spin() << "\n";
      _print_vector(get_neighbor_spins(), "Starting Neighbor Spins Flipped");
    #endif

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
    start_spin = get_agent_spin(); //Now, start spin is the member data
    segments.emplace_back(start_site);
    int curr_site = start_site;
    int new_site = 0;
    do {
        lcter++;

        // TODO: How to prevent extention failures.
        new_site = _loop_extention(curr_site);

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
        } else if ((new_site == end_sites[0]) || (new_site == end_sites[1])) {
            // It may meet the starting point?
            // Long loop is created.
            status = 1;
            segments.emplace_back(new_site);
            /// OOOOO! BUG: the 'Actual' site used here
            put_and_flip_agent(new_site);
            #ifdef DEBUG
              std::cout << "\tAgent spin: " << get_agent_spin() << "\n";
              _print_vector(get_neighbor_sites(), "\tNeighbor Sites");
              _print_vector(get_neighbor_spins(), "\tNeighbor Spins (Before flipping)");
              std::cout << "\tE = " << _cal_energy_density_of_state(state_tp1) << "\n";
            #endif
            stop = true;
        } else {
            // Grow the loop as usual.
            segments.emplace_back(new_site);
            put_and_flip_agent(new_site);
            #ifdef DEBUG
              std::cout << "\tAgent spin: " << get_agent_spin() << "\n";
              _print_vector(get_neighbor_sites(), "\tNeighbor Sites");
              _print_vector(get_neighbor_spins(), "\tNeighbor Spins (Before flipping)");
              std::cout << "\tE = " << _cal_energy_density_of_state(state_tp1) << "\n";
            #endif
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

vector<int> SQIceGame::GuideAction() {
    //Args:
    //  * According to given site and corresponding spin, giving guides.
    //  * return vector of candidates of action
    // * But this function can be more tricky, that energy should be considered.
    // ** Hope this function is modified to propose one step which doesnt change energy.
    vector<int> candidates;
    int curr_site = agent_site;
    int curr_idx = _site1d_to_index(curr_site);

    int s1, s2, s3, n1, n2, n3;
    // latt returns index
    bool is_head = false;
    if (get_spin(curr_site) == start_spin) {
        // spin up
        n1 = latt.NN[curr_idx][3];
        n2 = latt.NN[curr_idx][4];
        n3 = latt.NN[curr_idx][5];
        is_head = false;
    } else {
        // spin down
        n1 = latt.NN[curr_idx][0];
        n2 = latt.NN[curr_idx][1];
        n3 = latt.NN[curr_idx][2];
        is_head = true;
    }
    int s0 = state_tp1[curr_idx];
    s1 = state_tp1[n1];
    s2 = state_tp1[n2];
    s3 = state_tp1[n3];
    // handy way to check rule
    int icerule_sum = -1*s0 + s1 + s2 + s3;

    if (icerule_sum == 0) {
        // Head or Tail
        if (is_head) {
            if (s1 == s2) {
                // Choose n1, n2
                candidates.emplace_back(static_cast<int>(ActDir::Head_0));
                candidates.emplace_back(static_cast<int>(ActDir::Head_1));
            } else if (s2 == s3) {
                // Choose n2, n3
                candidates.emplace_back(static_cast<int>(ActDir::Head_1));
                candidates.emplace_back(static_cast<int>(ActDir::Head_2));
            } else if (s3 == s1) {
                // Choose n1, n3
                candidates.emplace_back(static_cast<int>(ActDir::Head_0));
                candidates.emplace_back(static_cast<int>(ActDir::Head_2));
            }
        } else {
            // Tail
            if (s1 == s2) {
                // Choose n1, n2
                candidates.emplace_back(static_cast<int>(ActDir::Tail_0));
                candidates.emplace_back(static_cast<int>(ActDir::Tail_1));
            } else if (s2 == s3) {
                // Choose n2, n3
                candidates.emplace_back(static_cast<int>(ActDir::Tail_1));
                candidates.emplace_back(static_cast<int>(ActDir::Tail_2));
            } else if (s3 == s1) {
                // Choose n1, n3
                candidates.emplace_back(static_cast<int>(ActDir::Tail_0));
                candidates.emplace_back(static_cast<int>(ActDir::Tail_2));
            }
        }
    } else {
        std::cout << "[GAME] Guide action fails!\n"; // show or not?
        candidates.emplace_back(NULL_SITE);
    }

    return candidates;
}

int SQIceGame::_loop_extention(int curr_site){
    // NOTE: site in this function means index.
    // NOTE: in some sense, this function do suggestion
    int new_site = NULL_SITE;
    int new_idx = NULL_SITE;
    int curr_idx = _site1d_to_index(curr_site);

    int s1, s2, s3, n1, n2, n3;
    // latt returns index
    if (get_spin(curr_site) == start_spin) {
        // spin up
        n1 = latt.NN[curr_idx][3];
        n2 = latt.NN[curr_idx][4];
        n3 = latt.NN[curr_idx][5];
    } else {
        // spin down
        n1 = latt.NN[curr_idx][0];
        n2 = latt.NN[curr_idx][1];
        n3 = latt.NN[curr_idx][2];
    }
    int s0 = state_tp1[curr_idx];
    s1 = state_tp1[n1];
    s2 = state_tp1[n2];
    s3 = state_tp1[n3];
    // handy way to check rule
    int icerule_sum = -1*s0 + s1 + s2 + s3;

    #ifdef DEBUG
        std::cout << "site: " << curr_site 
            << ", icerule sum = " << icerule_sum << "\n";
    #endif

    if (icerule_sum == 0) {
        double dice = uni01_sampler();
        if (s1 == s2) {
            new_idx = dice > 0.5 ? n1 : n2;
        } else if (s2 == s3) {
            new_idx = dice > 0.5 ? n2 : n3;
        } else if (s3 == s1) {
            new_idx = dice > 0.5 ? n3 : n1;
        }
        new_site = _index_to_site1d(new_idx);
    } else {
        std::cout << "[GAME] __extention fails!\n";
        new_idx = NULL_SITE;
        new_site = NULL_SITE;
    }
    // map index back to site

    return new_site;
}

int SQIceGame::_icerule_head_check(int site) {
    //TODO: Change to index based function!
    vector<int> neighbors = _get_neighbor_of_index(_site1d_to_index(site));
    int idx = _site1d_to_index(site);
    int sum = state_tp1[idx] + 
                state_tp1[neighbors[0]] + 
                state_tp1[neighbors[1]] + 
                state_tp1[neighbors[2]];
    return sum;
}

int SQIceGame::_icerule_tail_check(int site) {
    // get indices of the given site
    vector<int> neighbors = _get_neighbor_of_index(_site1d_to_index(site));
    int idx = _site1d_to_index(site);
    int sum = state_tp1[idx] + 
                state_tp1[neighbors[3]] + 
                state_tp1[neighbors[4]] + 
                state_tp1[neighbors[5]];
    return sum;
}

vector<int> SQIceGame::_end_sites(int site) {
    // Get the ending sites of the loop
    int idx = _site1d_to_index(site);
    // the following are indices
    int NN0 = latt.NN[idx][0];
    int NN1 = latt.NN[idx][1];
    int NN2 = latt.NN[idx][2];
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
    // map back to sites
    return _indices_to_sites1d(ends);
}

vector<int> SQIceGame::_indices_to_sites1d(const vector<int> indices) {
    vector<int> site1d;
    for (const auto &i : indices) {
        site1d.emplace_back(latt.site1d[i]);
    }
    return site1d;
}

vector<int> SQIceGame::_sites1d_to_indices(const vector<int> sites) {
    vector<int> indices;
    for (const auto &s : sites) {
        indices.emplace_back(latt.indices[s]);
    }
    return indices;
}

void SQIceGame::show_information() {
    // Game information
    std::cout << "Game information --- \n";
    std::cout << "\tGlobal step: " << num_total_steps << " - Local step: "
        << same_ep_counter << " in " << num_episode << " episode.\n";
    std::cout << "\t Number of configuration resets: " << num_config_resets << "\n";
    std::cout << "\t Number of successful updates: " <<  updated_counter << "\n";

    // Agent information
    std::cout << "Agent information --- \n";
    std::cout << "\tAgent site: " << agent_site  << " with spin: " << get_agent_spin() << "\n";
    _print_vector(get_neighbor_sites(), "\tNeighbor sites");
    _print_vector(get_neighbor_spins(), "\tNeighbor spins");
    _print_vector(GuideAction(), "\tAction suggestions");
    std::cout << "\tEnergy Density = " << _cal_energy_density_of_state(state_tp1) << "\n";
    // How about some statistics information? // majorly processed in pyhton.
}