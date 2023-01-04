/*!
 *  \file   simulation.h
 *  \author A. Fabbri (alessandro.fabbri27@unibo.it), C. Mizzi (chiara.mizzi2@unibo.it), S. Sinigardi (stefano.sinigardi@unibo.it)
 *  \brief  Declaration file for #simulation object.
 *  \details This file contains the definitions, as well as some implementation, of #simulation class together with various support objects. Please note that the method #simulation::run is template so is implemented inside this header file.
 */

#ifndef _MINIMOCAS_SIMULATION_H
#define _MINIMOCAS_SIMULATION_H

#include <tuple>
#include <cstdio>

#include <physycom/time.hpp>
#include <physycom/histo.hpp>

#include <pawn.h>

#ifdef MESSAGE
#undef MESSAGE
#endif

#ifdef VERBOSE_DEBUG
#undef VERBOSE_DEBUG
#endif

#define VERBOSE_DEBUG 0
#if VERBOSE_DEBUG
#define MESSAGE(...) std::fprintf(stdout, __VA_ARGS__)
#else
#define MESSAGE(...)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// ATTRACTION
////////////////////////////////////////////////////////////////////////////////////////////////////

struct attraction
{
  int node_lid;
  int visitors;
  int visit_time;
  int rate_in;
  std::vector<double> weight;
  std::string tag;
  std::deque<int> queue;
  std::vector<int> timecap;
  std::normal_distribution<double> rnd_vis_time;

  attraction();
  attraction(const std::string &tag, const int &nlid, const std::vector<int> &timecap, const int &visit_time);
};
typedef std::vector<attraction>::iterator attr_it;

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// TRANSPORT
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int DEFAULT_SLICE_TRANSP = 1800;

struct transport
{
  int route_id;
  int trip_id;
  std::vector<int> stops;
  std::unordered_map<int, std::map<int, std::pair<int, int>>> tt; // < nlid, nlid > = time of travel [s]
  int capacity, max_capacity;

  transport();
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// SINK
////////////////////////////////////////////////////////////////////////////////////////////////////

struct sink
{
  int node_lid;
  std::string tag;
  std::vector<int> despawn_dt;

  sink();
  sink(const std::string &tag_, const std::vector<int> &ddt, const int &plid);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// SOURCE
////////////////////////////////////////////////////////////////////////////////////////////////////

enum
{
  SOURCE_STD = 0,
  SOURCE_CTRL = 1,
  SOURCE_TOT = 2
};

struct source
{
  int creation_dt;                 // time interval between creation [s]
  std::vector<int> creation_rate;  // number of pawn created per time interval
  std::string tag;
  jsoncons::json pawns_spec;
  bool is_localized;
  bool is_geofenced;
  bool is_control;
  int source_type;
  int orientation;
  int node_lid;
  int node_lid_dest;
  int poly_lid;
  double s;
  double lat, lon;
  std::vector<std::vector<std::string>> binned_wr;
  std::vector<std::vector<std::pair<std::string, double>>> wr_bin; // < L_idx, W_idx , wroute tag, cumulative weight >

  point_base loc;
  poly_it poly;
  node_it node;

  source();
  source(const std::string &tag_, const jsoncons::json &jconf, cart* c);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// WEIGHTED ROUTE
////////////////////////////////////////////////////////////////////////////////////////////////////

struct wroute
{
  double weight, len;
  std::string tag;
  std::vector<int> node_seq;

  wroute();
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// BARRIER
////////////////////////////////////////////////////////////////////////////////////////////////////

struct barrier
{
  std::string tag;
  point_base loc;
  poly_it poly;
  node_it node;
  double s;
  int cnt_TF, cnt_FT;
  bool geofenced;

  barrier();
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// POLYGON
////////////////////////////////////////////////////////////////////////////////////////////////////

struct polygon
{
  std::vector<point_base> points;
  double area, perimeter;
  point_base centroid;
  std::map<std::string, std::string> pro;
  std::vector<poly_it> poly_in;
  polygon();
  polygon(const jsoncons::json &feature);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// SIMULATION
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr double density_critical     = 4.0;  // pawn per linear meter
constexpr double speed_critical       = 0.15; // m/s
constexpr char separator              = ';';
constexpr double default_beta_bp_miss = 0.5;
constexpr double default_alpha_we     = 1.0;
constexpr double default_height       = 0.0;
constexpr double default_alpha_speed  = 0.0;

enum
{
  DYNW_MODE_OFF = 0,
  DYNW_MODE_LVLPS_HC,
  DYNW_MODE_BRIDGE_LVL,
  DYNW_MODE_PG_CLOSED,
  DYNW_MODE_HYBRID
};

struct simulation
{
  int start_time, stop_time, sim_time;
  std::string start_date, stop_date;
  int midn_t_start, midn_t_stop;
  int dt, sampling_dt;
  int iter, niter;
  int dump_state_dt; // dump full pawn state after this interval
  int dump_cam_dt;

  cart *c;
  std::vector<std::vector<pawn>> pawns;
  std::vector<source> sources;
  std::vector<sink> sinks;
  std::vector<transport> transports;
  std::vector<attraction> attractions;
  std::map<std::string, barrier> barriers;
  std::map<std::string, point_proj> locations;
  std::map<std::string, std::vector<int>> routes;
  std::map<std::string, wroute> attr_wr;
  std::vector<std::string> attr_route;
  std::map<char, int> attr_An;
  std::map<int, std::vector<std::vector<std::string>>> binned_wr;
  std::map<int, std::map<std::string, int>> cam_counters;
  std::vector<polygon> polygons;
  std::set<int> poly_closed;

  char separator;
  std::string state_basename;
  std::string barrier_outfile;
  std::string polygon_outfile;
  std::string poly_outfile;
  std::string grid_outfile;
  std::string uniq_outfile;
  std::ofstream pawn_out;
  std::ofstream net_out;
  std::ofstream influxgrid_out;
  std::ofstream polygon_out;
  std::ofstream population_out;
  std::ofstream status_out;
  std::ofstream pawnstats_out;
  std::ofstream wrstats_out;
  std::vector<int> net_state;
  std::string influx_meas_name;
  grid_base grid;
  bool enable_stats;
  bool enable_status;
  bool enable_uniq_poly;

  // dynamic weights and bestpath
  int dynw_mode;             // dynamic weights mode switch

  // dynamic weights - water lvl ps
  int dynw_time_idx;         // active water lvl ps time index
  int dynw_time_idx2;
  std::vector<int> lvlps_tt; // internal timetable of water level ps
  std::vector<std::vector<int>> pg_closed;

  // weighted route
  double L_max, L_binw;
  int N_bin;
  double a, b, Lc;
  std::discrete_distribution<int> rnd_bmdist;
  std::map<std::string, wroute> wroutes;
  std::string wroute_info;

  // init
  simulation();
  simulation(const jsoncons::json &jconf, cart *c_);
  ~simulation();
  void init_from_file(const jsoncons::json &jconf);

  std::string info();

  // pawn generation core function
  std::vector<std::vector<pawn>> make_pawns(const jsoncons::json &jconf);
  std::vector<std::vector<pawn>> make_pawns_weight(const source &src);
  void make_attr_route();

  // pawn evolution
  void evolve(pawn &p);
  void evolve_poly(pawn &p, const node_it &node);
  void kill(pawn &p, const std::string &mode = "goal");

  // generic utilities
  inline int get_pawn_number()
  {
    return std::accumulate(pawns.begin(), pawns.end(), 0, [](int sum, std::vector<pawn> &p){ return sum + (int)p.size(); });
  }
  bool find_trip(int node_start, int node_stop, std::pair<int, int> &sel_couple);

  // init functions
  void init_barriers(const jsoncons::json &jconf, cart* c);

  // dump utilities
  void dump_pawn_state();
  void dump_net_state();
  void dump_influxgrid();
  void dump_state_json();
  void dump_barriers();
  void dump_polygons();
  void dump_population();
  void dump_geodata();
  void dump_poly_uniq();

  // update mechanics
  void update_sink();
  void update_sources();
  void update_velocity();
  void update_transports();
  void update_attractions();
  void update_weights();
  void update_barriers();
  void update_attr_weights();
  void update_status();

  // internal stuff
  enum
  {
    NODE_BASE       = 1,
    NODE_SINK       = 2,
    NODE_SOURCE     = 4,
    NODE_ATTRACTION = 8,
  };
  std::map<int, int> node_status;                          // < node_lid , node_status enum >
  std::map<int, std::pair<int, attr_it>> node_attractions; // < node_lid , < attraction_status, attraction_iterator > >
  std::map<int, std::pair<int, int>> pawnproxy;            // id -> index tuple < type_idx , pawn_idx >
  std::map<int, std::vector<std::string>> bin_route;       // binned routes
  std::vector<std::vector<std::string>> poly2barrier;
  std::vector<std::vector<std::vector<std::vector<polygon>::iterator>>> grid2polygon;
  std::map<std::string, int> type2idx;                     // < pawn type tag, pawn vector index >
  std::map<std::string, pawn_param> pawn_types; // <tag, <alpha_we, beta_bp, height, alpha_s> >
  std::map<std::string, std::map<int, std::vector<std::pair<int,double>>>> cherry_pick_map;
  std::map<std::string, std::vector<std::pair<int, double>>> cherry_pick_sn;
  double route_dist_binw;

  // run
  template<typename cb_1, typename cb_2> void run(cb_1 cb1, cb_2 cb2)
  {
    // start simulation
    for(; sim_time < stop_time; sim_time += dt)
    {
      //std::cout << "Iter : " << iter << " " << sim_time <<  " " << physycom::unix_to_date(sim_time) << std::endl;

      // update midnight time ref to account for multi-day simulation
      if ( sim_time - midn_t_start > 24 * 3600 - 1 )
      {
        auto datenow = physycom::unix_to_date(sim_time, "%Y-%m-%d");
        midn_t_start = int(physycom::date_to_unix(datenow, "%Y-%m-%d"));
        midn_t_stop = midn_t_start + (24 * 3600);
      }
      // evolution loop
      for (auto &type : pawns) for (int i = 0; i < (int)type.size(); ++i) evolve(type[i]); MESSAGE("evolve DONE!\n");
      update_sink();         MESSAGE("update_sink DONE!\n");
      update_sources();      MESSAGE("update_sources DONE!\n");
      update_transports();   MESSAGE("update_transports DONE!\n");
      update_attractions();  MESSAGE("update_attractions DONE!\n");
      update_velocity();     MESSAGE("update_velocity DONE!\n");
      update_weights();      MESSAGE("update_weights DONE!\n");
      update_barriers();     MESSAGE("update_barriers DONE!\n");
      update_attr_weights(); MESSAGE("update_attr_weights DONE!\n");
      update_status();       MESSAGE("update_status DONE!\n");

      // callback
      if ( (sim_time - start_time) % sampling_dt == 0 ) cb1();
      if ( (sim_time - start_time) == dump_state_dt ) cb2();

      ++iter;
    }
  }
};

#endif // _MINIMOCAS_SIMULATION_H
