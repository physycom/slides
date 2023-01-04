/*!
 *  \file   pawn.h
 *  \author A. Fabbri (alessandro.fabbri27@unibo.it), C. Mizzi (chiara.mizzi2@unibo.it), S. Sinigardi (stefano.sinigardi@unibo.it)
 *  \brief  Declaration file for #pawn object.
 *  \details This file contains the definitions of #pawn class together with various support objects.
 */

#ifndef _MINIMOCAS_PAWN_HPP
#define _MINIMOCAS_PAWN_HPP

#include <algorithm>
#include <numeric>
#include <random>
#include <list>
#include <tuple>

#include <carto.h>

enum
{
  NODE_NORMAL = 0,
  NODE_ATTRACTION
};

constexpr double pawn_vmin = 0.5;
constexpr double pawn_vmax = 1.5;

constexpr int default_dest_node = 4944;

// RNG
static std::mt19937 generator;
static std::uniform_real_distribution<double> rnd_01;
static std::uniform_int_distribution<int> rnd_node;
static std::uniform_int_distribution<int> rnd_int;
static std::exponential_distribution<double> rnd_exp;
static std::uniform_int_distribution<int> rnd_ttb;

typedef double (*dist_type)(const double &x, const std::vector<double> &params); // move inside class

enum
{
  PAWN_ACTIVE         = 0,
  PAWN_DEAD           = 1,
  PAWN_AWAITING       = 2,
  PAWN_QUEUED         = 3,
  PAWN_VISITING       = 4,
  PAWN_AWAITING_TRANS = 5,
  PAWN_TRANSPORT      = 6,
};

//enum {
//  PAWN = 0,
//  BOAT_1 = 1,
//  BOAT_2 = 2,
//  BOAT_3 = 3
//};


inline std::string get_pawn_status(const int &status)
{
  std::string s;
  switch(status)
  {
    case PAWN_ACTIVE:         s = "active";               break;
    case PAWN_AWAITING:       s = "awaiting";             break;
    case PAWN_DEAD:           s = "dead";                 break;
    case PAWN_QUEUED:         s = "queued";               break;
    case PAWN_VISITING:       s = "visiting";             break;
    case PAWN_AWAITING_TRANS: s = "awaiting_trans";       break;
    case PAWN_TRANSPORT:      s = "transport";            break;
    default:                  s = std::to_string(status); break;
  }
  return s;
}

struct pawn
{
  int id;                                    // unique progressive identifier
  int status;                                // mobility status
  int current_poly;                          // [lid]
  int current_dest, next_node, last_node;    // [lid]
  std::list<int> dest;                       // [lid]
  double current_s;                          // [m]
  double dlen;                               // [m]
  double speed;                              // [m/s]
  std::string tag;
  int trip_tstart, trip_tstop, idletime;     // [s]
  double beta_bp;                            // prob to miss bestpath
  bool ferrypawn;                            // enables interaction with transport and wroute mechanism
  int transport_time;

  double totdist, dist_thresh;
  int triptime, lifetime;
  double TLB; // Total Length Budget
  int TTB; // Total Time Budget

  std::string route_id;

  // CRW vars
  dist_type crw_dist;
  std::vector<double> crw_params;
  std::string crw_tag;

  // internal pointer
  void (pawn::*pick_next)(cart *c, const node_it &node);

  // constructors
  pawn();
  pawn(const int &poly_lid, const double &speed_mps, const std::string &tag_, const std::list<int> &dest_, const double &beta_miss);
  pawn(const int &poly_lid, const double &speed_mps, const std::string &tag_, const std::list<int> &dest_, const double &beta_miss, const std::string &dists, const std::vector<double> &crw_params);

  // evolution methods
  void pick_next_bp(cart *c, const node_it &node);
};

#endif // _MINIMOCAS_PAWN_HPP
