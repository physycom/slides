#include <algorithm>
#include <numeric>
#include <random>
#include <list>
#include <tuple>

#include <carto.h>
#include <pawn.h>

double dist_none(const double &/* &x */, const std::vector<double> &/* &params */) // comment for 'unused variable' warning
{
  return 1.;
}

double dist_exp(const double &x, const std::vector<double> &params)
{
  return std::exp( - x / params[0] );
}

double dist_logistic(const double &x, const std::vector<double> &params)
{
  return 1. - 1. / ( 1. + std::exp( -params[0] * ( x - params[1]) ));
}

std::map<std::string, dist_type> dist_map({
  {"none",     &dist_none},
  {"exp",      &dist_exp},
  {"logistic", &dist_logistic}
});

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// PAWN
////////////////////////////////////////////////////////////////////////////////////////////////////

int pawn_id = 0;

pawn::pawn()
{
  pick_next = &pawn::pick_next_bp;
}

pawn::pawn(const int &poly_lid, const double &speed_mps, const std::string &tag_, const std::list<int> &dest_, const double &beta_miss)
{
  id = pawn_id++;
  status = PAWN_ACTIVE;
  dest = dest_;
  current_poly = poly_lid;
  current_dest = dest.front();
  dest.pop_front();
  last_node = -1;
  next_node = -1;
  current_s = 0.;
  dlen = 0.;
  speed = speed_mps;
  tag = tag_;
  trip_tstart = -1;
  trip_tstop = -1;
  beta_bp = beta_miss;
  ferrypawn = false;

  totdist = 0.;
  triptime = 0;
  lifetime = 0;

  TTB = -1;

  // select next_poly algo
  pick_next = &pawn::pick_next_bp;
}

pawn::pawn(const int &poly_lid, const double &speed_mps, const std::string &tag_, const std::list<int> &dest_, const double &beta_miss, const std::string &crw_tag_, const std::vector<double> &crw_p) :
  pawn(poly_lid, speed_mps, tag_, dest_, beta_miss)
{
  crw_params = crw_p;
  crw_dist = dist_map.at(crw_tag_);
  crw_tag = crw_tag_;
}

// pick next 2.0
void pawn::pick_next_bp(cart *c, const node_it &node)
{
  //std::cout << "pick_next_bp " << node->lid << std::endl;

  if (std::count_if(c->bpweight[tag][node->lid].begin(),
                    c->bpweight[tag][node->lid].end(),
                    [](const std::pair<int, double> &w) {
                      return w.second == MAX_WEIGHT;
                    }) == int(node->link.size()) - 1)
  {
    for (const auto &l : node->link)
      if(c->bpweight[tag][node->lid][l.first->lid] < MAX_WEIGHT)
        next_node = l.first->lid; // bounce on dead end
  }
  else if (std::count_if(c->bpweight[tag][node->lid].begin(),
                        c->bpweight[tag][node->lid].end(),
                        [](const std::pair<int, double> &w) {
                        return w.second == MAX_WEIGHT;
                        }) == int(node->link.size()))
  {
    status = PAWN_DEAD; //the pawn is trapped
  }
  else
  {
    double miss = rnd_01(generator);
    if (miss >= beta_bp)
    {
      next_node = c->bpmatrix[tag].at(current_dest)[node->lid];
    }
    else
    {
      std::vector<std::tuple<double, poly_it, node_it>> choices;
      for (const auto &p : node->link)
      {
        if (p.second->lid == current_poly) continue; // skip current poly
        if (c->bpweight[tag][node->lid][p.first->lid] == MAX_WEIGHT) continue;
        choices.push_back(std::make_tuple(c->trans.at(current_poly).at(p.second->lid), p.second, p.first));
      }

      double norm = std::accumulate(choices.begin(), choices.end(), 0.0, [](double sum, const std::tuple<double, poly_it, node_it> &p) {
        return sum + std::get<0>(p);
      });

      if (norm < 1.0) for (auto &p : choices) std::get<0>(p) = 1.0 / choices.size();
      else            for (auto &p : choices) std::get<0>(p) /= norm;

      double roll = rnd_01(generator);
      double roll_sum = 0;
      int next = 0;
      while (roll >= roll_sum) {
        roll_sum += std::get<0>(choices[next]); next++;
      }
      next_node = std::get<2>(choices[next - 1])->lid;
    }
  }

  //std::cout << "pick_next_bp out" << std::endl;
}


