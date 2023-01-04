#include <boost/filesystem.hpp>
#include <simulation.h>

////////////////////////////////////
////////////// DUMP UTILITIES
////////////////////////////////////

void simulation::dump_pawn_state()
{
  if (!pawn_out.is_open())
  {
    auto pawn_date = physycom::unix_to_date(start_time, "%y%m%d_%H%M%S");
    pawn_out.open(state_basename + "_pawn_" + pawn_date + ".csv");
    pawn_out << "pawn_id" << separator
      << "tag" << separator
      << "timestamp" << separator
      << "poly_lid" << separator
      << "s"
      << std::endl;
  }

  for (const auto &type : pawns)
    for (const auto &p : type)
      pawn_out << p.id << separator
      << p.tag << separator
      << sim_time << separator
      << p.current_poly << separator
      << p.current_s << std::endl;
}

void simulation::dump_net_state()
{
  if (net_out.is_open())
  {
    net_state = std::vector<int>(c->poly.size(), 0);
    for (const auto &type : pawns)
      for (const auto &p : type)
        ++net_state[p.current_poly];

    auto dump_date = physycom::unix_to_date(sim_time);
    net_out << sim_time;
    for (const auto &p : net_state)
      net_out << separator << p;
    net_out << std::endl;
  }
}

void simulation::dump_state_json()
{
  // dump time tag
  auto dump_time = start_time + dump_state_dt;
  auto dump_date = physycom::unix_to_date(dump_time, "%y%m%d_%H%M%S");

  // dump pawns
  std::ofstream stateofs(state_basename + "_state_" + dump_date + ".csv");
  stateofs << "pawn_id" << separator
    << "tag" << separator
    << "beta_bp" << separator
    << "alpha_we" << separator
    << "status" << separator
    << "current_poly" << separator
    << "current_s" << separator
    << "current_speed" << separator
    << "current_dest" << separator
    << "last_node" << separator
    << "next_node" << separator
    << "tot_dist" << separator
    << "trip_time" << separator
    << "crw_tag" << separator
    << "crw_params" << separator
    << "dest"
    << std::endl;

  for (const auto &type : pawns)
    for (const auto &p : type)
    {
      if (p.status == PAWN_DEAD) continue;
      std::stringstream ss;
      ss << p.id << separator
        << p.tag << separator
        << p.beta_bp << separator
        << pawn_types[p.tag].alpha_we << separator
        << p.status << separator
        << p.current_poly << separator
        << p.current_s << separator
        << p.speed << separator
        << p.current_dest << separator
        << p.last_node << separator
        << p.next_node << separator
        << p.totdist << separator
        << p.triptime << separator
        << p.crw_tag << separator;
      for (const auto &c : p.crw_params) ss << c << " ";
      ss << separator;
      for (const auto &d : p.dest) ss << d << " ";
      stateofs << ss.str() << std::endl;
    }
  stateofs.close();

  // collect and dump environment
  jsoncons::json jstate;
  jstate["start_time"] = sim_time;
  jstate["stop_time"] = stop_time;
  jstate["transport"] = "coming_soon";

  std::set<int> uniq_dest;
  std::transform(c->bpmatrix.begin()->second.begin(), c->bpmatrix.begin()->second.end(), std::inserter(uniq_dest, uniq_dest.end()),
    [](auto pair) {
    return pair.first;
  });
  jstate["uniq_dest"] = uniq_dest;


  std::ofstream jstateofs(state_basename + "_state_" + dump_date + ".json");
  jstateofs << pretty_print(jstate) << std::endl;
  jstateofs.close();

}

void simulation::dump_influxgrid()
{
  if (influxgrid_out.is_open())
  {
    auto grid_state = std::vector<int>(grid.gcol * grid.grow, 0);
    for (const auto &type : pawns)
      for (const auto &p : type)
      {
        auto pt = c->poly[p.current_poly].get_point(p.current_s);
        int row = (pt.ilat - grid.gilatmin) / grid.gdlat;
        int col = (pt.ilon - grid.gilonmin) / grid.gdlon;
        ++grid_state[row * grid.gcol + col];
      }

    int cnt = 0;
    for (const auto &p : grid_state)
      influxgrid_out << influx_meas_name << ","
      << "id=" << cnt++ << " cnt=" << p << " " << (unsigned long long)(sim_time * 1e9) << std::endl;
  }
}

void simulation::dump_barriers()
{
  if (barriers.size())
  {
    if (barrier_outfile == "")
      barrier_outfile = state_basename + "_barriers_" + physycom::unix_to_date(start_time, "%y%m%d_%H%M%S") + ".csv";

    std::ofstream out_cam(barrier_outfile);
    out_cam << "datetime" << separator << "timestamp";
    for (const auto &p : cam_counters.begin()->second)
      out_cam << separator << p.first;
    out_cam << std::endl;

    for (const auto &c : cam_counters) {
      int timestamp = start_time + dump_cam_dt * c.first;
      std::string datetime = physycom::unix_to_date(timestamp);
      out_cam << datetime << separator << timestamp;
      for (const auto &cc : c.second) out_cam << separator << cc.second;
      out_cam << std::endl;
    }
  }
}

void simulation::dump_polygons()
{
  if (polygon_out.is_open())
  {
    // count pawn per polygon
    std::map<std::vector<polygon>::iterator, std::map<std::string, int>> cnt; // < polygon iterator, pawn type string >
    for (auto &type : pawns)
      for (auto &p : type)
      {
        auto pt = c->poly[p.current_poly].get_point(p.current_s);
        int r, c;
        grid.coord_to_grid(pt.ilat, pt.ilon, r, c);
        for (const auto &pol : grid2polygon[r][c])
          if (physycom::point_in_polygon(pol->points, pt))
            ++cnt[pol][p.tag];
      }

    // dump to file
    auto dump_date = physycom::unix_to_date(sim_time);
    for (auto &i : cnt)
    {
      int id = -1;
      auto ppro = i.first->pro;
      if (ppro.find("PK_UID") != ppro.end())   id = std::stoi(ppro["PK_UID"]);
      else if (ppro.find("uid") != ppro.end()) id = std::stoi(ppro["uid"]);

      polygon_out << dump_date << separator << id;
      for (auto &t : pawn_types) polygon_out << separator << i.second[t.first];
      polygon_out << std::endl;
    }
  }
}

void simulation::dump_population()
{
  if (population_out.is_open())
  {
    // count pawn per type
    std::map<std::string, int> cnt; // < pawn type string, counter >
    for (auto &type : pawns)
      for (auto &p : type)
      {
        if (p.status != PAWN_DEAD)
          ++cnt[p.tag];
        if (p.status == PAWN_AWAITING_TRANS)
          ++cnt["awaiting_transport"];
        if (p.status == PAWN_TRANSPORT)
          ++cnt["transport"];
      }

    // dump to file
    auto dump_date = physycom::unix_to_date(sim_time);
    population_out << dump_date << separator << sim_time;
    for (auto &t : pawn_types)
      population_out << separator << cnt[t.first];
    population_out << separator << cnt["transport"];
    population_out << separator << cnt["awaiting_transport"];
    population_out << std::endl;
  }
}

void simulation::dump_geodata()
{
  std::ofstream out(state_basename + "_cartodata.csv");
  char sep = ';';
  out << "tag"
    << sep << "poly_lid"
    << sep << "poly_length[m]"
    << sep << "lat"
    << sep << "lon"
    << sep << "nF_lat"
    << sep << "nF_lon"
    << sep << "nT_lat"
    << sep << "nT_lon"
    << std::endl;
  for (const auto &b : barriers)
  {
    out << b.second.tag
      << sep << b.second.poly->lid
      << sep << b.second.poly->length
      << sep << b.second.loc.ilat * 1e-6
      << sep << b.second.loc.ilon * 1e-6
      << sep << b.second.poly->nF->ilat * 1e-6
      << sep << b.second.poly->nF->ilon * 1e-6
      << sep << b.second.poly->nT->ilat * 1e-6
      << sep << b.second.poly->nT->ilon * 1e-6
      << std::endl;
  }
}

void simulation::dump_poly_uniq()
{
  if (enable_uniq_poly)
  {
    uniq_outfile = state_basename + "_uniq_" + physycom::unix_to_date(start_time, "%y%m%d_%H%M%S") + ".csv";

    std::ofstream out_uniq(uniq_outfile);
    out_uniq << "poly_lid"<<separator << "poly_cid" << separator <<"oneway";
    for (auto &u : c->poly.front().uniq_pwn)
      out_uniq << separator << physycom::unix_to_date(int(u.first * dump_cam_dt + start_time), "%Y-%m-%d %H:%M:%S");
    out_uniq << std::endl;

    for (const auto &poly : c->poly)
    {
      out_uniq << poly.lid << separator << poly.cid <<separator<<"FT";
      for (auto &uv : poly.uniq_pwn)
        out_uniq << separator << uv.second.first.size();
      out_uniq << std::endl;
    }

    for (const auto &poly : c->poly)
    {
      out_uniq << poly.lid << separator << poly.cid << separator << "TF";
      for (auto &uv : poly.uniq_pwn)
        out_uniq << separator << uv.second.second.size();
      out_uniq << std::endl;
    }

    out_uniq.close();
  }
}
