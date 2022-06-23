#include <set>
#include <string>
#include <ctime>

#include <jsoncons/json.hpp>

#include <physycom/time.hpp>
#include <physycom/combinatoric.hpp>
#include <physycom/histo.hpp>

#include <simulation.h>
#include <connection.h>

extern std::map<std::string, dist_type> dist_map;
extern int pawn_id;

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// ATTRACTION
////////////////////////////////////////////////////////////////////////////////////////////////////

attraction::attraction() {}
attraction::attraction(const std::string &tag, const int &nlid, const std::vector<int> &timecap, const int &visit_time)
  : node_lid(nlid), visitors(0), visit_time(visit_time), rate_in(0), tag(tag), timecap(timecap)
{
  rnd_vis_time = std::normal_distribution<double>(visit_time, 0.05*visit_time);
}

inline std::vector<attraction> init_attraction(const jsoncons::json &jconf, cart *c)
{
  std::vector<attraction> attractions;
  if (jconf.has_member("attractions"))
  {
    attractions.reserve(jconf["attractions"].size());
    for (const auto &s : jconf["attractions"].object_range())
    {
      if (s.value().has_member("node_lid"))
      {
        attractions.emplace_back(
          std::string(s.key()),
          s.value()["node_lid"].as<int>(),
          s.value()["timecap"].as<std::vector<int>>(),
          s.value()["visit_time"].as<int>()
        );
      }
      else
      {
        auto n = c->get_nearest_node(point_base(int(s.value()["lat"].as<double>() * 1e6), int(s.value()["lon"].as<double>() * 1e6)));
        if (n == c->node.end())
        {
          std::cout << "[init_attractions] Skipping attraction " << s.key() << " unable to geofence nearest node." << std::endl;
          continue;
        }
        attractions.emplace_back(
          std::string(s.key()),
          n->lid,
          s.value()["timecap"].as<std::vector<int>>(),
          s.value()["visit_time"].as<int>()
        );
      }

      if (s.value().has_member("weight"))
      {
        try
        {
          attractions.back().weight = s.value()["weight"].as<std::vector<double>>();
        }
        catch (...)
        {
          double weight_temp = s.value()["weight"].as<double>();
          for (auto &tc : attractions.back().timecap) {
            if (tc == 0)
              attractions.back().weight.push_back(0.0);
            else
              attractions.back().weight.push_back(weight_temp);
          }
        }

        double weight_sum = std::accumulate(attractions.back().weight.begin(), attractions.back().weight.end(), 0.0);
        if (weight_sum == 0.0) attractions.pop_back();
      }
    }
  }
  return attractions;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// TRANSPORT
////////////////////////////////////////////////////////////////////////////////////////////////////

transport::transport() {}
std::vector<point_base> excluded; //temp added
std::map<int, std::map<int, std::map<int, std::vector<int>>>> transp_proxy; //<timestamp, <start_stop,<stop_stop,vector<index in transports>>>>
int slice_transp;
std::string transport_info;

inline std::vector<transport> init_transports(const jsoncons::json &jconf, cart *c)
{
  std::vector<transport> transp;
  std::vector<std::vector<std::string>> transp_data;

  /********** REWORK IN THIS WAY TO ENABLE FOR DIFFERENT SOURCES OF DATA ************

  // store data from file
  if (jconf.has_member("transport_file"))
  {
    std::ifstream tf(jconf["transport_file"].as<std::string>());
    if (!tf)
    {
      std::cout << "init_transports() : Transport timetable file " << jconf["transport_file"].as<std::string>() << " not found. Skipping...";;
    }
    else
    {
      std::string line;
      std::string sep = " ";
      std::vector<std::string> tok;
      std::getline(tf, line); // skip header
      while (std::getline(tf, line))
      {
        physycom::split(tok, line, sep, physycom::token_compress_on);
        transp_data.push_back(tok);
      }
    }
  }

  // store data from url
  if (jconf.has_member("transport_url"))
  {
    std::string url = jconf["transport_url"].as<std::string>();
    try
    {
      auto res = curlpp_get(url);
      for (const auto &line : jsoncons::json::parse(res).as<std::vector<std::vector<std::string>>>())
        transp_data.push_back(line);
    }
    catch (std::exception &e)
    {
      std::cout << "init_transports() EXC transport_url : " << e.what() << std::endl;
    }
  }

  // create objects from data
  for (const auto tok : transp_data)
  {
    transport t;
    t.name = tok[0];
    // add logic here
    // ...
    transp.push_back(t);
  }
  */

  std::map<int, node_it> stops_collector;

  if (jconf.has_member("transports"))
  {
    std::stringstream transport_ss; // for logging
    transport_ss << "(";
    std::string stops_filename, trips_filename;
    try
    {
      stops_filename = jconf["transports"]["stops"].as<std::string>();
      trips_filename = jconf["transports"]["trips"].as<std::string>();
    }
    catch (std::exception &e)
    {
      throw std::runtime_error("[init_transports] Error in transport subjson : " + std::string(e.what()));
    }

    std::ifstream sf(stops_filename);
    if (!sf) throw std::runtime_error("[init_transports] Unable to open stops file " + stops_filename);

    std::string line;
    std::getline(sf, line); //skip header

    std::vector<std::string> tok;
    std::string sep = ";";
    while (std::getline(sf, line))
    {
      physycom::split(tok, line, sep, physycom::token_compress_off);

      point_base pw;
      pw.ilat = int(stod(tok[4])*1e6);
      pw.ilon = int(stod(tok[5])*1e6);

      int row, col;
      c->grid.coord_to_grid(pw.ilat, pw.ilon, row, col);
      double d, dmin = std::numeric_limits<double>::max();
      node_it nearest = c->node.end();
      for (auto nn = c->grid.grid[row][col].node.begin(); nn != c->grid.grid[row][col].node.end(); ++nn)
      {
        d = distance(pw, **nn);
        if (d < dmin)
          for (auto i : (*nn)->link)
            if (i.second->oneway != ONEWAY_CLOSED) {
              dmin = d;
              nearest = *nn;
              break;
            }
      }

      if (nearest != c->node.end())
      {
        stops_collector[stoi(tok[0])] = nearest;
        nearest->stops = true;
      }
      else
        excluded.push_back(pw);

    }
    transport_ss << "stops " << stops_collector.size() << " excluded " << excluded.size();
    sf.close();

    std::ifstream tf(trips_filename);
    if (!tf) throw std::runtime_error("[init_transports] Unable to open trips file" + trips_filename);

    std::getline(tf, line);

    sep = ";";
    while (std::getline(tf, line))
    {
      transport t;
      physycom::split(tok, line, sep, physycom::token_compress_on);
      t.trip_id = stoi(tok[1]);
      t.route_id = stoi(tok[2]);

      int st_start, ts_start;
      int st_end, ts_end;
      for (int i = 3; i < int(tok.size()) - 4; i += 2)
      {
        st_start = std::stoi(tok[i]);
        ts_start = std::stoi(tok[i + 1]);
        st_end = std::stoi(tok[i + 2]);
        ts_end = std::stoi(tok[i + 3]);

        if (stops_collector.find(st_start) != stops_collector.end() && stops_collector.find(st_end) != stops_collector.end())
        {
          t.tt[stops_collector[st_start]->lid][stops_collector[st_end]->lid] = std::make_pair(ts_start, ts_end);
          t.stops.emplace_back(stops_collector[st_start]->lid);
        }
        if (stops_collector.find(st_start) != stops_collector.end() && stops_collector.find(st_end) == stops_collector.end())
        {
          while (stops_collector.find(std::stoi(tok[i + 2])) == stops_collector.end() && i + 2 < int(tok.size() - 3)) i += 2;
          if (i + 2 >= int(tok.size() - 3))
          {
            t.stops.emplace_back(stops_collector[st_start]->lid);
            break;
          }
          else
          {
            st_end = std::stoi(tok[i + 2]);
            ts_end = std::stoi(tok[i + 3]);
            t.tt[stops_collector[st_start]->lid][stops_collector[st_end]->lid] = std::make_pair(ts_start, ts_end);
            t.stops.emplace_back(stops_collector[st_start]->lid);
          }
        }
      }
      if (stops_collector.find(std::stoi(tok[tok.size() - 2])) != stops_collector.end())
        t.stops.emplace_back(std::stoi(tok[tok.size() - 2]));

      for (int d1 = 0; d1 < (int)t.stops.size() - 1; ++d1)
        for (int d2 = d1 + 2; d2 < (int)t.stops.size(); ++d2)
          t.tt[t.stops[d1]][t.stops[d2]] = std::make_pair(t.tt[t.stops[d1]][t.stops[d2 - 1]].first, t.tt[t.stops[d2 - 1]][t.stops[d2]].second);

      if (t.stops.size() > 1)
        transp.push_back(t);
    }

    // make transport proxy list
    if (jconf["transports"].has_member("slice_transp_dt"))
    {
      slice_transp = jconf["transports"]["slice_transp_dt"].as<int>();
    }
    else
    {
      slice_transp = DEFAULT_SLICE_TRANSP;
    }
    int start_time = (int)physycom::date_to_unix(jconf["start_date"].as<std::string>());
    int stop_time = (int)physycom::date_to_unix(jconf["stop_date"].as<std::string>());
    for (int t = 0; t < int(transp.size()); t++)
    {
      for (const auto &s : transp[t].tt)
      {
        for (const auto &v : s.second)
        {
          if (v.second.first > start_time && v.second.first < stop_time)
          {
            int idx = int((v.second.first - start_time) / slice_transp);
            transp_proxy[idx][s.first][v.first].push_back(t);
          }
        }
      }
    }
    transport_ss << " proxy " << transp_proxy.size();

    // collect debug info
    transport_ss << ")";
    transport_info = transport_ss.str();
  }

  /*
  // check transport_proxy
  for(auto i : transp_proxy)
  {
    for(auto j : i.second)
    {
      for (auto k : j.second)
      {
        std::cout << " time idx " << i.first << " n_start " << j.first << " n_stop " << k.first << " ts";
        for( auto l : k.second)
        {
          std::cout << " " << l;
        }
        std::cout << std::endl;
      }
    }
  }
  */

  return transp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// SINK
////////////////////////////////////////////////////////////////////////////////////////////////////

sink::sink() {}

sink::sink(const std::string &tag, const std::vector<int> &ddt, const int &nlid) : node_lid(nlid), tag(tag), despawn_dt(ddt) {}

inline std::vector<sink> init_sinks(const jsoncons::json &jconf)
{
  std::vector<sink> sinks;

  if (jconf.has_member("sinks"))
  {
    for (const auto &s : jconf["sinks"].object_range())
      sinks.emplace_back(std::string(s.key()),
        s.value()["despawn_timetable"].as<std::vector<int>>(),
        s.value()["node_lid"].as<int>());
  }

  return sinks;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// SOURCE
////////////////////////////////////////////////////////////////////////////////////////////////////

source::source()
{
  is_localized = false;
  is_geofenced = false;
  is_control = false;
}

source::source(const std::string &tag_, const jsoncons::json &jconf, cart* c) :
source()
{
  tag = tag_;
  creation_dt = jconf.has_member("creation_dt") ? jconf["creation_dt"].as<int>() : 300;
  creation_rate = jconf.has_member("creation_rate") ? jconf["creation_rate"].as<std::vector<int>>() : std::vector<int>({ 1 });
  std::string ctlr_type = jconf.has_member("source_type") ? jconf["source_type"].as<std::string>() : std::string("std");
  if (ctlr_type == std::string("std"))
  {
    source_type = SOURCE_STD;
    is_control = false;
  }
  else if (ctlr_type == std::string("ctrl"))
  {
    source_type = SOURCE_CTRL;
    is_control = true;
  }
  else if (ctlr_type == std::string("tot"))
  {
    source_type = SOURCE_TOT;
    is_control = true;
  }
  else
  {
    source_type = SOURCE_STD;
    is_control = false;
  }

  // edit creation_rate
  if (!is_control)
  {
    int datacamsec = 24 * 3600 / int(creation_rate.size());
    int nbin = datacamsec / creation_dt;
    double rest = 0.0;
    double fractional, whole;
    std::vector<int> creation_rate_dt;
    for (const auto &cnt : creation_rate)
    {
      for (int i = 0; i < nbin; ++i)
      {
        fractional = std::modf(double(cnt) / nbin, &whole);
        rest += fractional;
        if (rest > 1.0) {
          whole += 1;
          rest -= 1.0;
        }
        creation_rate_dt.push_back(int(whole));
      }
    }
    creation_rate = creation_rate_dt;
  }

  // inherit pawn json to reuse init_pawn
  if (jconf.has_member("pawns"))
  {
    pawns_spec["pawns"] = jconf["pawns"];
    for (auto &s : pawns_spec["pawns"].object_range())
      s.value()["number"] = creation_rate;
  }

  if (jconf.has_member("dest_location")) {
    auto loc = jconf["dest_location"];
    double lat = loc["lat"].as<double>();
    double lon = loc["lon"].as<double>();
    auto p = point_base(int(lat * 1e6), int(lon * 1e6));

    point_proj pp = c->project(p, c->get_nearest_arc(p));

    if (pp.a != c->arc.end())
    {
      auto poly = pp.a->p;
      double s = pp.a->s + pp.s;
      auto node = s < poly->length - s ? poly->nF : poly->nT;
      node_lid_dest = node->lid;

      if (jconf.has_member("pawns"))
      {
        for (auto &sp : pawns_spec["pawns"].object_range())
          sp.value()["dest"] = node_lid_dest;
      }
      else if (jconf.has_member("pawns_from_weight"))
      {
        pawns_spec["pawns_from_weight"] = jconf["pawns_from_weight"];
        for (auto &sp : pawns_spec["pawns_from_weight"].object_range())
          sp.value()["number"] = creation_rate;
      }
    }
    else
    {
      std::cout << "[init_sources] WARNING Dest '" << tag << "' has no close POLY." << std::endl;
    }
  }

  if (jconf.has_member("source_location"))
  {
    is_localized = true;
    auto loc = jconf["source_location"];
    lat = loc["lat"].as<double>();
    lon = loc["lon"].as<double>();
    auto p = point_base(int(lat * 1e6), int(lon * 1e6));
    this->loc = p;

    is_geofenced = true;
    point_proj pp = c->project(p, c->get_nearest_arc(p));

    if (pp.a != c->arc.end())
    {
      poly = pp.a->p;
      this->s = pp.a->s + pp.s;
      node = this->s < poly->length - this->s ? poly->nF : poly->nT;
      node_lid = node->lid;

      if (tag.find("_IN") != std::string::npos)
        orientation = FT;
      else if (tag.find("_OUT") != std::string::npos)
        orientation = TF;
      else
      {
        std::cout << "[source] WARNING Source " << tag << " is not of type IN/OUT. Setting orientation to FT." << std::endl;
        orientation = FT;
      }

      if (jconf.has_member("pawns"))
      {
        for (auto &sp : pawns_spec["pawns"].object_range())
          sp.value()["start_node_lid"] = node_lid;
      }
      else if (jconf.has_member("pawns_from_weight"))
      {
        pawns_spec["pawns_from_weight"] = jconf["pawns_from_weight"];
        for (auto &sp : pawns_spec["pawns_from_weight"].object_range())
          sp.value()["number"] = creation_rate;
      }
    }
    else
    {
      std::cout << "[init_sources] WARNING Source '" << tag << "' has no close POLY." << std::endl;
      is_geofenced = false;
    }
  }

}

inline std::vector<source> init_sources(const jsoncons::json &conf, cart* c)
{
  std::vector<source> sources;

  if (conf.has_member("sources"))
  {
    for (const auto &s : conf["sources"].object_range())
    {
      sources.emplace_back(std::string(s.key()), s.value(), c);
    }
  }

  std::set<int> sources_node;
  for (const auto &s : sources)
    if (s.is_geofenced)
      sources_node.insert(s.node_lid);
  c->add_bpmatrix(sources_node);

  return sources;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// WEIGHTED ROUTE
////////////////////////////////////////////////////////////////////////////////////////////////////

wroute::wroute() : weight(1.0), len(0.0), tag(""), node_seq(0) {}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// BARRIER
////////////////////////////////////////////////////////////////////////////////////////////////////

barrier::barrier() : cnt_TF(0), cnt_FT(0), tag(""), geofenced(false) {}


void simulation::init_barriers(const jsoncons::json &jconf, cart* c)
{
  if (jconf.has_member("file_barrier"))
  {
    std::ifstream b_file(jconf["file_barrier"].as<std::string>());
    if (!b_file)
      throw std::runtime_error("Barrier file not found.");

    std::vector<std::string> b_list;
    if (jconf.has_member("barrier_list"))
    {
      std::set<std::string> b_set;
      for (const auto &s : jconf["barrier_list"].as<std::vector<std::string>>())
      {
        b_set.insert(s.substr(0, s.find_last_of("_")));
      }
      b_list = std::vector<std::string>(b_set.begin(), b_set.end());
    }
    std::string line;
    std::vector<std::string> tok;
    std::string separator = ";";
    std::getline(b_file, line); //skip header
    while (std::getline(b_file, line))
    {
      physycom::split(tok, line, separator, physycom::token_compress_off);

      barrier b;
      b.tag = tok[0];
      b.tag = b.tag.substr(0, b.tag.find_last_of("_"));
      b.geofenced = true;

      if (b_list.size() && std::find(b_list.begin(), b_list.end(), b.tag) == b_list.end()) continue;  // barrier filtered by config file
      if (barriers.find(b.tag) != barriers.end()) continue;                          // barrier already present, possibile duplicate direction

      point_base p(int(stod(tok[2])*1e6), int(stod(tok[3])*1e6));
      b.loc = p;

      point_proj pp = c->project(p, c->get_nearest_arc(p));

      if (pp.a == c->arc.end())
      {
        b.geofenced = false;
        std::cout << "[simulation] Fail to associate POLY to barrier " << b.tag << std::endl;
        continue;
      }
      else
      {
        b.poly = pp.a->p;
        b.s = pp.a->s + pp.s;
        b.node = b.s < b.poly->length - b.s ? b.poly->nF : b.poly->nT;
      }

      barriers[b.tag] = b;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// POLYGON
////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<std::vector<std::vector<polygon>::iterator>>> grid2polygon;
std::map<int, int> pg_id2idx;
std::string polygons_info;

polygon::polygon() {}

polygon::polygon(const jsoncons::json &polygonlist)
{
  for (const auto &pt : polygonlist[0].array_range())
  {
    points.emplace_back(int(pt[1].as<double>() * 1e6), int(pt[0].as<double>() * 1e6));
  }
}

inline std::vector<polygon> init_polygons(const jsoncons::json &jconf, cart *c)
{
  std::vector<polygon> polygons;
  int polygon_cnt = 0;
  if (jconf.has_member("file_polygons"))
  {
    std::ifstream polyifs(jconf["file_polygons"].as<std::string>());
    if (!polyifs)
      throw std::runtime_error("unable to open polygons file " + jconf["file_polygons"].as<std::string>());

    auto jpoly = jsoncons::json::parse(polyifs);
    for (const auto &feature : jpoly["features"].array_range())
    {
      if (!feature["geometry"]["coordinates"].size()) continue;
      auto type = feature["geometry"]["type"].as<std::string>();

      if (type == "MultiPolygon")
      {
        //std::cout << "feature type MultiPolygon size " << feature["geometry"]["coordinates"].size() << std::endl;
        for (const auto &pol : feature["geometry"]["coordinates"].array_range())
        {
          ++polygon_cnt;
          polygon pw = polygon(pol);
          for (const auto &pro : feature["properties"].object_range())
            pw.pro[std::string(pro.key())] = pro.value().as<std::string>();
          pw.centroid = physycom::centroid(pw.points);
          if (!c->is_in_bbox(pw.centroid)) continue; // skip if out of cart roi
          polygons.push_back(pw);
        }
      }
      else if (type == "Polygon")
      {
        //std::cout << "feature type Polygon size " << feature["geometry"]["coordinates"].size() << std::endl;
        ++polygon_cnt;
        auto pol = feature["geometry"]["coordinates"];
        polygon pw = polygon(pol);
        for (const auto &pro : feature["properties"].object_range())
          pw.pro[std::string(pro.key())] = pro.value().as<std::string>();
        pw.centroid = physycom::centroid(pw.points);
        if (!c->is_in_bbox(pw.centroid)) continue; // skip if out of cart roi
        polygons.push_back(pw);
      }
      else
      {
        std::cerr << "GEOJSON feature type " << type << " unsupported" << std::endl;
        return polygons;
      }
    }
  }

  for (auto &pg : polygons)
  {
    //select set of poly closest
    std::set<poly_it> poly_closest;
    for (const auto &cord : pg.points)
    {
      int row, col;
      c->grid.coord_to_grid(cord.ilat, cord.ilon, row, col);
      for (const auto &a : c->grid.grid[row][col].arc)
        poly_closest.insert(a->p);
    }

    // for each poly find intersection
    for (const auto &poly : poly_closest)
    {
      int n_intersect = 0;
      for (int bc = 0; bc < int(pg.points.size() - 1); ++bc)
      {
        for (int i = 0; i < int(poly->point.size() - 1); ++i)
        {
          n_intersect += c->find_intersection(*poly->point[i], *poly->point[i + 1], pg.points[bc], pg.points[bc + 1]);
        }
      }
      if (n_intersect > 0) pg.poly_in.push_back(poly);
    }
  }

  for (int i = 0; i < int(polygons.size()); ++i)
  {
    auto ppro = polygons[i].pro;
    if (ppro.find("PK_UID") != ppro.end()) {
      pg_id2idx[std::stoi(ppro.at("PK_UID"))] = i;
    }
    else if (ppro.find("uid") != ppro.end()) {
      pg_id2idx[std::stoi(ppro.at("uid"))] = i;
    }
    else
      throw std::runtime_error("unable to find polygons ID in geojson");
  }

  std::stringstream ss;
  ss << "(parsed " << polygon_cnt << ")";
  polygons_info = ss.str();

  return polygons;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// SIMULATION
////////////////////////////////////////////////////////////////////////////////////////////////////

simulation::simulation() {}

simulation::simulation(const jsoncons::json &jconf, cart *c) :
  iter(0), c(c),
  dynw_time_idx(-1),
  dynw_time_idx2(-1),
  separator(';'),
  polygon_outfile(""),
  poly_outfile("")
{
  generator.seed(1);
  rnd_node = std::uniform_int_distribution<int>(0, int(c->node.size()) - 1);
  rnd_01 = std::uniform_real_distribution<double>(0., 1.);
  auto lambda = jconf.has_member("rnd_exp_mean") ? jconf["rnd_exp_mean"].as<double>() : 1. / 5000.;
  rnd_exp = std::exponential_distribution<double>(1. / lambda);
  rnd_ttb = std::uniform_int_distribution<int>(0, 3600);

  start_time = jconf.has_member("start_time") ? jconf["start_time"].as<int>() : 1522850400;
  stop_time = jconf.has_member("stop_time") ? jconf["stop_time"].as<int>() : 1522854000;

  if (jconf.has_member("start_date"))
  {
    start_date = jconf["start_date"].as<std::string>();
    start_time = (int)physycom::date_to_unix(start_date);
  }
  else
  {
    start_date = physycom::unix_to_date(start_time);
  }

  std::string midn_start_date = start_date.substr(0, 11) + "00:00:00";
  midn_t_start = (int)physycom::date_to_unix(midn_start_date);
  midn_t_stop = midn_t_start + (24 * 3600);

  if (jconf.has_member("stop_date"))
  {
    stop_date = jconf["stop_date"].as<std::string>();
    stop_time = (int)physycom::date_to_unix(stop_date);
  }
  else
  {
    stop_date = physycom::unix_to_date(stop_time);
  }
  sim_time = start_time;

  dump_cam_dt = jconf.has_member("cam_dump_dt") ? jconf["cam_dump_dt"].as<int>() : 3600;
  dt = jconf.has_member("dt") ? jconf["dt"].as<int>() : 5;
  sampling_dt = jconf.has_member("sampling_dt") ? jconf["sampling_dt"].as<int>() : 300;
  if (sampling_dt < dt) sampling_dt = dt;
  if (sampling_dt % dt != 0)
  {
    sampling_dt = dt * int(sampling_dt / dt);
    std::cerr << "WARNING: setting sampling_dt to " << sampling_dt << std::endl;
  }
  dump_state_dt = jconf.has_member("dump_state_dt") ? jconf["dump_state_dt"].as<int>() : -1; // enforces no output, see cb2 in run loop

  state_basename = jconf.has_member("state_basename") ? jconf["state_basename"].as<std::string>() : std::string("sim");

  auto cell_side = jconf.has_member("state_grid_cell_m") ? jconf["state_grid_cell_m"].as<double>() : 25.0;
  grid = grid_base(c->ilat_max, c->ilat_min, c->ilon_max, c->ilon_min, cell_side, c->node, c->arc);
  if (jconf.has_member("enable_grid_geojson") && jconf["enable_grid_geojson"].as<bool>())
    grid.dump_geojson("grid_tiles.geojson");

  influx_meas_name = jconf.has_member("influx_meas_name") ? jconf["influx_meas_name"].as<std::string>() : std::string("gridstate");

  if (jconf.has_member("barrier_outfile"))
    barrier_outfile = jconf["barrier_outfile"].as<std::string>();
  else
    barrier_outfile = "";

  enable_uniq_poly = jconf.has_member("enable_uniq_poly") ? jconf["enable_uniq_poly"].as<bool>() : false;

  enable_status = jconf.has_member("enable_status") ? jconf["enable_status"].as<bool>() : false;
  if (enable_status)
  {
    niter = (stop_time - start_time) / dt;
    status_out.open(state_basename + ".status");
    status_out << "index"
            << separator << "niter"
            << separator << "datetime"
            << std::endl;
  }

  enable_stats = jconf.has_member("enable_stats") ? jconf["enable_stats"].as<bool>() : false;
  if (enable_stats)
  {
    auto stats_date = physycom::unix_to_date(start_time, "%y%m%d_%H%M%S");
    pawnstats_out.open(state_basename + "_pstats_" + stats_date + ".csv");
    pawnstats_out << "id"
      << separator << "tag"
      << separator << "totdist"
      << separator << "triptime"
      << separator << "lifetime"
      << separator << "event_time"
      << separator << "event_misc"
      << separator << "event_type"
      << std::endl;

    wrstats_out.open(state_basename + "_wrstats_" + stats_date + ".csv");
    wrstats_out << "source"
      << separator << "wr_tag"
      << separator << "length"
      << separator << "weight"
      << separator << "event_type"
      << std::endl;
  }

  // dynamic weights mode selection
  dynw_mode = DYNW_MODE_OFF;
  if (jconf.has_member("dynamic_weights"))
  {
    auto mode = jconf["dynamic_weights"]["mode"].as<std::string>();
    if (mode == "off")
      dynw_mode = DYNW_MODE_OFF;
    else if (mode == "lvlps_hc")
    {
      dynw_mode = DYNW_MODE_LVLPS_HC;
      lvlps_tt = jconf["dynamic_weights"]["timetable"].as<std::vector<int>>();
    }
    else if (mode == "bridge_lvl")
    {
      dynw_mode = DYNW_MODE_BRIDGE_LVL;
      lvlps_tt = jconf["dynamic_weights"]["timetable"].as<std::vector<int>>();
    }
    else if (mode == "polygon_closed")
    {
      dynw_mode = DYNW_MODE_PG_CLOSED;
      pg_closed = jconf["dynamic_weights"]["timetable"].as<std::vector<std::vector<int>>>();
      //std::cout << "pg " << pg_closed.size() << std::endl; for (auto &p : pg_closed) { for (auto &i : p) std::cout << i << " "; std::cout << std::endl; }
    }
    else if (mode == "hybrid") {
      dynw_mode = DYNW_MODE_HYBRID;
      pg_closed = jconf["dynamic_weights"]["timetable_pgc"].as<std::vector<std::vector<int>>>();
      lvlps_tt = jconf["dynamic_weights"]["timetable_wl"].as<std::vector<int>>();
    }
  }

  // wroute params
  L_max = 15.;
  N_bin = 10;
  L_binw = L_max / N_bin;
  a = 7.5;
  b = 0.84;
  Lc = 0.5;
  auto dmdist = [this](const double &x) {
    return b * std::exp(-b * x) * std::pow((1 + std::exp(a * Lc)) / (1 + std::exp(-a * (x - Lc))), b / a + 1) / (1 + std::exp(a * Lc));
  };
  std::vector<double> probs;
  for (int i = 0; i < N_bin; ++i)
    probs.push_back(dmdist(L_binw * i));
  rnd_bmdist = std::discrete_distribution<int>(probs.begin(), probs.end());

  // init pawns from file
  if (jconf.has_member("file_init_state"))
  {
    init_from_file(jconf);
  }
  // init pawns from config
  auto ptemp = make_pawns(jconf);
  if (ptemp.size()) pawns.insert(pawns.end(), ptemp.begin(), ptemp.end());

  /*
  if (jconf.has_member("pawns_from_route"))
  {
    auto ptemp = make_pawns_route(jconf);
    if (ptemp.size()) pawns.insert(pawns.end(), ptemp.begin(), ptemp.end());
  }
  */

  // init sources, sinks and transports
  sources = init_sources(jconf, c);

  // collect global type map
  for (const auto &type : pawns)
  {
    if (type.size())
    {
      if (pawn_types.find(type.front().tag) == pawn_types.end()) {
        pawn_param pp(default_alpha_we, default_beta_bp_miss, default_height, default_alpha_speed);
        pawn_types[type.front().tag] = pp;
      }
    }
  }

  // init cherry_pick_map
  for (auto &n0 : c->node)
    for (auto &n1 : c->node)
      if (c->bpmatrix["locals"].at(n0.lid).at(n1.lid) != NO_ROUTE && n0.lid != n1.lid)
        cherry_pick_map["all"][n0.lid].push_back(std::make_pair(n1.lid, 1.0));

  if (jconf.has_member("cherry_pick_mode")) {

    std::map<std::string, std::vector<int>> city2nodelid;
    for (const auto &n : c->node)
      city2nodelid[n.city_membership].push_back(n.lid);

    for (const auto &p : jconf["cherry_pick_mode"].object_range()) {
      std::string name = p.name();
      if (p.value() == "locals")
      {
        for (const auto &l0 : city2nodelid[name])
          for (const auto &l1 : city2nodelid[name])
            if (c->bpmatrix["locals"].at(l0).at(l1) != NO_ROUTE && l0 != l1)
              cherry_pick_map[name][l0].push_back(std::make_pair(l1, c->node[l1].score));
      }
      else
      {
        double radius_meters = p.value().as<double>()*1000.;
        for (const auto &nw : c->node)
        {
          int row, col;
          c->grid.coord_to_grid(nw.ilat, nw.ilon, row, col);
          int delta_cell = (int)(radius_meters / c->grid.gside + 1);

          int row_min = row - delta_cell; if (row_min < 0) row_min = 0;
          int row_max = row + delta_cell; if (row_max > c->grid.grow) row_max = c->grid.grow;
          int col_min = col - delta_cell; if (col_min < 0) col_min = 0;
          int col_max = col + delta_cell; if (col_max > c->grid.gcol) col_max = c->grid.gcol;


          for (int rw = row_min; rw < row_max; ++rw)
            for (int cl = col_min; cl < col_max; ++cl)
              for (const auto &ng : c->grid.grid[rw][cl].node)
              {
                double dist = distance(*ng, nw);
                if (dist <= radius_meters && c->bpmatrix["locals"].at(nw.lid).at(ng->lid) != NO_ROUTE && nw.lid != ng->lid)
                  cherry_pick_map[name][nw.lid].push_back(std::make_pair(ng->lid, ng->score));
              }
        }
      }
    }
  }

  // normalize and sort score
  for (auto &cp : cherry_pick_map) {
    for (auto &n : cp.second) {
      double total_score = 0.0;
      for (const auto &s : n.second) total_score += s.second;
      for (auto &s : n.second) s.second /= total_score;
      std::sort(n.second.begin(), n.second.end(), [](auto &left, auto &right) {
        return left.second < right.second;
      });
      for (int m = 1; m < int(n.second.size()); ++m)
        n.second[m].second += n.second[m - 1].second;
    }
  }

  //init cherry_pick_sn
  for (const auto &cp : cherry_pick_map) {
    for (const auto &cpn : cp.second)
      cherry_pick_sn[cp.first].push_back(std::make_pair(cpn.first, c->node[cpn.first].score));
  }

  // normalize and sort score
  for (auto &cps : cherry_pick_sn) {
    double total_score = 0.0;
    for (auto &sn : cps.second) total_score += sn.second;
    for (auto &sn : cps.second) sn.second /= total_score;
    std::sort(cps.second.begin(), cps.second.end(), [](auto &left, auto &right) {
      return left.second < right.second;
    });
    for (int m = 1; m < int(cps.second.size()); ++m)
      cps.second[m].second += cps.second[m - 1].second;
  }


  for (const auto &src : sources)
  {
    if (src.pawns_spec.has_member("pawns"))
    {
      for (const auto &ps : src.pawns_spec["pawns"].object_range())
      {
        double alpha_we_tag = ps.value().has_member("alpha_we") ? ps.value()["alpha_we"].as<double>() : default_alpha_we;
        double beta_tag = ps.value().has_member("beta_bp_miss") ? ps.value()["beta_bp_miss"].as<double>() : default_beta_bp_miss;
        double height_tag = ps.value().has_member("boat_height") ? ps.value()["boat_height"].as<double>() : default_height;
        double alpha_s_tag = ps.value().has_member("alpha_speed") ? ps.value()["alpha_speed"].as<double>() : default_alpha_speed;
        pawn_param pp(alpha_we_tag, beta_tag, height_tag, alpha_s_tag);
        pawn_types[std::string(ps.key())] = pp;

      }
    }
    else if (src.pawns_spec.has_member("pawns_from_weight"))
    {
      for (const auto &ps : src.pawns_spec["pawns_from_weight"].object_range())
      {
        double alpha_we_tag = ps.value().has_member("alpha_we") ? ps.value()["alpha_we"].as<double>() : default_alpha_we;
        double beta_tag = ps.value().has_member("beta_bp_miss") ? ps.value()["beta_bp_miss"].as<double>() : default_beta_bp_miss;
        double height_tag = ps.value().has_member("boat_height") ? ps.value()["boat_height"].as<double>() : default_height;
        double alpha_s_tag = ps.value().has_member("alpha_speed") ? ps.value()["alpha_speed"].as<double>() : default_alpha_speed;
        pawn_param pp(alpha_we_tag, beta_tag, height_tag, alpha_s_tag);
        pawn_types[std::string(ps.key())] = pp;
      }
    }
  }
  //for (auto t : pawn_types) std::cout << t.first << std::endl;
  update_weights();
  c->update_weight(pawn_types);
  for (const auto &bm : c->bpmatrix)
    c->update_bpmatrix(bm.first);

  if (jconf.has_member("enable_population") && jconf["enable_population"].as<bool>())
  {
    // population output file
    auto population_date = physycom::unix_to_date(start_time, "%y%m%d_%H%M%S");
    population_out.open(state_basename + "_population_" + population_date + ".csv");
    population_out << "datetime" << separator << "timestamp";
    for (const auto &t : pawn_types) population_out << separator << t.first;
    population_out << separator << "transport";
    population_out << separator << "awaiting_transport";
    population_out << std::endl;
  }

  if (jconf.has_member("enable_influxgrid") && jconf["enable_influxgrid"].as<bool>())
  {
    auto grid_date = physycom::unix_to_date(start_time, "%y%m%d_%H%M%S");
    grid_outfile = state_basename + "_grid_" + grid_date + ".influx";
    influxgrid_out.open(grid_outfile);
  }

  if (jconf.has_member("enable_netstate") && jconf["enable_netstate"].as<bool>())
  {
    auto net_date = physycom::unix_to_date(start_time, "%y%m%d_%H%M%S");
    poly_outfile = state_basename + "_netstate_" + net_date + ".csv";
    net_out.open(poly_outfile);
    net_out << "timestamp";
    for (int i = 0; i < (int)c->poly.size(); ++i) net_out << separator << i;
    net_out << std::endl;
  }

  // init attractions
  attractions = init_attraction(jconf, c);
  std::set<int> udest;
  for (int i = 0; i < (int)attractions.size(); ++i)
  {
    udest.insert(attractions[i].node_lid);
    node_attractions[attractions[i].node_lid] = std::make_pair(NODE_ATTRACTION, attractions.begin() + i);
  }
  c->add_bpmatrix(udest);

  // init sinks and transports
  sinks = init_sinks(jconf);
  transports = init_transports(jconf, c);

  // init barriers
  init_barriers(jconf, c);
  poly2barrier.resize(c->poly.size());
  for (const auto &b : barriers)
  {
    poly2barrier[b.second.poly->lid].emplace_back(b.first);
  }
  // init polygons
  polygons = init_polygons(jconf, c);

  if (polygons.size())
  {
    // populate grid to polygon map
    grid2polygon.resize(grid.grow);
    for (auto &r : grid2polygon) r.resize(grid.gcol, std::vector<std::vector<polygon>::iterator>(0));
    for (auto p = polygons.begin(); p != polygons.end(); ++p)
    {
      for (int i = 0; i < int(p->points.size()); ++i)
      {
        int Fr, Fc, Tr, Tc;
        grid.coord_to_grid(p->points[i].ilat, p->points[(i) % p->points.size()].ilon, Fr, Fc);
        grid.coord_to_grid(p->points[i + 1].ilat, p->points[(i + 1) % p->points.size()].ilon, Tr, Tc);

        if (Tr < Fr) { Fr = Fr ^ Tr; Tr = Fr ^ Tr; Fr = Fr ^ Tr; }  // swap content
        if (Tc < Fc) { Fc = Fc ^ Tc; Tc = Fc ^ Tc; Fc = Fc ^ Tc; }

        for (int r = Fr - 1; r <= Tr + 1; ++r)
          for (int c = Fc - 1; c <= Tc + 1; ++c)
            if (std::find(grid2polygon[r][c].begin(), grid2polygon[r][c].end(), p) == grid2polygon[r][c].end())
              grid2polygon[r][c].push_back(p);
      }
    }

    // DEBUG
    //for (int r = 0; r < grid.grow; r++)
    //  for (int c = 0; c < grid.gcol; c++)
    //    if (grid2polygon[r][c].size())
    //      std::cout << "polygon grid " << r << " " << c << " " << grid2polygon[r][c].size() << std::endl;

    // polygon output file
    if (jconf.has_member("enable_polygons") && jconf["enable_polygons"].as<bool>())
    {
      auto polygon_date = physycom::unix_to_date(start_time, "%y%m%d_%H%M%S");
      polygon_outfile = state_basename + "_polygon_" + polygon_date + ".csv";
      polygon_out.open(polygon_outfile);
      polygon_out << "timestamp;PK_UID";
      for (const auto &t : pawn_types) polygon_out << separator << t.first;
      polygon_out << std::endl;
    }
  }

  // locations
  if (jconf.has_member("file_locations"))
  {
    jsoncons::json jloc;
    try
    {
      jloc = jsoncons::json::parse_file(jconf["file_locations"].as<std::string>());
    }
    catch (...)
    {
      throw std::runtime_error("Locations file not found.");
    }

    for (const auto &l : jloc.array_range())
    {
      point_base pt(int(l["lat"].as<double>() * 1e6), int(l["lon"].as<double>() * 1e6));
      auto ptp = c->project(pt, c->get_nearest_arc(pt));
      locations[l["ID"].as<std::string>()] = ptp;
    }
  }

  // eta
  if (jconf.has_member("eta"))
  {
    for (const auto &r : jconf["eta"].object_range())
    {
      std::vector<int> route;
      c->bestpath(r.value()["check_poly"].as<std::vector<int>>(), route);
      routes[std::string(r.key())] = route;
    }
  }

  // make weighted-route
  make_attr_route();
  update_attr_weights();

  // populate node status map
  for (const auto &n : c->node) node_status[n.lid] = NODE_BASE;
  for (const auto &s : sinks) node_status[s.node_lid] |= NODE_SINK;
  for (const auto &a : attractions) node_status[a.node_lid] |= NODE_ATTRACTION;
  for (const auto &s : sources)
  {
    if (s.pawns_spec.has_member("pawns"))
    {
      for (const auto &spec : s.pawns_spec["pawns"].object_range())
      {
        auto node_lid = spec.value()["start_node_lid"].as<int>();
        if (node_lid != -1)
          node_status[node_lid] |= NODE_SOURCE;
      }
    }
    if (s.is_localized)
    {
      node_status[s.node_lid] |= NODE_SOURCE;
    }
  }

  /*
    // NODEMAP DEBUG
    for(const auto &n : node_status)
    {
      int status = n.second;
      std::cout << "Node " << n.first << " : ";
      if ( status & NODE_BASE ) std::cout << "BASE ";
      if ( status & NODE_SINK ) std::cout << "SINK ";
      if ( status & NODE_SOURCE ) std::cout << "SOURCE ";
      if ( status & NODE_ATTRACTION ) std::cout << "ATTRACTION ";
      std::cout << std::endl;
    }
  */
}

simulation::~simulation()
{
  if (influxgrid_out.is_open()) influxgrid_out.close();
  if (pawn_out.is_open()) pawn_out.close();
  if (net_out.is_open()) net_out.close();
  if (polygon_out.is_open()) polygon_out.close();
  if (population_out.is_open()) population_out.close();
  if (status_out.is_open()) status_out.close();
  if (pawnstats_out.is_open()) pawnstats_out.close();
  if (wrstats_out.is_open()) wrstats_out.close();
  if (enable_status)
  {
    if (!status_out.is_open())
      status_out.open(state_basename + ".status", std::ofstream::out | std::ofstream::app );
    status_out << niter
                << separator << niter
                << separator << physycom::unix_to_date(std::time(nullptr))
                << std::endl;
    status_out.close();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// INIT FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////

void simulation::init_from_file(const jsoncons::json &jconf)
{
  std::ifstream initstate(jconf["file_init_state"].as<std::string>() + ".csv");
  if (!initstate) throw std::runtime_error("Unable to open pawn init state file.");
  pawn_param pp(default_alpha_we, default_beta_bp_miss, default_height, default_alpha_speed);
  pawn_types["locals"] = pp;
  std::string line;
  std::getline(initstate, line); // skip header
  std::map<std::string, std::vector<pawn>> mpawn;
  std::map<std::string, pawn_param> pt_temp;
  int maxid = -1;
  while (std::getline(initstate, line))
  {
    std::vector<std::string> tok;
    std::string sep = ";";
    std::string sep_vec = " ";
    physycom::split(tok, line, sep, physycom::token_compress_off);
    pawn p;
    p.id = std::stoi(tok[0]);
    maxid = (p.id > maxid) ? p.id : maxid;
    p.tag = tok[1];
    p.beta_bp = std::stod(tok[2]);
    double alpha_we = std::stod(tok[3]);
    p.status = std::stoi(tok[4]);
    p.current_poly = std::stoi(tok[5]);
    p.current_s = std::stod(tok[6]);
    p.speed = std::stod(tok[7]);
    p.current_dest = std::stoi(tok[8]);
    p.last_node = std::stoi(tok[9]);
    p.next_node = std::stoi(tok[10]);
    p.totdist = std::stoi(tok[11]);
    p.triptime = std::stoi(tok[12]);
    p.crw_tag = tok[13];
    std::vector<std::string> crw_param_vec;
    physycom::split(crw_param_vec, tok[14], sep_vec, physycom::token_compress_on);
    for (int j = 0; j < int(crw_param_vec.size()); ++j) p.crw_params.emplace_back(std::stod(crw_param_vec[j]));
    p.crw_dist = dist_map.at(p.crw_tag);
    std::vector<std::string> dest_vec;
    physycom::split(dest_vec, tok[15], sep_vec, physycom::token_compress_on);
    for (int i = 0; i < int(dest_vec.size()); ++i) p.dest.emplace_back(std::stoi(dest_vec[i]));
    mpawn[p.tag].push_back(p);
    pawn_param pptemp(alpha_we, p.beta_bp, default_height, default_alpha_speed);
    pt_temp[p.tag] = pptemp; // default_height here is broken for boat resume
  }
  initstate.close();

  for (const auto &i : pt_temp)  pawn_types[i.first] = i.second;

  update_weights();

  // update max pawn id
  pawn_id = maxid + 1;

  jsoncons::json jstate = jsoncons::json::parse_file(jconf["file_init_state"].as<std::string>() + ".json");
  start_time = jstate.has_member("start_time") ? jstate["start_time"].as<int>() : -1;
  if (jstate.has_member("start_date"))
  {
    start_date = jstate["start_date"].as<std::string>();
    start_time = (int)physycom::date_to_unix(start_date);
  }
  else
  {
    start_date = physycom::unix_to_date(start_time);
  }

  stop_time = start_time + (jconf.has_member("duration_time") ? jconf["duration_time"].as<int>() : 3600);
  stop_date = physycom::unix_to_date(stop_time);

  for (const auto &type : mpawn) pawns.push_back(type.second);

  std::set<int> uniq_dest = jstate["uniq_dest"].as<std::set<int>>();
  c->add_bpmatrix(uniq_dest);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// PAWN CREATION
////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<int, std::map<int, std::pair<int, int>>> points2stops;

std::vector<std::vector<pawn>> simulation::make_pawns(const jsoncons::json &jconf)
{
  std::vector<std::vector<pawn>> pawnv;
  if (pawn_types.size() == 0) {
    pawn_param pp(default_alpha_we, default_beta_bp_miss, default_height, default_alpha_speed);
    pawn_types["locals"] = pp;
  }
  std::set<int> uniq_dest;
  if (jconf.has_member("pawns"))
  {
    // collect all destination and fill up
    // bproutes (node-based destination map)
    for (const auto &s : jconf["pawns"].object_range())
    {
      try
      {
        auto dest = s.value().has_member("dest") ? s.value()["dest"].as<std::list<int>>() : std::list<int>({ default_dest_node });
        std::copy_if(dest.begin(), dest.end(),
          std::inserter(uniq_dest, uniq_dest.end()),
          [](int p) {
          return p >= 0;
        });
      }
      catch (...)
      {
        continue;
      }
    }
    c->add_bpmatrix(uniq_dest);

    // populate pawn vector
    for (const auto &s : jconf["pawns"].object_range())
    {
      int number = s.value()["number"].as<int>();
      if (number == 0) continue;

      int start_node = s.value()["start_node_lid"].as<int>();
      std::list<int> dest;
      try
      {
        dest = s.value().has_member("dest") ? s.value()["dest"].as<std::list<int>>() : std::list<int>({ default_dest_node });
      }
      catch (...)
      {
        try
        {
          int node_lid_dest = s.value()["dest"].as<int>();
          dest = std::list<int>({ node_lid_dest });
        }
        catch (...)
        {
          throw std::runtime_error("pawns dest unhandled : " + std::string(s.key()));
          //continue;
        }
      }

      int first_dest = -1;
      for (const auto &d : dest)
      {
        if (d < 0)
          continue;
        else
        {
          first_dest = d;
          break;
        }
      }
      int start_poly = 0;

      int idletime = std::accumulate(dest.begin(), dest.end(), 0,
        [](int sum, const int &d) {
        return sum + ((d < 0) ? -d : 0);
      });
      double beta_bp = s.value().has_member("beta_bp_miss") ? s.value()["beta_bp_miss"].as<double>() : default_beta_bp_miss;
      double alpha_we = s.value().has_member("alpha_we") ? s.value()["alpha_we"].as<double>() : default_alpha_we;
      double boat_height = s.value().has_member("boat_height") ? s.value()["boat_height"].as<double>() : default_height;
      double alpha_s = s.value().has_member("alpha_speed") ? s.value()["alpha_speed"].as<double>() : default_alpha_speed;
      std::string user_tag = std::string(s.key());

      //update pawn_types vector
      if (pawn_types.find(user_tag) == pawn_types.end()) {
        pawn_param pp(alpha_we, beta_bp, boat_height, alpha_s);
        pawn_types[user_tag] = pp;
      }
      update_weights();

      double speed_const = s.value().has_member("speed_mps") ? s.value()["speed_mps"].as<double>() : -1.0;
      double vmin = s.value().has_member("vmin_mps") ? s.value()["vmin_mps"].as<double>() : pawn_vmin;
      double vmax = s.value().has_member("vmax_mps") ? s.value()["vmax_mps"].as<double>() : pawn_vmax;
      std::uniform_real_distribution<double> rnd_speed(vmin, vmax);

      double TLB = s.value().has_member("TLB") ? s.value()["TLB"].as<double>() : -1.0;
      int TTB = s.value().has_member("TTB") ? s.value()["TTB"].as<int>() : -1;

      auto crw_type = std::string("none");
      auto crw_params = std::vector<double>();
      if (s.value().has_member("crw"))
      {
        crw_type = s.value()["crw"]["type"].as<std::string>();
        crw_params = s.value()["crw"]["params"].as<std::vector<double>>();
      }

      bool ferrypawn = s.value().has_member("ferrypawn") ? s.value()["ferrypawn"].as<bool>() : false;

      std::string cherry_key = s.value().has_member("cherry_pick") ? s.value()["cherry_pick"].as<std::string>() : "all";

      std::vector<pawn> type_pawns;
      type_pawns.reserve(number);

      for (int i = 0; i < number; ++i)
      {
        pawn pw;
        double speed = (speed_const == -1.0) ? rnd_speed(generator) : speed_const;
        if (start_node != -1 && first_dest != -1) // orig-dest
        {
          pw = pawn(start_poly, speed, std::string(s.key()), dest, beta_bp, crw_type, crw_params);
          pw.idletime = idletime;
          pw.lifetime = idletime;
          pw.TLB = TLB;
          pw.last_node = start_node;
          pw.next_node = c->bpmatrix[pw.tag][first_dest][start_node];
          if (pw.next_node == NO_ROUTE)
          {
            std::stringstream ss;
            ss << "[make_pawns] Path from " << start_node << " to " << first_dest << " impossible for user tag" << pw.tag << "!!" << std::endl;
            throw std::runtime_error(ss.str());
          }
          pw.current_poly = c->node2poly(start_node, pw.next_node)->lid;

          int speedsign;
          if (c->node[start_node].isT(c->poly[pw.current_poly]))
          {
            pw.current_s = c->poly[pw.current_poly].length;
            speedsign = -1;
          }
          else if (c->node[start_node].isF(c->poly[pw.current_poly]))
          {
            pw.current_s = 0.;
            speedsign = 1;
          }
          else
          {
            throw std::runtime_error("[make_pawns] ERROR orig-dest : no front no tail");
          }
          pw.speed *= speedsign;
        }
        else if (start_node == -1 && first_dest != -1) // random-dest
        {
          pw = pawn(start_poly, speed, std::string(s.key()), dest, beta_bp, crw_type, crw_params);
          pw.idletime = idletime;
          pw.lifetime = idletime;
          pw.TLB = TLB;
          if (TTB > 0) {
            std::uniform_int_distribution<int> rnd_ttb_max(900, TTB);
            pw.TTB = rnd_ttb_max(generator);
          }
          else
            pw.TTB = TTB;

          // roll random origin
          int sn;
          if (cherry_pick_map[cherry_key].find(first_dest) != cherry_pick_map[cherry_key].end()) {
            double score_ext = rnd_01(generator);
            auto ss = std::lower_bound(cherry_pick_map[cherry_key][first_dest].begin(),
              cherry_pick_map[cherry_key][first_dest].end(),
              score_ext,
              [](const std::pair<int, double> &p1, const double &w) {
              return p1.second < w;
            }
            );
            sn = ss[0].first;
            c->add_bpmatrix(std::set<int>({ sn }));
          }
          else {
            if (cherry_pick_map["all"].find(first_dest) == cherry_pick_map["all"].end()) {
              std::cout << "You have selected an isolated node as dest node" << std::endl;
            }
            static std::uniform_int_distribution<int> rnd_cherry_node;
            rnd_cherry_node = std::uniform_int_distribution<int>(0, int(cherry_pick_map[cherry_key].size()) - 1);
            do
            {
              sn = rnd_cherry_node(generator);
              c->add_bpmatrix(std::set<int>({ sn }));
            } while (c->bpmatrix[pw.tag].at(first_dest).at(sn) == NO_ROUTE);
          }

          auto sp = c->node2poly(sn, c->bpmatrix[pw.tag][first_dest][sn])->lid;

          int speedsign = 1;
          double ss;
          if (c->node[sn].isT(c->poly[sp]))
          {
            ss = c->poly[sp].length;
            speedsign = -1;
          }
          else if (c->node[sn].isF(c->poly[sp]))
          {
            ss = 0.;
            speedsign = 1;
          }
          else
          {
            throw std::runtime_error("ERROR orig-random make_pawns : no front no tail");
          }
          pw.last_node = sn;
          pw.next_node = c->bpmatrix[pw.tag][first_dest][sn];
          pw.current_poly = sp;
          pw.current_s = ss;
          pw.speed *= speedsign;
        }
        else if (start_node != -1 && first_dest == -1) // orig-random
        {
          pw = pawn(start_poly, speed, std::string(s.key()), dest, beta_bp, crw_type, crw_params);
          pw.idletime = idletime;
          pw.lifetime = idletime;
          pw.TLB = TLB;
          if (TTB > 0) {
            std::uniform_int_distribution<int> rnd_ttb_max(900, TTB);
            pw.TTB = rnd_ttb_max(generator);
          }
          else
            pw.TTB = TTB;

          // roll random dest
          int dn;
          if (cherry_pick_map[cherry_key].find(start_node) != cherry_pick_map[cherry_key].end()) {
            double score_ext = rnd_01(generator);
            auto ss = std::lower_bound(cherry_pick_map[cherry_key][start_node].begin(),
              cherry_pick_map[cherry_key][start_node].end(),
              score_ext,
              [](const std::pair<int, double> &p1, const double &w) {
              return p1.second < w;
            }
            );
            dn = ss[0].first;
            c->add_bpmatrix(std::set<int>({ dn }));
          }
          else {
            if (cherry_pick_map["all"].find(start_node) == cherry_pick_map["all"].end()) {
              std::cout << "You have selected an isolated node as start node" << std::endl;
            }
            static std::uniform_int_distribution<int> rnd_cherry_node;
            rnd_cherry_node = std::uniform_int_distribution<int>(0, int(cherry_pick_map[cherry_key].size()) - 1);
            do
            {
              dn = rnd_cherry_node(generator);
              c->add_bpmatrix(std::set<int>({ dn }));
            } while (c->bpmatrix[pw.tag].at(dn).at(start_node) == NO_ROUTE);
          }

          pw.dest = std::list<int>(0);
          pw.current_dest = dn;
          auto sp = c->node2poly(start_node, c->bpmatrix[pw.tag][dn][start_node])->lid;
          //std::cout << i << ") " << start_node << " " << c->bpmatrix[pw.tag][dn][start_node] << " p" << sp << std::endl;

          int speedsign = 1;
          double ss;
          if (c->node[start_node].isT(c->poly[sp]))
          {
            ss = c->poly[sp].length;
            speedsign = -1;
          }
          else if (c->node[start_node].isF(c->poly[sp]))
          {
            ss = 0.;
            speedsign = 1;
          }
          else
          {
            throw std::runtime_error("WTF orig-random make_pawns : no front no tail");
          }

          pw.last_node = start_node;
          pw.next_node = c->bpmatrix[pw.tag][dn][start_node];
          pw.current_poly = sp;
          pw.current_s = ss;
          pw.speed *= speedsign;

        }
        else if (start_node == -1 && first_dest == -1) // random-random
        {
          pw = pawn(start_poly, speed, std::string(s.key()), dest, beta_bp, crw_type, crw_params);

          // roll random origin
          int sn;
          double score_ext = rnd_01(generator);
          auto ssn = std::lower_bound(cherry_pick_sn[cherry_key].begin(),
            cherry_pick_sn[cherry_key].end(),
            score_ext,
            [](const std::pair<int, double> &p1, const double &w) {
            return p1.second < w;
          }
          );
          sn = ssn[0].first;
          c->add_bpmatrix(std::set<int>({ sn }));

          // roll random dest
          int dn;
          score_ext = rnd_01(generator);
          auto ddn = std::lower_bound(cherry_pick_map[cherry_key][sn].begin(),
            cherry_pick_map[cherry_key][sn].end(),
            score_ext,
            [](const std::pair<int, double> &p1, const double &w) {
            return p1.second < w;
          }
          );
          dn = ddn[0].first;
          c->add_bpmatrix(std::set<int>({ dn }));
          pw.dest = std::list<int>(0);
          pw.current_dest = dn;

          // find poly
          auto sp = c->node2poly(sn, c->bpmatrix[pw.tag][dn][sn])->lid;

          int speedsign = 1;
          double ss;
          if (c->node[sn].isT(c->poly[sp]))
          {
            ss = c->poly[sp].length;
            speedsign = -1;
          }
          else if (c->node[sn].isF(c->poly[sp]))
          {
            ss = 0.;
            speedsign = 1;
          }
          else
          {
            throw std::runtime_error("WTF random-random make_pawns : no front no tail");
          }

          pw.last_node = sn;
          pw.next_node = c->bpmatrix[pw.tag][dn][sn];
          pw.current_poly = sp;
          pw.current_s = ss;
          pw.speed *= speedsign;
          if (TTB > 0) {
            std::uniform_int_distribution<int> rnd_ttb_max(900, TTB);
            pw.TTB = rnd_ttb_max(generator);
          }
          else
            pw.TTB = TTB;
        }

        pw.dist_thresh = rnd_exp(generator); // PAWNKILL
        //ferryboat stuff
        pw.ferrypawn = ferrypawn;

        type_pawns.push_back(pw);
      }
      c->add_bpmatrix(uniq_dest);
      pawnv.push_back(type_pawns);
    }
  }

  return pawnv;
}

void simulation::make_attr_route()
{
  int min_attr = 1, max_attr = 4;
  max_attr = int(attractions.size()) > max_attr ? max_attr : int(attractions.size());

  std::stringstream ss;
  ss << "attr MIN " << min_attr << " MAX " << max_attr;
  wroute_info += ss.str();

  if (!attractions.size())
  {
    //std::cerr << "[make_weight_route] WARNING No attractions found, skipping wroute generation." << std::endl;
    wroute_info = "no attractions";
    return;
  }

  // map attractions to capital letters
  std::map<int, char> attr_nA;
  std::map<char, int> attr_An;
  for (int i = 0; i < (int)attractions.size(); ++i)
  {
    attr_nA[i] = 'A' + i;
    attr_An['A' + i] = i;
  }
  auto attr_list = std::accumulate(attr_An.begin(), attr_An.end(), std::string(), [](std::string s, std::pair<char, int> p) {
    return s + p.first;
  });

  // create attractions distance matrix
  std::map<int, std::map<int, double>> attr_dist;
  for (int i = 0; i < int(attr_list.size()); ++i)
  {
    for (int j = 0; j < int(attr_list.size()); ++j)
    {
      if (i == j) continue;
      std::vector<int> poly_path;
      auto ni = attractions[attr_An[attr_list[i]]].node_lid;
      auto nj = attractions[attr_An[attr_list[j]]].node_lid;
      auto ret = c->bestpath(ni, nj, poly_path); // replace with correct bp
      if (ret == BP_PATHFOUND)
      {
        attr_dist[ni][nj] = accumulate(poly_path.begin(), poly_path.end(), 0., [this](double sum, const int &p) {
          return sum + c->poly[p].length;
        });
      }
      else
      {
        throw std::runtime_error("make_weight_route : BP attr-attr " + std::to_string(ni) + " - " + std::to_string(nj) + " " + c->bestpath_err(ret));
      }
    }
  }

  for (int k = min_attr; k <= max_attr; ++k) {
    auto attr_comb = physycom::combinations(attr_list, k);
    for (auto group : attr_comb) {
      auto group_perm = physycom::permutations(group);
      attr_route.insert(attr_route.end(), group_perm.begin(), group_perm.end());
    }
  }

  //std::cout << "attr route : " << attr_route.size() << std::endl;
  for (auto as : attr_route)
  {
    wroute wr;
    wr.weight = 0.;
    wr.tag = as;

    for (int i = 0; i < int(as.size() - 1); ++i)
    {
      auto n1 = attractions[attr_An[as[i]]].node_lid;
      auto n2 = attractions[attr_An[as[i + 1]]].node_lid;
      wr.len += attr_dist[n1][n2];
      // weight combination logic
      //wr.weight *= attractions[attr_An[as[i]]].weight;

      wr.node_seq.push_back(n1);
    }
    wr.node_seq.push_back(attractions[attr_An[as.back()]].node_lid);
    //wr.weight *= attractions[attr_An[as.back()]].weight;
    //wr.weight += attractions[attr_An[as.back()]].weight;
    attr_wr[wr.tag] = wr;
  }

  // map source to small letters
  std::map<int, char> source_nA;
  std::map<char, int> source_An;
  for (int i = 0; i < int(sources.size()); ++i)
  {
    if (!sources[i].is_geofenced) continue;
    source_nA[i] = 'a' + i;
    source_An['a' + i] = i;
  }

  //connect source-startwr, endwr-source
  std::map<std::string, double> conn_source2wr;
  std::vector<int> poly_path_;
  for (auto s : source_nA)
  {
    for (auto a : attr_An)
    {
      //from source to attractions
      auto ret_ = c->bestpath(sources[s.first].node_lid, attractions[a.second].node_lid, poly_path_);
      if (ret_ == BP_PATHFOUND)
      {
        conn_source2wr[std::string() + s.second + a.first] = std::accumulate(poly_path_.begin(), poly_path_.end(), 0., [this](double sum, const int &p) {
          return sum + c->poly[p].length;
        });
      }
      else
      {
        conn_source2wr[std::string() + s.second + a.first] = -1;
        throw std::runtime_error("make_weight_route : BP src-attr " + std::to_string(sources[s.first].node_lid) + "-" + std::to_string(attractions[a.second].node_lid) + " " + c->bestpath_err(ret_));
      }
      poly_path_.clear();
      //from attractions to source
      ret_ = c->bestpath(attractions[a.second].node_lid, sources[s.first].node_lid, poly_path_);
      if (ret_ == BP_PATHFOUND) {
        conn_source2wr[std::string() + a.first + s.second] = std::accumulate(poly_path_.begin(), poly_path_.end(), 0., [this](double sum, const int &p) {
          return sum + c->poly[p].length;
        });
      }
      else {
        conn_source2wr[std::string() + a.first + s.second] = -1;
        throw std::runtime_error("make_weight_route : BP attr-src " + std::to_string(attractions[a.second].node_lid) + "-" + std::to_string(sources[s.first].node_lid) + " " + c->bestpath_err(ret_));
      }
      poly_path_.clear();
    }
  }

  // enlarge attractions wroute with start-end points from sources
  std::map<int, std::vector<wroute>> source_wr;
  for (int i = 0; i < int(sources.size()); ++i)
  {
    if (!sources[i].is_localized) continue;

    auto tag = sources[i].tag;
    std::replace(tag.begin(), tag.end(), '_', '-'); // for gnuplot labels
    for (auto p : attr_wr)
    {
      auto wr = p.second;
      std::string tag_source_attr = std::string() + source_nA[i] + wr.tag[0];
      //attach source as first node
      if (conn_source2wr[tag_source_attr] != -1)
      {
        wr.len += conn_source2wr[tag_source_attr];
        wr.node_seq.insert(wr.node_seq.begin(), sources[i].node_lid);
        wr.tag = source_nA[i] + wr.tag;
      }
      else
      {
        throw std::runtime_error("make_weight_route : BP src-attr " + std::to_string(sources[i].node_lid) + "-" + std::to_string(wr.node_seq.front()));
      }
      std::string tag_attr_source = std::string() + wr.tag[-1] + source_nA[i];
      // attach source as last node
      if (conn_source2wr[tag_attr_source] != -1)
      {
        wr.len += conn_source2wr[tag_attr_source];
        wr.node_seq.insert(wr.node_seq.end(), sources[i].node_lid);
        wr.tag += source_nA[i];
      }
      else
      {
        throw std::runtime_error("make_weight_route : BP attr-src " + std::to_string(wr.node_seq.front()) + "-" + std::to_string(sources[i].node_lid));
      }

      source_wr[i].push_back(wr);
      wroutes[wr.tag] = wr;
    }
  }

  // make binned wroute matrix < len_idx, weight_idx > and store in source object
  for (const auto &s : source_wr)
  {
    binned_wr[s.first] = std::vector<std::vector<std::string>>(N_bin, std::vector<std::string>(0));
    for (int i = 0; i < int(s.second.size()); ++i)
    {
      int idx = int(s.second[i].len / (L_binw * 1e3));
      //std::cout << idx << std::endl;
      binned_wr[s.first][idx].push_back(s.second[i].tag);
    }
  }

  if (enable_stats)
  {
    for (const auto& wr : wroutes)
    {
      auto wrt = wr.second;
      wrstats_out << sources[source_An[wrt.tag[0]]].tag
        << separator << wrt.tag
        << separator << wrt.len
        << separator << wrt.weight
        << separator << "created"
        << std::endl;
    }
  }
}

bool simulation::find_trip(int start_node, int stop_node, std::pair<int, int> &sel_couple)
{
  try {
    sel_couple = points2stops.at(start_node).at(stop_node);
  }
  catch (...) {
    int row, col;
    int slice_val = 12; //slice_val*2*default_cell_side = side of box of observation
    std::set<int> start_stops, stop_stops;
    //start_node
    grid.coord_to_grid(c->node[start_node].ilat, c->node[start_node].ilon, row, col);
    for (int slice = -slice_val; slice < slice_val; ++slice) {
      for (auto nn = grid.grid[row + slice][col + slice].node.begin(); nn != grid.grid[row + slice][col + slice].node.end(); ++nn)
      {
        if ((*nn)->stops) start_stops.insert((*nn)->lid);
      }
    }
    //stop_node
    grid.coord_to_grid(c->node[stop_node].ilat, c->node[stop_node].ilon, row, col);
    for (int slice = -slice_val; slice < slice_val; ++slice) {
      for (auto nn = grid.grid[row + slice][col + slice].node.begin(); nn != grid.grid[row + slice][col + slice].node.end(); ++nn)
      {
        if ((*nn)->stops) stop_stops.insert((*nn)->lid);
      }
    }

    if (!start_stops.size() || !stop_stops.size()) {
      points2stops[start_node][stop_node] = std::make_pair(0, 0);
      return false;
    }

    // First step: get couple of stops with max occurrance in trasp_proxy
    // Next: consider time of simulation
    int occurrance = 0;
    for (const auto &stas : start_stops) {
      for (const auto &stos : stop_stops) {
        std::vector<int> index_tt;
        for (const auto &tp : transp_proxy) {
          try {
            auto index_t = tp.second.at(stas).at(stos);
            index_tt.insert(index_tt.end(), index_t.begin(), index_t.end());
          }
          catch (...)
          {
          }
        }
        if (int(index_tt.size()) > occurrance) {
          occurrance = int(index_tt.size());
          sel_couple.first = stas;
          sel_couple.second = stos;
        }
      }
    }
    points2stops[start_node][stop_node] = sel_couple;
  }
  return true;
}

std::vector<std::vector<pawn>> simulation::make_pawns_weight(const source &src)
{
  std::vector<std::vector<pawn>> pawnv(0);

  if (src.pawns_spec.has_member("pawns_from_weight"))
  {
    for (const auto &s : src.pawns_spec["pawns_from_weight"].object_range())
    {
      auto num = s.value().has_member("number") ? s.value()["number"].as<int>() : 0;
      if (num == 0) continue;

      auto tag = src.tag;
      std::replace(tag.begin(), tag.end(), '_', '-'); // for gnuplot labels

      std::vector<pawn> type_pawns;
      type_pawns.reserve(num);
      for (int n = 0; n < num; ++n)
      {
        auto L_idx = rnd_bmdist(generator);
        while (!src.wr_bin[L_idx].size() || std::accumulate(src.wr_bin[L_idx].begin(), src.wr_bin[L_idx].end(), 0.0, [](auto &a, auto &b) {return a + b.second; }) == 0.0)
        {
          L_idx = rnd_bmdist(generator);
        }
        // roll cumulative weight
        auto Lbin = src.wr_bin[L_idx];
        int start_node;
        int first_dest;
        std::list<int> dest;
        //for (auto &pp : Lbin)
        //  std::cout << "- " << pp.second <<" " << pp.first << std::endl; //per il futuro: controlliamo la questione zeri pushati
        if (Lbin.back().second != 0.0) {
          auto w = rnd_01(generator) * Lbin.back().second;
          auto lb = std::lower_bound(Lbin.begin(),
            Lbin.end(),
            w,
            [](const std::pair<std::string, double> &p1, const double &w) {
            return p1.second < w;
          }
          );

          if (lb == Lbin.end()) lb = Lbin.end() - 1;
          auto w_idx = std::distance(Lbin.begin(), lb);
          auto wr = wroutes[Lbin[w_idx].first];

          start_node = wr.node_seq.front();
          dest = std::list<int>(wr.node_seq.begin() + 1, wr.node_seq.end()); // unsafe to assume bpmatrix is fully populated
          first_dest = *(std::find_if(wr.node_seq.begin() + 1, wr.node_seq.end(), [](const int &n) {
            return n >= 0;
          }));
        }
        else
        {
          std::cout << "There is no open attraction.Maybe you shouldn't be here tourist!" << std::endl;
          //wip  for direct them in random nodes.
          //start_node = wroutes[Lbin.front().first].node_seq.front();
          //int dest_rnd;
          //do
          //{
          //  dest_rnd = rnd_node(generator);
          //  std::cout << dest_rnd << std::endl;
          //  c->add_bpmatrix(std::set<int>({ dest_rnd }));
          //} while (c->bpmatrix["locals"].at(start_node).at(dest_rnd) == NO_ROUTE || dest_rnd == start_node);
          //std::list<int> temp{dest_rnd,start_node};
          //dest = temp; // unsafe to assume bpmatrix is fully populated
          //first_dest = dest.front();
        }
        // pawn parameters
        double alpha_we = s.value().has_member("alpha_we") ? s.value()["alpha_we"].as<double>() : default_alpha_we;
        double beta_bp = s.value().has_member("beta_bp_miss") ? s.value()["beta_bp_miss"].as<double>() : default_beta_bp_miss;
        double boat_height = s.value().has_member("boat_height") ? s.value()["boat_height"].as<double>() : default_height;
        double alpha_s = s.value().has_member("alpha_speed") ? s.value()["alpha_speed"].as<double>() : default_alpha_speed;
        std::string user_tag = std::string(s.key());

        //update pawn_types vector
        if (pawn_types.find(user_tag) == pawn_types.end()) {
          pawn_param pp(alpha_we, beta_bp, boat_height, alpha_s);
          pawn_types[user_tag] = pp;
        }

        update_weights();

        double speed_const = s.value().has_member("speed_mps") ? s.value()["speed_mps"].as<double>() : -1.0;
        double vmin = s.value().has_member("vmin_mps") ? s.value()["vmin_mps"].as<double>() : pawn_vmin;
        double vmax = s.value().has_member("vmax_mps") ? s.value()["vmax_mps"].as<double>() : pawn_vmax;
        std::uniform_real_distribution<double> rnd_speed(vmin, vmax);
        double speed = (speed_const == -1.0) ? rnd_speed(generator) : speed_const;

        bool ferrypawn = s.value().has_member("ferrypawn") ? s.value()["ferrypawn"].as<bool>() : false;

        auto crw_type = std::string("none");
        auto crw_params = std::vector<double>();
        if (s.value().has_member("crw"))
        {
          crw_type = s.value()["crw"]["type"].as<std::string>();
          crw_params = s.value()["crw"]["params"].as<std::vector<double>>();
        }

        /////////////////////
        //se p e' ferrypawn per ogni coppia cerca (punto, punto)->lista<coppia di dest>
        // aggiungi la coppia al set dest.
        // aggiungila anche a bpmatrix

        //// Per alle: incapsulami please :D
        std::list<int> new_dest;
        if (ferrypawn)
        {
          std::pair<int, int> sel_stops;
          if (find_trip(start_node, first_dest, sel_stops) && sel_stops.first != 0)
          {
            new_dest.push_back(sel_stops.first);
            new_dest.push_back(sel_stops.second);
            new_dest.push_back(first_dest);

            std::set<int> dest2add = { sel_stops.first , sel_stops.second };
            c->add_bpmatrix(dest2add);
          }
          else
            new_dest.push_back(first_dest);

          auto it1 = dest.begin();
          auto it2 = dest.begin();
          std::advance(it2, 1);
          for (int nd = 0; nd < int(dest.size()) - 1; ++nd) {
            if (find_trip(*it1, *it2, sel_stops) && sel_stops.first != 0)
            {
              new_dest.push_back(sel_stops.first);
              new_dest.push_back(sel_stops.second);
              new_dest.push_back(*it2);
              std::set<int> dest2add = { sel_stops.first , sel_stops.second };
              c->add_bpmatrix(dest2add);
            }
            else
              new_dest.push_back(*it2);

            std::advance(it1, 1);
            std::advance(it2, 1);
          }
          dest = new_dest;
        }
        //////////////////////////////////

        int next_node;
        try
        {
          next_node = c->bpmatrix[user_tag].at(first_dest).at(start_node);
          if (next_node == NO_ROUTE)
          {
            //std::cout << "Path from " << start_node << " to " << first_dest << " impossible for user type " << user_type << "!!" << std::endl;
            continue;
          }
        }
        catch (...)
        {
          //std::cout << "No BP " << start_node << " to " << first_dest << " skipping " << std::endl;
          continue;
        }
        int start_poly = c->node2poly(start_node, next_node)->lid;

        auto pw = pawn(start_poly, speed, std::string(s.key()), dest, beta_bp, crw_type, crw_params); // careful with init at -1, may crash

        // per-source wr stats
        if (enable_stats)
        {
          wrstats_out << src.tag
            << separator << "tag_broken"
            << separator << "0"
            << separator << "0"
            << separator << "pawn#" << pw.id
            << std::endl;
        }

        // lid's stuff
        pw.last_node = start_node;
        pw.next_node = next_node;

        // time stuff
        int idletime = std::accumulate(dest.begin(),
          dest.end(), 0,
          [](int sum, const int &d) {
          return sum + ((d < 0) ? -d : 0);
        });
        double TLB = s.value().has_member("TLB") ? s.value()["TLB"].as<double>() : -1.0;
        pw.idletime = idletime;
        pw.lifetime = idletime;
        pw.TLB = TLB;

        // speed stuff
        int speedsign;
        if (c->node[start_node].isT(c->poly[pw.current_poly]))
        {
          pw.current_s = c->poly[pw.current_poly].length;
          speedsign = -1;
        }
        else if (c->node[start_node].isF(c->poly[pw.current_poly]))
        {
          pw.current_s = 0.;
          speedsign = 1;
        }
        else
        {
          throw std::runtime_error("ERROR make_pawns_weight()");
        }
        pw.speed *= speedsign;

        //ferryboat stuff
        pw.ferrypawn = ferrypawn;

        type_pawns.push_back(pw);
      }
      pawnv.push_back(type_pawns);
    }
  }
  return pawnv;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// INFO
////////////////////////////////////////////////////////////////////////////////////////////////////

std::string simulation::info()
{
  std::stringstream ss;
  ss << "******** SIMULATION INFO *************************" << std::endl;
  ss << "* Start time         : " << start_date << " " << start_time << std::endl;
  ss << "* Stop time          : " << stop_date << " " << stop_time << std::endl;
  ss << "* Timestep           : " << dt << std::endl;
  ss << "* Sampling           : " << sampling_dt << std::endl;
  ss << "* Pawn types         : " << std::endl;
  for (const auto &t : pawn_types)
  {
    ss << "* - " << t.first
      << " : a_w " << t.second.alpha_we << " "
      << "  b " << t.second.beta_bp << "  "
      << "  h " << t.second.hight << "  "
      << "  a_s " << t.second.alpha_speed << "  "
      << std::endl;
  }
  ss << "* Population         : " << get_pawn_number() << std::endl;
  for (const auto &type : pawns) ss << "* - " << type.size() << " " << type.front().tag << std::endl;
  ss << "* Sources            : " << sources.size() << std::endl;
  ss << "* Sinks              : " << sinks.size() << std::endl;
  ss << "* Barriers           : " << barriers.size() << std::endl;
  ss << "* Polygons           : " << polygons.size() << " " << polygons_info << std::endl;
  ss << "* Locations          : " << locations.size() << std::endl;
  if (wroutes.size()) ss << "* Weighted Routes    : " << wroutes.size() << " (" << wroute_info << ")" << std::endl;
  else                ss << "* Weighted Routes    : " << wroutes.size() << std::endl;
  ss << "* LVL_PS timeslot    : " << lvlps_tt.size() << std::endl;
  ss << "* PG_CLOSED timeslot : " << pg_closed.size() << std::endl;
  ss << "* Transports         : " << transports.size() << " " << transport_info << std::endl;
  ss << "* Attractions        : " << attractions.size() << std::endl;
  ss << "* Grid               : " << grid.grow << "x" << grid.gcol << " (tot " << grid.grow * grid.gcol << ") " << grid.gside << "m" << std::endl;
  ss << "**************************************************" << std::endl;
  return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////

void simulation::update_sink()
{
  // delete sinked DEAD pawns
  for (const auto &s : sinks)
  {
    int time_idx = (sim_time - midn_t_start) * (int)s.despawn_dt.size() / (midn_t_stop - midn_t_start);
    if (s.despawn_dt[time_idx] == 0) continue;
    if ((sim_time - midn_t_start) % s.despawn_dt[time_idx] != 0) continue;

    for (auto &type : pawns)
    {
      type.erase(std::remove_if(type.begin(), type.end(), [&s](const pawn &p) {
        return p.status == PAWN_DEAD && p.last_node == s.node_lid;
      }), type.end());
    }
  }

  // delete non-sinked DEAD pawns
  for (auto &type : pawns)
  {
    type.erase(std::remove_if(type.begin(), type.end(), [this](const pawn &p) {
      return p.status == PAWN_DEAD && !(this->node_status[p.last_node] & NODE_SINK);
    }),
      type.end());
  }

  // safety erase
  pawns.erase(std::remove_if(pawns.begin(), pawns.end(), [](const std::vector<pawn> &t) {
    return !t.size();
  }), pawns.end());
}

void simulation::update_sources()
{
  source ctrl_src;
  std::vector<std::string> ctrl_types;
  for (auto &s : sources)
  {
    if ((sim_time - start_time) % s.creation_dt != 0) continue;
    int time_idx = int((sim_time - midn_t_start) * s.creation_rate.size() / (midn_t_stop - midn_t_start));

    // update pawns
    std::vector<std::vector<pawn>> pawn_tmp;
    if (s.pawns_spec.has_member("pawns"))
    {
      // handle control source
      if (s.is_control)
      {
        ctrl_src = s;
        for (auto &sp : s.pawns_spec["pawns"].object_range())
          ctrl_types.push_back(std::string(sp.key()));
        continue;
      }

      for (auto &sp : s.pawns_spec["pawns"].object_range())
        sp.value()["number"] = s.creation_rate[time_idx];
      pawn_tmp = make_pawns(s.pawns_spec);
    }

    // update pawns from route
    if (s.pawns_spec.has_member("pawns_from_weight"))
    {
      std::vector<std::vector<pawn>> pawn_tmp_weight;
      for (auto &sp : s.pawns_spec["pawns_from_weight"].object_range())
        sp.value()["number"] = s.creation_rate[time_idx];
      pawn_tmp_weight = make_pawns_weight(s);
      if (pawn_tmp_weight.size()) pawn_tmp.insert(pawn_tmp.end(), pawn_tmp_weight.begin(), pawn_tmp_weight.end());
    }

    // update pawn initial position
    if (s.is_geofenced)
    {
      for (auto &type : pawn_tmp)
        for (auto &p : type)
        {
          p.last_node = s.orientation == FT ? s.poly->nF->lid : s.poly->nT->lid;
          p.next_node = s.orientation == FT ? s.poly->nT->lid : s.poly->nF->lid;
          p.current_s = s.orientation == FT ? 0 : s.poly->length;                // more safety here?
          p.current_poly = s.poly->lid;
          p.speed = s.orientation == FT ? std::abs(p.speed) : -std::abs(p.speed); // more safety here?
        }
    }

    // update pawn parameters
    auto start_date = physycom::unix_to_date(sim_time);
    std::string date_stop = physycom::unix_to_date(sim_time, "%Y-%m-%d 23:00:00");
    int time_stop = int(physycom::date_to_unix(date_stop));
    int dt = time_stop - sim_time;
    dt = (dt > 0) ? dt : 3600;
    //dt = (dt < 7 * 3600) ? dt : 7 * 3600;
    for (auto &type : pawn_tmp)
    {
      for (auto &p : type)
      {
        p.trip_tstart = sim_time;
        if (enable_stats)
        {
          pawnstats_out << p.id
            << separator << p.tag
            << separator << p.totdist
            << separator << p.triptime
            << separator << p.lifetime
            << separator << start_date
            << separator << "-"
            << separator << "IN"
            << std::endl;
        }

        // TTB
        if (p.TTB < 0)
          p.TTB = dt + rnd_ttb(generator);

        //std::cout << "#" << p.id << " " << p.TTB << std::endl;
      }
    }

    // push new pawns in the correct type slot
    for (auto &type_tmp : pawn_tmp)
      for (auto &type : pawns)
        if (type.size() && type_tmp.size() && type.front().tag == type_tmp.front().tag)
        {
          type.insert(type.end(), type_tmp.begin(), type_tmp.end());
          type_tmp.clear();
        }

    // or append if type is a new type
    for (auto &type_tmp : pawn_tmp)
      if (type_tmp.size())
        pawns.push_back(type_tmp);

    // update type < tag, idx > map
    type2idx.clear();
    for (int i = 0; i < int(pawns.size()); ++i)
      type2idx[pawns[i].front().tag] = i;
  }

  // control total population
  if (ctrl_src.is_control)
  {
    if ((sim_time - start_time) % ctrl_src.creation_dt == 0)
    {
      int time_idx = int((sim_time - midn_t_start) * ctrl_src.creation_rate.size() / (midn_t_stop - midn_t_start));
      //for (auto type : pawns) std::cout << ( type.size() ? type[0].tag : std::string("nope") ) << " " << type.size() << std::endl;

      int tot = 0;
      if (ctrl_src.source_type == SOURCE_CTRL)
      {
        for (auto &sp : ctrl_src.pawns_spec["pawns"].object_range())
        {
          tot += int(pawns.size() ? pawns[type2idx[std::string(sp.key())]].size() : 0);
        }
      }
      else if (ctrl_src.source_type == SOURCE_TOT)
      {
        tot = get_pawn_number();
      }

      int tot_ctrl = ctrl_src.creation_rate[time_idx] * int(ctrl_src.pawns_spec["pawns"].size());
      int tot_created = tot_ctrl - tot;

      //std::cout << iter << "  tot " << tot << "  ctrl " << tot_ctrl << "  diff " << tot_created << std::endl;
      if (tot_created > 0)
      {
        for (auto &sp : ctrl_src.pawns_spec["pawns"].object_range())
        {
          sp.value()["number"] = tot_created / float(ctrl_src.pawns_spec["pawns"].size());
        }
        auto pawn_tmp = make_pawns(ctrl_src.pawns_spec);

        // push new pawns in the correct type slot
        for (auto &type_tmp : pawn_tmp)
          for (auto &type : pawns)
            if (type.size() && type_tmp.size() && type.front().tag == type_tmp.front().tag)
            {
              type.insert(type.end(), type_tmp.begin(), type_tmp.end());
              type_tmp.clear(); // clear type vector to avoid double pushback
            }

        // or append if type is a new type
        for (auto &type_tmp : pawn_tmp)
          if (type_tmp.size())
            pawns.push_back(type_tmp);
      }
      else if (tot_created < 0)
      {
        //std::cout << "deleting " << tot_created << " tot " << get_pawn_number() << std::endl;
        for (const auto &type_tag : ctrl_types)
        {
          auto type = &pawns[type2idx[type_tag]];
          //std::cout << "rimuovo " << tot_created << " da " << type_tag << " size " << type->size() << std::endl;
          int new_size = int(type->size() + tot_created / float(ctrl_src.pawns_spec["pawns"].size())); // + because tot_created < 0
          if (new_size > 0 && new_size < int(type->size()))
            type->resize(new_size);
          else
          {
            // move this to serious log system
            //std::cout << "[update_sources] WARNING control resize negative size " << new_size << std::endl;
          }
          //std::cout << "after removal " << type->size() << std::endl;
        }
      }
      //std::cout << "after tot " << get_pawn_number() << std::endl;
    }
  }

}

void simulation::update_velocity()
{
  // reset all poly speed
  for (auto &p : c->poly) p.counter = 0;

  // assign counters to each poly
  for (const auto &type : pawns)
    for (const auto &p : type)
    {
      //std::cout << "--- " << p.id << " " << p.current_poly << std::endl;
      c->poly[p.current_poly].counter++;
    }

  for (auto &p : c->poly)
  {
    p.density = p.counter / p.length;
    p.speed_t = (p.density > density_critical) ? speed_critical : (1 - p.density / density_critical) * p.speed;
    if (p.speed_t < speed_critical) p.speed_t = speed_critical;
  }
}

void simulation::update_transports()
{
  if (transports.size())
  {
    int time_idx = int((sim_time - start_time) / slice_transp);
    for (auto &type : pawns)
    {
      for (auto &p : type)
      {
        //std::cout << iter << ") #" << p.id << " " << p.tag << " " << p.status << " " << std::endl;
        if (!p.ferrypawn) continue;

        switch (p.status)
        {
        case PAWN_ACTIVE:
          if (p.last_node == p.current_dest
            &&
            p.dest.size())
          {
            try
            {
              std::vector<int> trip_list = transp_proxy[time_idx].at(p.current_dest).at(p.dest.front());
              //std::cout << iter << " set to AWATING " << p.current_poly << " " << p.current_dest << std::endl;
              std::vector<int> time_deps;
              for (const auto &v : trip_list)
              {
                //std::cout << "t " << time_idx << " #" << p.id << " adding " << transports[v].tt[p.current_dest][p.dest.front()].first << " to " << p.current_dest << " " << p.dest.front() << std::endl;
                time_deps.push_back(transports[v].tt[p.current_dest][p.dest.front()].first);
              }
              auto it = std::upper_bound(time_deps.begin(), time_deps.end(), sim_time);
              if (it != time_deps.end())
              {
                //std::cout << "awaiting transp " << p.id << std::endl;
                p.status = PAWN_AWAITING_TRANS;
                int departure_time = transports[trip_list[it - time_deps.begin()]].tt[p.current_dest][p.dest.front()].first;
                int arrival_time = transports[trip_list[it - time_deps.begin()]].tt[p.current_dest][p.dest.front()].second;
                p.transport_time = arrival_time - departure_time;
                p.current_dest = departure_time - sim_time;
              }
            }
            catch (...) {
              p.current_dest = p.dest.front();
              p.dest.pop_front();
              continue;
            }
          }
          break;
        case PAWN_AWAITING_TRANS:
          //std::cout << "case PAWN_AWAITING_TRANS" << p.id << std::endl;
          if (p.current_dest < 0) // take the bus!
          {
            p.status = PAWN_TRANSPORT;
            p.current_dest = p.transport_time;
          }
          break;
        case PAWN_TRANSPORT:
          //std::cout << "case PAWN_TRANSPORT" << p.id << std::endl;
          if (p.current_dest < 0) // get off the bus!
          {
            //std::cout << iter << " set to ACTIVE " << p.current_poly << " " << p.current_dest << std::endl;
            p.status = PAWN_ACTIVE;
            std::vector<int> path;

            int node_stop = p.dest.front();
            p.dest.pop_front();
            p.current_dest = p.dest.front();
            p.dest.pop_front();
            int next_n = c->bpmatrix[p.tag].at(p.current_dest)[node_stop];
            p.current_poly = c->node2poly(node_stop, next_n)->lid;
            p.next_node = next_n;

            if (c->node[node_stop].isF(c->poly[p.current_poly]))
              p.current_s = 0.0;
            else
              p.current_s = c->poly[p.current_poly].length;
          }
          break;
        case PAWN_QUEUED:
        case PAWN_AWAITING:
        case PAWN_VISITING:
        case PAWN_DEAD:
          break;
        default:
          throw std::runtime_error("[update_trasports] unknow pawm state " + std::to_string(p.status));
          break;
        }
      }
    }
  }

  //std::cout << "[update_trasports] done " << sim_time << std::endl;
}

void simulation::update_attractions()
{
  // update pawn proxy map
  if (attractions.size())
    for (int type = 0; type < (int)pawns.size(); ++type)
      for (int idx = 0; idx < (int)pawns[type].size(); ++idx)
        pawnproxy[pawns[type][idx].id] = std::make_pair(type, idx);

  auto sim_date = physycom::unix_to_date(sim_time);
  for (auto &a : attractions)
  {
    int time_idx = ((sim_time - midn_t_start) * (int)a.timecap.size()) / (midn_t_stop - midn_t_start);
    while (a.queue.size() && a.visitors < a.timecap[time_idx])
    {
      auto id = a.queue.front();
      a.queue.pop_front();
      auto p = &pawns[pawnproxy[id].first][pawnproxy[id].second];
      p->status = PAWN_VISITING;
      //p->current_dest = a.visit_time;
      p->current_dest = int(a.rnd_vis_time(generator));

      if (enable_stats)
      {
        pawnstats_out << p->id
          << separator << p->tag
          << separator << p->totdist
          << separator << p->triptime
          << separator << p->lifetime
          << separator << sim_date
          << separator << a.tag
          << separator << "ATTR-IN"
          << std::endl;

      }
      ++a.visitors;
      ++a.rate_in;
    }
  }
}

void simulation::update_weights()
{
  if (pawn_types.size() != c->bpweight.size())
  {
    for (const auto &pt : pawn_types)
    {
      if (c->bpweight.find(pt.first) == c->bpweight.end())
      {
        c->bpweight[pt.first] = c->bpweight["locals"];
        c->bpmatrix[pt.first] = c->bpmatrix["locals"];
        dynw_time_idx = sim_time;
        dynw_time_idx2 = sim_time;
      }
    }
    c->update_weight(pawn_types);
    for (const auto &bm : c->bpmatrix)
      c->update_bpmatrix(bm.first);
  }
  switch (dynw_mode)
  {
  case DYNW_MODE_LVLPS_HC:
  {
    int time_idx = ((sim_time - midn_t_start) * int(lvlps_tt.size())) / (midn_t_stop - midn_t_start);
    if (time_idx != dynw_time_idx)
    {
      dynw_time_idx = time_idx;
      c->global_lvl_ps = lvlps_tt[time_idx];
      c->update_weight_lvlps("locals", pawn_types["locals"].alpha_we, pawn_types["locals"].hight, pawn_types["locals"].alpha_speed);
      c->update_bpmatrix("locals");
      //std::cout << "[sim] " << sim_time << " idx " << time_idx << " lvlps " << c->global_lvl_ps << std::endl;
    }
    break;
  }
  case DYNW_MODE_BRIDGE_LVL:
  {
    int time_idx = ((sim_time - midn_t_start) * int(lvlps_tt.size())) / (midn_t_stop - midn_t_start);
    if (time_idx != dynw_time_idx)
    {
      dynw_time_idx = time_idx;
      c->global_lvl_ps = lvlps_tt[time_idx];
      for (const auto &i : pawn_types) {
        c->update_weight_lvlps(i.first, i.second.alpha_we, i.second.hight, i.second.alpha_speed);
        c->update_bpmatrix(i.first);
      }
    }
    break;
  }
  case DYNW_MODE_PG_CLOSED:
  {
    int time_idx = ((sim_time - start_time) * int(pg_closed.size())) / (stop_time - start_time);
    if (time_idx != dynw_time_idx)
    {
      //std::cout << time_idx << " " << sim_time <<"  " << pg_closed.size() << std::endl;
      dynw_time_idx = time_idx;
      int cnt_unchanged = 0;
      std::set<int> temp;
      for (const auto &pgc : pg_closed[time_idx])
      {
        if (pg_id2idx.find(pgc) != pg_id2idx.end())
        {
          for (const auto &pc : polygons[pg_id2idx[pgc]].poly_in)
          {
            temp.insert(pc->lid);
            if (poly_closed.find(pc->lid) != poly_closed.end())
              cnt_unchanged++;
            else {
              pc->oneway = ONEWAY_CLOSED;
            }
          }
        }
      }

      //set difference
      std::set<int> poly_open;
      std::set_difference(poly_closed.begin(), poly_closed.end(), temp.begin(), temp.end(), std::inserter(poly_open, poly_open.begin()));
      for (const auto &po : poly_open)
        c->poly[po].oneway = ONEWAY_BOTH; // in future, needed restore to the original ONEWAY (BOTH, FT or TF)

      //update if needed
      if ((poly_open.size() != 0 && cnt_unchanged != int(poly_closed.size())) || (time_idx == 0 && temp.size() != 0))
      {
        c->update_weight(pawn_types);
        for (const auto &pt : pawn_types)
          c->update_bpmatrix(pt.first);
      }

      //reassign set
      poly_closed = temp;
    }
    break;
  }
  case DYNW_MODE_HYBRID:
  {
    int time_idx2 = ((sim_time - start_time) * int(pg_closed.size())) / (stop_time - start_time);
    if (time_idx2 != dynw_time_idx2)
    {
      //std::cout << time_idx << " " << sim_time << std::endl;
      dynw_time_idx2 = time_idx2;
      int cnt_unchanged = 0;
      std::set<int> temp;
      for (const auto &pgc : pg_closed[time_idx2])
      {
        if (pg_id2idx.find(pgc) != pg_id2idx.end())
        {
          for (const auto &pc : polygons[pg_id2idx[pgc]].poly_in)
          {
            temp.insert(pc->lid);
            if (poly_closed.find(pc->lid) != poly_closed.end())
              cnt_unchanged++;
            else {
              pc->oneway = ONEWAY_CLOSED;
            }
          }
        }
      }

      //set difference
      std::set<int> poly_open;
      std::set_difference(poly_closed.begin(), poly_closed.end(), temp.begin(), temp.end(), std::inserter(poly_open, poly_open.begin()));
      for (const auto &po : poly_open)
        c->poly[po].oneway = ONEWAY_BOTH; // in future, needed restore to the original ONEWAY (BOTH, FT or TF)

      //update if needed
      if ((poly_open.size() != 0 && cnt_unchanged != int(poly_closed.size())) || (time_idx2 == 0 && temp.size() != 0))
      {
        c->update_weight(pawn_types);
        for (auto &i : pawn_types)
          c->update_bpmatrix(i.first);
      }

      //reassign set
      poly_closed = temp;
    }

    int time_idx = ((sim_time - midn_t_start) * int(lvlps_tt.size())) / (midn_t_stop - midn_t_start);
    if (time_idx != dynw_time_idx)
    {
      dynw_time_idx = time_idx;
      c->global_lvl_ps = lvlps_tt[time_idx];
      c->update_weight_lvlps("locals", pawn_types["locals"].alpha_we, pawn_types["locals"].hight, pawn_types["locals"].alpha_speed);
      c->update_bpmatrix("locals");
      //std::cout << "[sim] " << sim_time << " idx " << time_idx << " lvlps " << c->global_lvl_ps << std::endl;
    }
    break;
  }
  case DYNW_MODE_OFF:
  default:
    break;
  }
}

void simulation::update_barriers()
{
  if ((sim_time - start_time) % dump_cam_dt == 0)
  {
    int time_idx = int((sim_time - start_time) / dump_cam_dt);
    for (auto &p : barriers)
    {
      cam_counters[time_idx][p.second.tag + "_OUT"] = p.second.cnt_TF;
      cam_counters[time_idx][p.second.tag + "_IN"] = p.second.cnt_FT;
      p.second.cnt_TF = 0;
      p.second.cnt_FT = 0;
    }
    for (auto &po : c->poly){
      std::pair<std::set<int>, std::set<int>> first;
      po.uniq_pwn[time_idx] = first;
    }
  }

  if (enable_uniq_poly) {
    int time_idx = int((sim_time - start_time) / dump_cam_dt);
    for (auto &type : pawns)
    {
      for (auto &pawn : type)
      {
        if (pawn.speed > 0.0)
          c->poly[pawn.current_poly].uniq_pwn[time_idx].first.insert(pawn.id);
        else
          c->poly[pawn.current_poly].uniq_pwn[time_idx].second.insert(pawn.id);
      }
    }
  }
}

void simulation::update_attr_weights()
{
  if ((sim_time - start_time) % 3600 == 0) {
    int time_idx = int((sim_time - start_time) / 3600);
    //std::cout << "Update weights attractions routes: " << time_idx << std::endl;

    for (auto as : attr_route)
    {
      attr_wr[as].weight = 1.0;
      for (int i = 0; i < int(as.size() - 1); ++i)
      {
        //weight combination logic
        attr_wr[as].weight *= attractions[attr_An[as[i]]].weight[time_idx];
      }
      attr_wr[as].weight *= attractions[attr_An[as.back()]].weight[time_idx];
    }

    // sort wroutes by weight
    for (auto &wr : wroutes) {
      std::string attr_tag = wr.first.substr(1, wr.first.size() - 2);
      wr.second.weight = attr_wr[attr_tag].weight;
    }

    for (auto &bwr : binned_wr)
    {
      std::vector<std::vector<std::pair<std::string, double>>> wrb;
      for (auto &bin : bwr.second)
      {
        std::vector<std::pair<std::string, double>> tmp;
        std::sort(bin.begin(), bin.end(), [this](const std::string &wr1, const std::string wr2) {
          return wroutes[wr1].weight > wroutes[wr2].weight;
        });

        // create cumulative weights
        std::vector<double> w_vec, w_cumul;
        w_vec.reserve(bin.size());
        w_cumul.resize(bin.size());
        std::transform(bin.begin(),
          bin.end(),
          std::back_inserter(w_vec),
          [this](const std::string &wr) {
          return wroutes[wr].weight;
        });
        //std::cout << "w_vec "; for (auto &v : w_vec) std::cout << v << " "; std::cout << std::endl;
        std::partial_sum(w_vec.begin(), w_vec.end(), w_cumul.begin());

        tmp.resize(bin.size());
        for (int i = 0; i < int(bin.size()); ++i)
        {
          tmp.push_back(std::make_pair(bin[i], w_cumul[i]));
        }
        wrb.push_back(tmp);

        //std::cout << "w_cum "; for (auto &v : w_cumul) std::cout << v << " "; std::cout << std::endl;
        //for (auto i : bin) std::cout << wroutes[i].weight << " "; std::cout << std::endl;
      }
      sources[bwr.first].wr_bin = wrb;
      sources[bwr.first].binned_wr = bwr.second;
    }
  }
}

void simulation::update_status()
{
  if (enable_status)
  {
    int iter_now = (sim_time - start_time) / dt;
    if (iter_now % 25 == 0)
    {
      status_out << iter_now
                 << separator << niter
                 << separator << physycom::unix_to_date(std::time(nullptr))
                 << std::endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// PAWN EVOLUTION
////////////////////////////////////////////////////////////////////////////////////////////////////

void simulation::evolve(pawn &p)
{
  //std::cout << "Evolving iter " << iter << " ts " << sim_time << " p#" << p.id << " : s " << p.current_s << " p " << p.current_poly << " d " << p.current_dest << " nl " << p.last_node << " nn " << p.next_node << " v " << p.speed << " dist " << p.totdist << " st " << p.status <<" tag " << p.tag << std::endl;
  p.lifetime += dt;

  switch (p.status)
  {
  case PAWN_DEAD:
  {
    p.lifetime -= dt;
    break;
  }
  case PAWN_AWAITING:
  {
    if (p.current_dest == -1)
    {
      p.current_dest = p.dest.front();
      p.dest.pop_front();            // if last dest is a rest time this causes segfault <- FIX ME!
      p.status = PAWN_ACTIVE;
    }
    else
    {
      p.current_dest += dt;
      p.current_dest = (p.current_dest > -1) ? -1 : p.current_dest;
    }
    break;
  }
  case PAWN_ACTIVE:
  {
    // TTB
    if (p.TTB > 0 && p.lifetime > p.TTB)
    {
      if (p.dest.size())
      {
        kill(p, "ttb_kill");
        //std::cout << "#" << p.id << " vado a casa " << p.current_dest << std::endl;
        return;
      }
    }

    // conditional random walk
    double roll = rnd_01(generator);
    double stop_prob = (*p.crw_dist)(p.totdist, p.crw_params);
    if (roll > stop_prob) return;

    // take a step
    p.speed = ((p.speed < 0.0) ? -1.0 : 1.0) * c->poly[p.current_poly].speed_t;
    auto ds = dt * p.speed;

    // barrier crossing
    if (poly2barrier[p.current_poly].size())
    {
      for (const auto &bt : poly2barrier[p.current_poly])
      {
        double sb = barriers[bt].s;
        if ((p.current_s < sb && p.current_s + ds > sb) || (p.current_s > sb && p.current_s - ds < 0.0))      // crossing FT
          barriers[bt].cnt_FT++;
        else if ((p.current_s > sb && p.current_s + ds < sb) || (p.current_s < sb && p.current_s - ds > c->poly[p.current_poly].length)) // crossing TF
          barriers[bt].cnt_TF++;
      }
    }
    p.current_s += ds;

    p.totdist += std::abs(ds);
    p.triptime += dt;

    // change poly if needed
    if (p.current_s > c->poly[p.current_poly].length)
    {
      p.dlen = p.current_s - c->poly[p.current_poly].length;
      evolve_poly(p, c->poly[p.current_poly].nT);
    }
    else if (p.current_s < 0.0)
    {
      p.dlen = -1.0 * p.current_s;
      evolve_poly(p, c->poly[p.current_poly].nF);
    }
    break;
  }
  case PAWN_TRANSPORT:
  {
    p.current_dest -= dt;
    break;
  }
  case PAWN_QUEUED:
    break;
  case PAWN_VISITING:
  {
    p.current_dest -= dt;
    if (p.current_dest < 0)
    {
      p.current_dest = p.dest.front();
      p.dest.pop_front();
      p.status = PAWN_ACTIVE;

      if (enable_stats)
      {
        pawnstats_out << p.id
          << separator << p.tag
          << separator << p.totdist
          << separator << p.triptime
          << separator << p.lifetime
          << separator << physycom::unix_to_date(sim_time)
          << separator << node_attractions[p.last_node].second->tag
          << separator << "ATTR-OUT"
          << std::endl;
      }

      --(node_attractions[p.last_node].second->visitors);
    }
    break;
  }
  case PAWN_AWAITING_TRANS:
    p.current_dest -= dt;
    break;
  default:
    throw std::runtime_error("Uknown status(" + std::to_string(p.status) + ") pawn #" + std::to_string(p.id));
  }

  //if(p.id == 133) std::cout << "Evolve ok" << std::endl;
  return;
}

void simulation::evolve_poly(pawn &p, const node_it &node)
{
  //std::cout << "Evolve_poly #" << p.id << " s " << p.current_s << std::endl;

  // handle destination
  p.last_node = p.next_node;
  if (p.last_node == p.current_dest)
  {
    if (p.dest.size() == 0)
    {
      kill(p);
      return;
    }
    else if (node_attractions[p.current_dest].first == NODE_ATTRACTION)
    {
      int time_idx_attr = ((sim_time - midn_t_start) * (int)node_attractions[p.current_dest].second->timecap.size()) / (midn_t_stop - midn_t_start);
      if (node_attractions[p.current_dest].second->timecap[time_idx_attr] > 0)
      {
        p.status = PAWN_QUEUED;
        node_attractions[p.current_dest].second->queue.emplace_back(p.id);
        return;
      }
      else
      {
        if (p.dest.size() == 0)
        {
          kill(p);
          return;
        }
        else
        {
          p.current_dest = p.dest.front();
          p.dest.pop_front();
        }
      }
    }
    else
    {
      for (const auto &t : transports)
        if (std::find(t.stops.begin(), t.stops.end(), p.last_node) != t.stops.end())
          return;
      p.current_dest = p.dest.front();
      p.dest.pop_front();
      if (p.current_dest < 0)
      {
        p.status = PAWN_AWAITING;
        return;
      }
    }
  }

  // next dest handling
  p.pick_next_bp(c, node);
  if (p.next_node == -1) p.status = PAWN_DEAD; //the pawn is trapped!!
  if (p.status == PAWN_DEAD) return;

  // correctly handle direction on next poly
  p.current_poly = c->node2poly(node->lid, p.next_node)->lid;
  if (p.dlen > c->poly[p.current_poly].length)  // check if dlen is longer than next poly
  {
    // std::cout << "Warning: Pawn movement exceeded poly length. Try with lower dt." << std::endl;
    //std::cout << " -- " << p.dlen << " / " <<  c->poly[p.current_poly].length << " -- " << std::endl;
    p.dlen = c->poly[p.current_poly].length;
  }
  if (node->isT(c->poly[p.current_poly]))                       // case for
  {                                                             // *--------*  T--------F
    p.current_s = c->poly[p.current_poly].length - p.dlen;      //          ^
    p.speed = -1. * std::abs(p.speed);
  }
  else if (node->isF(c->poly[p.current_poly]))                  // case for
  {                                                             // *--------*  F--------T
    p.current_s = p.dlen;                                       //          ^
    p.speed = std::abs(p.speed);
  }
  else                                                          // case for
  {                                                             // *--------*  *--------*
    p.speed = -1. * std::abs(p.speed);                          // ^
  }

  // update counters
  c->poly[p.current_poly].tot_count++;

  //std::cout << "Evolve_poly out" << std::endl;
  return;
}

void simulation::kill(pawn &p, const std::string &mode)
{
  p.status = PAWN_DEAD;
  if (enable_stats)
  {
    pawnstats_out << p.id
      << separator << p.tag
      << separator << p.totdist
      << separator << p.triptime
      << separator << p.lifetime
      << separator << physycom::unix_to_date(sim_time)
      << separator << mode
      << separator << "OUT"
      << std::endl;
  }
}
