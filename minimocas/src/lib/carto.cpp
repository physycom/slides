#include <algorithm>
#include <numeric>
#include <cstdlib> // rand
#include <set>
#include <chrono>

#include <omp.h>

#include <physycom/string.hpp>
#include <physycom/combinatoric.hpp>

#include <carto.h>

#define each(x, y)  auto x = y.begin(); x != y.end(); ++x
#define eachr(x, y) auto x = y.rbegin(); x != y.rend(); ++x

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// GEOMETRY
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename U>
double distance(const T &t, const U &u)
{
  double dx, dy;
  double dslon = DSLAT * std::cos(u.ilat*IDEG_TO_RAD);
  dx = dslon*(t.ilon - u.ilon);
  dy = DSLAT*(t.ilat - u.ilat);
  return std::sqrt(dx*dx + dy*dy);
}
template double distance(const point_base &pt1, const point_base &pt2);
template double distance(const point_base &pt, const node_base &n);
template double distance(const node_base &n, const point_base &pt);
template double distance(const node_base &n1, const node_base &n2);
template double distance(const point_proj &pt1, const point_base &pt2);

template<> double distance(const point_it &pt1, const point_it &pt2) { return distance(*pt1, *pt2); }
template<> double distance(const node_it &n1, const node_it &n2) { return distance(*n1, *n2); }

template<>
double distance(const point_base &pt, const arc_base &a)
{
  int dplat, dplon;
  double dslon, px, py;
  dplon = pt.ilon - a.ptF->ilon;
  dplat = pt.ilat - a.ptF->ilat;
  dslon = DSLAT * std::cos(a.ptF->ilat * IDEG_TO_RAD);
  px = dslon * dplon;
  py = DSLAT * dplat;
  double a_p = a.ux * px + a.uy * py; // projection of p on a

  double dist = 0.0;
  if( a_p < 0 )
    dist = distance(pt, *a.ptF);
  else if ( a_p > a.length )
    dist = distance(pt, *a.ptT);
  else
    dist = std::abs( a.ux * py - a.uy * px );

  return dist;
}
template<> double distance(const arc_base &a, const point_base &pt) { return distance(pt, a); }

template<> double distance(const point_base &pt, const poly_base &p)
{
  std::vector<double> dist;
  for (const auto &a : p.arc)
  {
    dist.emplace_back(distance(pt, *a));
  }
  std::sort(dist.begin(), dist.end());
  return dist.front();
}
template <> double distance(const poly_base &p, const point_base &pt) { return distance(pt, p); }

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// POINT
////////////////////////////////////////////////////////////////////////////////////////////////////

point_base::point_base() : ilat(0), ilon(0) {}

point_base::point_base(const int &ilat, const int &ilon) : ilat(ilat), ilon(ilon) {}

std::ostream &
operator<<(std::ostream &stream, const point_base &pt)
{
  stream << pt.ilat << " " << pt.ilon;
  return stream;
}

point_proj::point_proj() : point_base() {}

point_proj point_proj::operator=(const point_base &pt)
{
  ilat = pt.ilat;
  ilon = pt.ilon;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// ARC
////////////////////////////////////////////////////////////////////////////////////////////////////

arc_base::arc_base(const int &_lid, const point_it &_ptF, const point_it &_ptT, const poly_it &_p, const double &_length, const double &_s) :
lid(_lid),
length(_length), s(_s),
ptF(_ptF), ptT(_ptT), p(_p)
{
  int dalat, dalon;
  dalon = ptT->ilon - ptF->ilon;
  dalat = ptT->ilat - ptF->ilat;
  double dslon = DSLAT * std::cos(ptF->ilat * IDEG_TO_RAD);
  ux = dslon * dalon / length;
  uy = DSLAT * dalat / length;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// POLY
////////////////////////////////////////////////////////////////////////////////////////////////////

poly_base::poly_base() :
  cid(0),
  oneway(ONEWAY_BOTH),
  cntFT(1),
  cntTF(1),
  ok_edit(false),
  tot_count(0),
  speed(1.0),
  counter(0),
  density(0),
  nFcid(0),
  nTcid(0) {}

poly_base::poly_base(const unsigned long long int &_cid, const unsigned long long int &_nFcid, const unsigned long long int &_nTcid) :
    cid(_cid),
    oneway(ONEWAY_BOTH),
    cntFT(1),
    cntTF(1),
    ok_edit(false),
    tot_count(0),
    speed(1.0),
    counter(0),
    density(0),
    nFcid(_nFcid),
    nTcid(_nTcid) {}

point_base poly_base::get_point(const double &s)
{
  if( s > length )
    return *point[point.size()-1];
  else if ( s < 0 )
    return *point[0];

  auto ca = arc.begin();
  for (; ca != arc.end(); ++ca) if (s < (*ca)->s) break;
  --ca;
  //std::cout << "Poly " << lid  << " " << length << " (" << arc.size() << ") " << (*ca)->lid << " " << (*ca)->s << std::endl;

  point_base pt;
  pt.ilat = (*ca)->ptF->ilat + int( (s - (*ca)->s) / (*ca)->length * ((*ca)->ptT->ilat - (*ca)->ptF->ilat) );
  pt.ilon = (*ca)->ptF->ilon + int( (s - (*ca)->s) / (*ca)->length * ((*ca)->ptT->ilon - (*ca)->ptF->ilon) );

  return pt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// NODE
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T> bool node_base::isF(const T &t) const { return false; }
template<typename T> bool node_base::isT(const T &t) const { return false; }
template<> bool node_base::isF(const poly_base &p) const { return ( *(p.nF)  == *this ) ? true : false; }
template<> bool node_base::isF(const poly_it &p)   const { return ( *(p->nF) == *this ) ? true : false; }
template<> bool node_base::isT(const poly_base &p) const { return ( *(p.nT)  == *this ) ? true : false; }
template<> bool node_base::isT(const poly_it &p)   const { return ( *(p->nT) == *this ) ? true : false; }

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// GRID AND CELL
////////////////////////////////////////////////////////////////////////////////////////////////////

cell_base::cell_base() {}

grid_base::grid_base() {}

grid_base::grid_base(const int &ilat_max, const int &ilat_min, const int &ilon_max, const int &ilon_min, const double &cell_side, std::vector<node_base> &node, std::vector<arc_base> &arc)
{
  // set grid parameters
  gside = cell_side;
  gdlat = int(gside / DSLAT);
  double dslon = DSLAT * std::cos(0.5 * (ilat_max + ilat_min) * IDEG_TO_RAD);
  gdlon = int(gside / dslon);
  gilatmax = ilat_max + gdlat;
  gilatmin = ilat_min - gdlat;
  gilonmax = ilon_max + gdlon;
  gilonmin = ilon_min - gdlon;
  grow = int((gilatmax - gilatmin) / gdlat) + 1;
  gcol = int((gilonmax - gilonmin) / gdlon) + 1;
  grid.resize(grow, std::vector<cell_base>(gcol));

  // set grid cell gnss data
  for(int row = 0; row < grow; ++row)
    for(int col = 0; col < gcol; ++col)
    {
      grid[row][col].ilatbot  = gilatmin + (row) * gdlat;
      grid[row][col].ilonleft = gilonmin + (col) * gdlon;
      grid[row][col].ilatcen  = gilatmin + int((0.5 + row) * gdlat);
      grid[row][col].iloncen  = gilonmin + int((0.5 + col) * gdlon);
    }

  // link node to grid
  for(auto n = node.begin(); n != node.end(); ++n)
  {
    int row, col;
    coord_to_grid(n->ilat, n->ilon, row, col);
    for(int i=-1; i<=1; ++i) for(int j=-1; j<=1; ++j) grid[row+i][col+j].node.push_back(n);
  }

  // link arc to grid
  for(auto a = arc.begin(); a != arc.end(); ++a)
  {
    int Fr, Fc, Tr, Tc;
    coord_to_grid(a->ptF->ilat, a->ptF->ilon, Fr, Fc);
    coord_to_grid(a->ptT->ilat, a->ptT->ilon, Tr, Tc);

    if( Tr < Fr ) { Fr = Fr ^ Tr; Tr = Fr ^ Tr; Fr = Fr ^ Tr; }  // swap content
    if( Tc < Fc ) { Fc = Fc ^ Tc; Tc = Fc ^ Tc; Fc = Fc ^ Tc; }

    for(int r = Fr-1; r <= Tr+1; ++r)
      for(int c = Fc-1; c <= Tc+1; ++c)
        if( distance( point_base({grid[r][c].ilatcen, grid[r][c].iloncen}), *a) < 1.81 * gside )
          grid[r][c].arc.push_back(a);
  }
}

void grid_base::dump_geojson(const std::string &filename)
{
  jsoncons::ojson geojson;
  jsoncons::ojson features = jsoncons::ojson::array();
  geojson["type"] = "FeatureCollection";

  for(int col=0; col<gcol; ++col)
  {
    for(int row=0; row<grow; ++row)
    {
      auto c = grid[row][col];

      jsoncons::ojson feature;
      jsoncons::ojson coordinates = jsoncons::ojson::parse("[[]]");
      double latbot   = (c.ilatbot) * 1e-6;
      double lattop   = (c.ilatbot + gdlat) * 1e-6;
      double lonleft  = (c.ilonleft) * 1e-6;
      double lonright = (c.ilonleft + gdlon) * 1e-6;
      coordinates[0].push_back(jsoncons::ojson::parse("[" + std::to_string(lonleft) + "," + std::to_string(latbot) + "]"));
      coordinates[0].push_back(jsoncons::ojson::parse("[" + std::to_string(lonright) + "," + std::to_string(latbot) + "]"));
      coordinates[0].push_back(jsoncons::ojson::parse("[" + std::to_string(lonright) + "," + std::to_string(lattop) + "]"));
      coordinates[0].push_back(jsoncons::ojson::parse("[" + std::to_string(lonleft) + "," + std::to_string(lattop) + "]"));
      coordinates[0].push_back(jsoncons::ojson::parse("[" + std::to_string(lonleft) + "," + std::to_string(latbot) + "]"));
      jsoncons::ojson geometry;
      geometry["coordinates"] = coordinates;
      geometry["type"] = "Polygon";
      jsoncons::ojson properties;
      properties["id"] = row*gcol + col;
      properties["cell"] = jsoncons::ojson::parse("[" + std::to_string(row) + "," + std::to_string(col) + "]");
      feature["type"] = "Feature";
      feature["properties"] = properties;
      feature["geometry"] = geometry;
      features.push_back(feature);
    }
  }
  geojson["features"] = features;

  std::ofstream out(filename);
  //out << jsoncons::pretty_print(geojson);
  out << geojson;
  out.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// CART
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
node_it cart::get_nearest_node(const point_base &pt)
{
  int r, c;
  grid.coord_to_grid(pt.ilat, pt.ilon, r, c);
  double d, dmin = std::numeric_limits<double>::max();
  node_it nearest = node.end();
  for (each(nn, grid.grid[r][c].node))
  {
    d = distance(pt, **nn);
    if (d < dmin)
    {
      dmin = d;
      nearest = *nn;
    }
  }
  return nearest;
}
template <>
node_it cart::get_nearest_node(const point_it &pt) { return get_nearest_node(*pt); }
template <>
node_it cart::get_nearest_node(const point_proj &ppt)
{
  auto pt = point_base(ppt.ilat, ppt.ilon);
  return get_nearest_node(pt);
}

template <>
arc_it cart::get_nearest_arc(const point_base &pt)
{
  int r, c;
  grid.coord_to_grid(pt.ilat, pt.ilon, r, c);
  double d, dmin = std::numeric_limits<double>::max();
  arc_it nearest = arc.end();
  for (each(a, grid.grid[r][c].arc))
  {
    d = distance(pt, **a);
    if (d < dmin)
    {
      dmin = d;
      nearest = *a;
    }
  }
  return nearest;
}
template <>
arc_it cart::get_nearest_arc(const point_it &a) { return get_nearest_arc(*a); }

point_proj cart::project(const point_base &pt, const arc_it &a)
{
  point_proj proj;
  proj.a = a;
  if( a == arc.end() ) return proj;

  int dplat, dplon;
  double px, py;
  double dslon = DSLAT * std::cos(a->ptF->ilat * IDEG_TO_RAD);
  dplon = pt.ilon - a->ptF->ilon;
  dplat = pt.ilat - a->ptF->ilat;
  px = DSLAT * dplat;
  py = dslon * dplon;

  double p_a = px * a->ux + py * a->uy;
  if ( p_a < 0 )
  {
    proj = *(a->ptF);
    proj.s = 0;
  }
  else if ( p_a > a->length )
  {
    proj = *(a->ptT);
    proj.s = a->length;
  }
  else
  {
    proj.ilat = a->ptF->ilat + int( p_a / a->length * (a->ptT->ilat - a->ptF->ilat) );
    proj.ilon = a->ptF->ilon + int( p_a / a->length * (a->ptT->ilon - a->ptF->ilon) );
    proj.s = p_a;
  }

  return proj;
}

std::list<unsigned long long> cart::get_poly_insquare(const int &lat, const int &lon, const double &side)
{
  double dslon = DSLAT * std::cos(lat*IDEG_TO_RAD);
  int lat_max = lat + int(side / DSLAT);
  int lat_min = lat - int(side / DSLAT);
  int lon_max = lon + int(side / dslon);
  int lon_min = lon - int(side / dslon);

  int row_max = (lat_max - grid.gilatmin) / grid.gdlat;
  int col_max = (lon_max - grid.gilonmin) / grid.gdlon;
  int row_min = (lat_min - grid.gilatmin) / grid.gdlat;
  int col_min = (lon_min - grid.gilonmin) / grid.gdlon;

  std::set<unsigned long long> poly_cid;
  for (int i=row_min; i<=row_max; ++i)
    for (int j = col_min; j<=col_max; ++j)
      if (i == row_min || i== row_max || j==col_min ||j==col_max)
        for (auto &k : grid[i][j].arc)
        {
          if (distance(*k,point_base(lat, lon))<=side) poly_cid.insert(k->p->cid);
        }
      else
        for (auto &k : grid[i][j].arc) poly_cid.insert(k->p->cid);

  return std::list<unsigned long long>(poly_cid.begin(), poly_cid.end());
}

template <>
double cart::get_weight<FT>(const poly_base &p, const double alpha_we, const double alpha_s) const
{
  return p.length * ( alpha_we + alpha_s * (VMIN_BP / p.speed) + (1 - alpha_we - alpha_s) * ( 1. / p.cntFT - 1. / max_cntFT ) / ( 1. / min_cntFT - 1. / max_cntFT ) );
}

template <>
double cart::get_weight<TF>(const poly_base &p, const double alpha_we, const double alpha_s) const
{
  return p.length * ( alpha_we  + alpha_s * (VMIN_BP / p.speed) + (1 - alpha_we - alpha_s) * ( 1. / p.cntTF - 1. / max_cntTF ) / ( 1. / min_cntTF - 1. / max_cntTF ) );
}

cart::cart() {}

cart::cart(jsoncons::json jconf, bool skip_check) :
jconf(jconf), a_eff(0.9),
global_lvl_ps(0),
max_cntFT(2), min_cntFT(1), max_cntTF(2), min_cntTF(1),
alpha_we(1.0),
skip_check(skip_check)
{
  grid_cell_side = int(jconf.has_member("carto_cell_side") ? jconf["carto_cell_side"].as<double>() : DEFAULT_CELL_SIDE); // why as int?
  n_thread = jconf.has_member("n_thread") ? jconf["n_thread"].as<int>() : -1;

  load_poly();      MESSAGE("load_poly() DONE!\n");       // Parse input and create poly vector
  make_node();      MESSAGE("make_node() DONE!\n");       // Create nodes vector
  make_arc();       MESSAGE("make_arc() DONE!\n");        // Create arc vector
  link_poly();      MESSAGE("link_poly() DONE!\n");       // Link poly vector to nodes and poly vector
  link_node();      MESSAGE("link_node() DONE!\n");       // Link nodes to nodes
  make_grid();      MESSAGE("make_grid() DONE!\n");       // Fill grid content
  make_noturn();    MESSAGE("load_noturn() DONE!\n");     // Read forbidden turn
  patch_cart();     MESSAGE("patch_cart() DONE!\n");      // Patch network structure
  check_network();  MESSAGE("check_network() DONE!\n");   // Check network consistency
  make_tmatrix();   MESSAGE("make_tmatrix() DONE!\n");    // Create transitions matrix
  make_bpmatrix();  MESSAGE("make_bpmatrix() DONE!\n");   // Create bestpath destination matrix
  make_polyusage(); MESSAGE("make_polyusage() DONE!\n");  // Parse poly usage file
  make_subgraph();  MESSAGE("make_subgraph() DONE!\n");   // Parse poly usage file
}

std::string cart::info()
{
  std::stringstream ss;
  ss << "******** CARTOGRAPHY INFO ************" << std::endl;
  ss << "* Poly      : " << poly.size() << std::endl;
  ss << "* Node      : " << node.size() << std::endl;
  ss << "* Point     : " << point.size() << std::endl;
  ss << "* Arc       : " << arc.size() << std::endl;
  ss << "* Grid      : " << grid.grow << "x" << grid.gcol << " (tot " << grid.grow * grid.gcol << ") " << grid.gside << "m" << std::endl;
  ss << "* LAT range : " << ilat_min << " " << ilat_max << std::endl;
  ss << "* LON range : " << ilon_min << " " << ilon_max << std::endl;
  ss << "* BPmatrix  : " << bpmatrix.size() << " x " << ( bpmatrix.size() ? bpmatrix["locals"].size() : 0 ) << " (alpha " << alpha_we << ") " << bpm_perf << std::endl;
  ss << "* Tmatrix   : " << trans.size() << std::endl;
  ss << "* Routes    : " << routes.size() << std::endl;
  ss << "* Subgraph  :"; for (const auto &i : subgraph) ss << " " << i.first << " (" << int(i.second.size() * 100.0 / node.size()) << "%)"; ss << std::endl;
  ss << "**************************************" << std::endl;
  return ss.str();
}

void cart::load_poly()
{
  std::ifstream pro, pnt;

  if (jconf.has_member("file_pro")) pro.open(jconf["file_pro"].as<std::string>());
  else pro.open("strade.pro");
  if (jconf.has_member("file_pnt")) pnt.open(jconf["file_pnt"].as<std::string>());
  else pnt.open("strade.pnt");

  if(!pro || !pnt) throw std::runtime_error("Cartography files not found.");

  // Precompute network dimensions
  auto poly_num = std::count(std::istreambuf_iterator<char>(pro), std::istreambuf_iterator<char>(), '\n');
  auto point_num = std::count(std::istreambuf_iterator<char>(pnt), std::istreambuf_iterator<char>(), '\n') - poly_num;
  auto arc_num = point_num - poly_num;
  auto node_num = poly_num * 2;
  pro.clear();
  pro.seekg(0);
  pnt.clear();
  pnt.seekg(0);

  MESSAGE("Prealloc poly  : %ld \n", poly_num);
  MESSAGE("Prealloc point : %ld \n", point_num);
  MESSAGE("Prealloc arc   : %ld \n", arc_num);
  MESSAGE("Prealloc node  : %ld \n", node_num);

  poly.reserve(int(1.2 * poly_num));
  point.reserve(int(1.2 * point_num));
  arc.reserve(int(1.2 * arc_num));
  node.reserve(int(1.2 * node_num));

  // Init bbox
  ilat_min = std::numeric_limits<int>::max();
  ilat_max = std::numeric_limits<int>::min();
  ilon_min = std::numeric_limits<int>::max();
  ilon_max = std::numeric_limits<int>::min();

  poly_base p;
  unsigned long long int skipi;
  float skipi_d;
  std::string skipi_s;
  int npoint;

  std::string header; std::getline(pro, header);
  int lid = 0;
  unsigned long long int check_cid;
  while( pro >> p.cid >> p.nFcid >> p.nTcid >> p.length >> p.lvl_ps >> p.type >> skipi_d >> skipi_s >> p.speed >> p.oneway >> p.name )
  {
    node_poly[p.nFcid][p.nTcid] = p.cid;
    node_poly[p.nTcid][p.nFcid] = p.cid;
    p.lid = lid++;
    p.speed = p.speed * kmh_to_ms;
    p.speed_t = p.speed;

    pnt >> check_cid >> skipi >> npoint;
    if ( check_cid != p.cid ) throw std::runtime_error("Cartography inconsistence!!! Broken poly cid : " + std::to_string(p.cid));
    point_base pt;
    while( npoint-- )
    {
      pnt >> pt.ilat >> pt.ilon;
      ilat_min = (ilat_min < pt.ilat) ? ilat_min : pt.ilat;
      ilat_max = (ilat_max > pt.ilat) ? ilat_max : pt.ilat;
      ilon_min = (ilon_min < pt.ilon) ? ilon_min : pt.ilon;
      ilon_max = (ilon_max > pt.ilon) ? ilon_max : pt.ilon;
      point.push_back(pt);
      p.point.push_back(point.end()-1);
    }

    poly.push_back(p);
    double len = 0.0;
    for (int i = 0; i<int(poly.back().point.size() - 1); ++i)
      len += distance(*poly.back().point[i], *poly.back().point[i+1]);
    poly.back().length = len;
    p.point.clear();
  }
  pro.close();
  pnt.close();

  ilat_center = (ilat_max + ilat_min) >> 1;
  ilon_center = (ilon_max + ilon_min) >> 1;

}

void cart::make_node()
{
  std::map<unsigned long long int, node_base> nodeset; // < cid, node >
  unsigned long long int nF, nT;
  for (auto p = poly.begin(); p != poly.end(); ++p)
  {
    // create F T nodes
    nF = p->nFcid;
    nT = p->nTcid;
    nodeset[nF].cid = nF;
    nodeset[nT].cid = nT;
    nodeset[nF].ilat = p->point.front()->ilat;
    nodeset[nF].ilon = p->point.front()->ilon;
    nodeset[nT].ilat = p->point.back()->ilat;
    nodeset[nT].ilon = p->point.back()->ilon;

    // store into connection map
    node_map[nF].insert(nT);
    node_map[nT].insert(nF);

    // link node to poly
    nodeset[nF].link.push_back(std::make_pair(node.begin(), p));
    nodeset[nT].link.push_back(std::make_pair(node.begin(), p));
  }

  // create nodes vector and generate lid's
  int lid = 0;
  for(auto &n : nodeset)
  {
    n.second.lid = lid++;
    node.push_back(n.second);

    // fill cid lid node map
    node_cid[n.second.cid] = n.second.lid;
  }

  // assign score and city for rnd extraction to each node
  if (jconf.has_member("file_score_pop")) {
    std::ifstream score_pop(jconf["file_score_pop"].as<std::string>());
    if (!score_pop)
      throw std::runtime_error("unable to open score pop file " + jconf["file_score_pop"].as<std::string>());

    std::string line;
    std::getline(score_pop, line); // skip header

    while (std::getline(score_pop, line))
    {
      std::vector<std::string> tok;
      std::string sep = ";";
      physycom::split(tok, line, sep, physycom::token_compress_off);
      node[stoi(tok[1])].score = stod(tok[2]);
      node[stoi(tok[1])].city_membership = tok[3];
    }
  }
}

void cart::make_arc()
{
  std::vector<point_it>::iterator pt_next;
  int arc_lid = 0;
  double len, s0;
  for (auto p = poly.begin(); p != poly.end(); ++p)
  {
    s0 = 0;
    for(auto pt = p->point.begin(); pt != p->point.end()-1; ++pt)
    {
      pt_next = (p->point.begin() + std::distance(p->point.begin(), pt) + 1);
      len = distance(**pt, **pt_next);
      arc.emplace_back(arc_lid++, *pt, *pt_next, p, len, s0);
      s0 += len;
    }
  }

  for(auto a = arc.begin(); a != arc.end(); ++a) a->p->arc.push_back(a);
}

void cart::link_poly()
{
  for(auto &p : poly)
  {
    // fill cid_lid poly map
    poly_cid[p.cid] = p.lid;
    p.nF = (node.begin() + node_cid[p.nFcid]);
    p.nT = (node.begin() + node_cid[p.nTcid]);
  }

  // link poly to poly
  for(const auto &n : node)
    for(const auto &ln1 : node_map[n.cid])
      for(const auto &ln2 : node_map[n.cid])
      {
        if( ln1 == ln2 ) continue;
        get_poly_cid(node_poly[n.cid][ln1])->poly.push_back(get_poly_cid(node_poly[n.cid][ln2]));
      }
}

void cart::link_node()
{
  for(auto &n : node)
    for(auto &l : n.link)
    {
      if( *l.second->nF == n )       l.first = l.second->nT;
      else if ( *l.second->nT == n ) l.first = l.second->nF;
      else continue;
    }
}

void cart::make_grid()
{
  grid = grid_base(ilat_max, ilat_min, ilon_max, ilon_min, grid_cell_side, node, arc);
}

void cart::make_noturn()
{

  if (jconf.has_member("file_noturn"))
  {
    std::ifstream noturnf(jconf["file_noturn"].as<std::string>());
    if( !noturnf ) throw std::runtime_error("File forbidden turn not found");
    std::string line;
    std::vector<std::string> tokens;
    std::getline(noturnf, line); // skip header
    while (std::getline(noturnf, line))
    {
      physycom::split(tokens, line, std::string(";"), physycom::token_compress_off);
      int cid1 = stoi(tokens[0]);
      int cid2 = stoi(tokens[1]);
      int lid1 = poly_cid[cid1];
      int lid2 = poly_cid[cid2];
      noturns[lid1].push_back(lid2);
    }

    noturnf.close();
  }
}

void cart::check_network()
{
  if (!skip_check)
  {
    std::map<int, std::map<int, std::set<int>>> connections;
    for (const auto &n: node)
      for (const auto &nl:n.link)
        connections[n.lid][nl.first->lid].insert(nl.second->lid);

    for (const auto &cn: connections)
    {
      for (const auto &nn: cn.second)
      {
        if ( (int(nn.second.size()) == 1) & (cn.first == nn.first) )
        {
          if ( poly[*(nn.second.begin())].length < 0.1 )
          {
            std::stringstream ss;
            ss << "Link " << cn.first << "(" << node[cn.first].cid << ")" << " - " << nn.first << "(" << node[nn.first].cid << ")"  << " is a NULL loop";
            throw check_network_error_loopnull(ss.str());
          }
          else
          {
            std::stringstream ss;
            ss << "Link " << cn.first << "(" << node[cn.first].cid << ")" << " - " << nn.first << "(" << node[nn.first].cid << ")"  << " is a loop";
            throw check_network_error_loop(ss.str().c_str());
          }
        }
        else if (nn.second.size() == 2 && poly[*nn.second.begin()].length==poly[*nn.second.end()].length)
        {
          std::stringstream ss;
          ss << "Link " << cn.first << "(" << node[cn.first].cid << ")" << " - " << nn.first << "(" << node[nn.first].cid << ")"  << " is double";
          throw check_network_error_double(ss.str().c_str());
        }
        else if (nn.second.size() > 2)
        {
          std::set<double> lenght_list;
          for (auto &p:nn.second){
            lenght_list.insert(poly[p].length);
          }
          if (nn.second.size()!=lenght_list.size()){
            std::stringstream ss;
            ss << "Link " << cn.first << "(" << node[cn.first].cid << ")" << " - " << nn.first << "(" << node[nn.first].cid << ") size " << nn.second.size() << " NOT IMPLEMENTED";
            throw check_network_error_multi(ss.str().c_str());
          }
        }
      }
    }
  }
}

void cart::make_tmatrix()
{
  max_cnt = 0;
  std::ifstream input;

  if (jconf.has_member("alpha_we"))
    alpha_we = jconf["alpha_we"].as<double>();

  if (jconf.has_member("file_cnt"))
  {
    input.open(jconf["file_cnt"].as<std::string>());

    if (!input) throw std::runtime_error("Counters file not found.");

    int lid, cid, cntTF, cntFT;
    std::string header;
    std::getline(input, header);
    while (input >> lid >> cid >> cntFT >> cntTF)
    {
      get_poly_cid(cid)->cntFT = (cntFT == 0) ? 1 : cntFT;
      get_poly_cid(cid)->cntTF = (cntTF == 0) ? 1 : cntTF;
      if (cntFT + cntTF > max_cnt) max_cnt = cntFT + cntTF;
    }
    input.close();
    auto cntFT_result = std::minmax_element(poly.begin(), poly.end(), [](const poly_base &p1, const poly_base &p2) {return p1.cntFT < p2.cntFT; });
    auto cntTF_result = std::minmax_element(poly.begin(), poly.end(), [](const poly_base &p1, const poly_base &p2) {return p1.cntTF < p2.cntTF; });
    min_cntFT = cntFT_result.first->cntFT;
    max_cntFT = cntFT_result.second->cntFT;
    min_cntTF = cntTF_result.first->cntTF;
    max_cntTF = cntTF_result.second->cntTF;
    //std::cout <<"FT: " <<min_cntFT << "  " << max_cntFT << std::endl;
    //std::cout <<"TF: " <<min_cntTF << "  " << max_cntTF << std::endl;
  }
  else
  {
    for (auto &p : poly)
    {
      p.cntFT = min_cntFT;
      p.cntTF = min_cntTF;
    }
  }

  // construct transition matrix
  for (const auto &p : poly)
  {
    if (p.nF->link.size() != 1)
    {
      double weight_sum = 0.;
      std::vector<std::pair<int, double>> weights; // <lid, weight>
      for (auto &l : p.nF->link)
      {
        if (l.second->lid == p.lid) continue;
        if (p.nF->isF(l.second)) weights.push_back(std::make_pair(l.second->lid, (double)l.second->cntFT));
        else                     weights.push_back(std::make_pair(l.second->lid, (double)l.second->cntTF));

        weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0, [](double sum, const std::pair<int, double> &p) {
          return sum + p.second;
        });

        if (weight_sum < 1.0)
          for (const auto &k : weights) trans[p.lid][k.first] = 1.0 / weights.size();
        else
          for (const auto &k : weights) trans[p.lid][k.first] = k.second / weight_sum;
      }
    }

    if (p.nT->link.size() != 1)
    {
      double weight_sum = 0.;
      std::vector<std::pair<int, double>> weights; // <lid, weight>
      for (auto &l : p.nT->link)
      {
        if (l.second->lid == p.lid) continue;
        if (p.nT->isF(l.second)) weights.push_back(std::make_pair(l.second->lid, (double)l.second->cntFT));
        else                     weights.push_back(std::make_pair(l.second->lid, (double)l.second->cntTF));

        weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0, [](double sum, const std::pair<int, double> &p) {
          return sum + p.second;
        });

        if( weight_sum < 1.0 )
          for (const auto &k : weights) trans[p.lid][k.first] = 1.0 / weights.size();
        else
          for (const auto &k : weights) trans[p.lid][k.first] = k.second / weight_sum;
      }
    }
  }
}

void cart::make_polyusage()
{
  poly_usage = std::vector<int>(poly.size(), 1);

  try
  {
    if (jconf.has_member("file_polyusage"))
    {
      enum
      {
        //  p.cid; p.lid;   nF.cid; nT.cid; FT.cnt; TF.cnt; tot.cnt;  length
        //      2;     1;  4100832; 133532;      0;      0;       0; 1013.58
        //      0;     1;        2;      3;      4;      5;       6;       7

        OFFSET_CID = 0,
        OFFSET_LID = 1,
        OFFSET_NF = 2,
        OFFSET_NT = 3,
        OFFSET_FLUXFT = 4,
        OFFSET_FLUXTF = 5,
        OFFSET_FLUXTOT = 6,
        OFFSET_LENGTH = 7
      };

      std::ifstream polyusagein(jconf["file_polyusage"].as<std::string>());
      if( !polyusagein ) throw std::runtime_error("File not found");

      std::string line;
      std::vector<std::string> tokens;
      std::getline(polyusagein, line); // skip header
      while (std::getline(polyusagein, line))
      {
        physycom::split(tokens, line, std::string(";"), physycom::token_compress_off);
        auto id = stoi(tokens[OFFSET_LID]);
        auto tot = stoi(tokens[OFFSET_FLUXTOT]);
        poly_usage[id] = tot;
      }

      polyusagein.close();
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "EXC: " << e.what() << " - (Error in parsing poly usage file)" << std::endl;
    std::exit(1);
  }
}

void cart::make_subgraph()
{
  if (!skip_check)
  {
    std::map<std::string, pawn_param> p_type_init;
    pawn_param pp(1.0, 0.5, 0.0, 0.0);
    p_type_init["locals"] = pp;
    update_weight(p_type_init);
    std::vector<int> nodeset(node.size());
    std::iota(nodeset.begin(), nodeset.end(), 0);
    std::set<int> isolated_nodes;

    int subgra_idx = 0;
    while( nodeset.size() )
    {
      auto tmp = dijkstra_explore(nodeset[0], bpweight["locals"]);
      std::vector<int> disco_node;
      for(int i = 0; i < (int)nodeset.size(); ++i)
      {
        if ( std::count_if(bpweight["locals"][nodeset[i]].begin(),
                          bpweight["locals"][nodeset[i]].end(),
                          [](const std::pair<int, double> &w) {
                            return w.second == MAX_WEIGHT;
                          }) == int(node[nodeset[i]].link.size()))
        {
          isolated_nodes.insert(nodeset[i]);
          node_subgra[nodeset[i]] = NO_ROUTE;
        }
        else if ( tmp[nodeset[i]] == NO_ROUTE )
        {
          disco_node.push_back(nodeset[i]);
        }
        else
        {
          subgraph[subgra_idx].push_back(nodeset[i]);
          node_subgra[nodeset[i]] = subgra_idx;
        }
      }

      ++subgra_idx;
      nodeset = disco_node;
    }
    if (isolated_nodes.size())
      subgraph[-1] = std::vector<int>(isolated_nodes.begin(), isolated_nodes.end());
  }
}

void cart::make_bpmatrix()
{
  if (!skip_check)
  {
    try
    {
      std::string parmode = "serial";
      std::map<std::string, pawn_param> p_type_init;
      pawn_param pp(1.0, 0.5, 0.0,0.0);
      p_type_init["locals"] = pp;
      update_weight(p_type_init);
      auto start = std::chrono::steady_clock::now();
      if ( jconf.has_member("explore_node") )
      {
        auto nodev = jconf["explore_node"].as<std::vector<int>>();
        for (int i = 0; i < (int)nodev.size(); ++i)
        {
          bpmatrix["locals"][nodev[i]] = dijkstra_explore(nodev[i], bpweight["locals"]);
        }
      }
      else
      {
        #ifdef _OPENMP
        parmode = "OMP";
        if (n_thread == -1) n_thread = omp_get_max_threads();
        #pragma omp parallel for num_threads(n_thread)
        #endif
        for(int n=0; n < (int)node.size(); ++n)
        {
          auto res = dijkstra_explore(n, bpweight["locals"]);
          #pragma omp critical
          bpmatrix["locals"][n] = res;
        }
      }
      auto end = std::chrono::steady_clock::now();
      std::stringstream ss;
      ss << "explored " << bpmatrix.size() << " (" << parmode << " " << n_thread << ") in : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms";
      bpm_perf = ss.str();
    }
    catch(std::exception &e)
    {
      std::cerr << "EXC: " << e.what() << "\nmake_bproutes() called out of range." << std::endl;
      std::exit(1);
    }
    /*
    // Debug
    std::cout << "SIZE BPMATRIX:  ---> " << bpmatrix.size() << std::endl;
    for (const auto &r : bpmatrix)
      for (int i = 0; i < int(r.second.size()); ++i)
        std::cout << "bp[" << r.first << "][" << i << "] = " << bpmatrix[r.first][i] << std::endl;
    */
  }
}

void cart::update_weight(std::map<std::string, pawn_param> pawn_types)
{
  if (bpweight.size() == 0)
  {
    for (const auto &p : poly)
    {
      switch (p.oneway)
      {
      case ONEWAY_BOTH:
        bpweight["locals"][p.nF->lid][p.nT->lid] = get_weight<FT>(p, 1.0, 0.0);
        bpweight["locals"][p.nT->lid][p.nF->lid] = get_weight<TF>(p, 1.0, 0.0);
        break;
      case ONEWAY_FT:
        bpweight["locals"][p.nF->lid][p.nT->lid] = get_weight<FT>(p, 1.0, 0.0);
        bpweight["locals"][p.nT->lid][p.nF->lid] = MAX_WEIGHT;
        break;
      case ONEWAY_TF:
        bpweight["locals"][p.nF->lid][p.nT->lid] = MAX_WEIGHT;
        bpweight["locals"][p.nT->lid][p.nF->lid] = get_weight<TF>(p, 1.0, 0.0);
        break;
      case ONEWAY_CLOSED:
        bpweight["locals"][p.nF->lid][p.nT->lid] = MAX_WEIGHT;
        bpweight["locals"][p.nT->lid][p.nF->lid] = MAX_WEIGHT;
        break;
      default:
        throw std::runtime_error("BPWeight: oneway not supported " + std::to_string(p.oneway));
        break;
      }
    }
  }
  else
  {
    for (const auto &p : poly)
    {
      switch (p.oneway)
      {
      case ONEWAY_BOTH:
        for (auto &bt : bpweight)
        {
          bt.second[p.nF->lid][p.nT->lid] = get_weight<FT>(p, pawn_types[bt.first].alpha_we, pawn_types[bt.first].alpha_speed);
          bt.second[p.nT->lid][p.nF->lid] = get_weight<TF>(p, pawn_types[bt.first].alpha_we, pawn_types[bt.first].alpha_speed);
        }
        break;
      case ONEWAY_FT:
        for (auto &bt : bpweight)
        {
          bt.second[p.nF->lid][p.nT->lid] = get_weight<FT>(p, pawn_types[bt.first].alpha_we, pawn_types[bt.first].alpha_speed);
          bt.second[p.nT->lid][p.nF->lid] = MAX_WEIGHT;
        }
        break;
      case ONEWAY_TF:
        for (auto &bt : bpweight)
        {
          bt.second[p.nF->lid][p.nT->lid] = MAX_WEIGHT;
          bt.second[p.nT->lid][p.nF->lid] = get_weight<TF>(p, pawn_types[bt.first].alpha_we, pawn_types[bt.first].alpha_speed);
        }
        break;
      case ONEWAY_CLOSED:
        for (auto &bt : bpweight)
        {
          bt.second[p.nF->lid][p.nT->lid] = MAX_WEIGHT;
          bt.second[p.nT->lid][p.nF->lid] = MAX_WEIGHT;
        }
        break;
      default:
        throw std::runtime_error("BPWeight: oneway not supported " + std::to_string(p.oneway));
        break;
      }
    }
  }
  //std::cout << "Size of map_weight: " << bpweight.size() << std::endl;
}

void cart::update_weight_lvlps(std::string user_tag, double alpha_we_tag, const double height_tag, const double alpha_s_tag)
{
  //std::cout << "[cart::update_weight_lvlps] global lvlps " << global_lvl_ps << std::endl;

  for (const auto &p : poly)
  {
    if ( p.lvl_ps - height_tag > global_lvl_ps || p.lvl_ps == 0.0)
    {
      //std::cout <<"user_type "<< user_type << " poly " << p.lid << " is open" << std::endl;
      switch (p.oneway)
      {
      case ONEWAY_BOTH:
        bpweight[user_tag][p.nF->lid][p.nT->lid] = get_weight<FT>(p, alpha_we_tag, alpha_s_tag);
        bpweight[user_tag][p.nT->lid][p.nF->lid] = get_weight<TF>(p, alpha_we_tag, alpha_s_tag);
        break;
      case ONEWAY_FT:
        bpweight[user_tag][p.nF->lid][p.nT->lid] = get_weight<FT>(p, alpha_we_tag, alpha_s_tag);
        bpweight[user_tag][p.nT->lid][p.nF->lid] = MAX_WEIGHT;
        break;
      case ONEWAY_TF:
        bpweight[user_tag][p.nF->lid][p.nT->lid] = MAX_WEIGHT;
        bpweight[user_tag][p.nT->lid][p.nF->lid] = get_weight<TF>(p, alpha_we_tag, alpha_s_tag);;
        break;
      case ONEWAY_CLOSED:
        bpweight[user_tag][p.nF->lid][p.nT->lid] = MAX_WEIGHT;
        bpweight[user_tag][p.nT->lid][p.nF->lid] = MAX_WEIGHT;
        break;
      default:
        throw std::runtime_error("BPWeight: oneway not supported " + std::to_string(p.oneway));
        break;
      }
    }
    else
    {
      //std::cout <<"user_type "<< user_type <<" poly " << p.lid << " is FLOOD **** with lvlps "<< p.lvl_ps << std::endl;
      bpweight[user_tag][p.nF->lid][p.nT->lid] = MAX_WEIGHT;
      bpweight[user_tag][p.nT->lid][p.nF->lid] = MAX_WEIGHT;
    }
  }
  //std::cout << "Size of map_weight: " << bpweight.size() << std::endl;
}

void cart::add_bpmatrix(const std::set<int> &nodedest)
{
  std::vector<int> nodes;
  for (const auto &n : nodedest)
  {
    if (bpmatrix.begin()->second.find(n) == bpmatrix.begin()->second.end())
      nodes.emplace_back(n);
  }
  if (nodes.size())
  {
    #ifdef _OPENMP
    if (n_thread == -1) n_thread = omp_get_max_threads();
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for (int n=0; n < int(nodes.size()); ++n)
    {
      for (auto &i : bpmatrix)
      {
        auto res = dijkstra_explore(nodes[n], bpweight[i.first]);
        #pragma omp critical
        bpmatrix[i.first][nodes[n]] = res;
      }
    }
  }
}

void cart::update_bpmatrix(std::string user_tag)
{
  std::vector<int> nodelist;
  for(const auto &n : bpmatrix[user_tag]) nodelist.emplace_back(n.first);
  bpmatrix[user_tag].clear();

  #ifdef _OPENMP
  if (n_thread == -1) n_thread = omp_get_max_threads();
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for (int n = 0; n < int(nodelist.size()); ++n)
  {
    auto res = dijkstra_explore(nodelist[n], bpweight[user_tag]);
    #pragma omp critical
    bpmatrix[user_tag][nodelist[n]] = res;
  }
}

void cart::dump_poly_geojson(const std::string &basename)
{
  jsoncons::ojson geojson;
  jsoncons::ojson features = jsoncons::ojson::array();
  geojson["type"] = "FeatureCollection";

  for (const auto &p : poly)
  {
    jsoncons::ojson feature;
    jsoncons::ojson coordinates = jsoncons::ojson::parse("[]");
    for (const auto &pt : p.point)
      coordinates.push_back(jsoncons::ojson::parse("[" + std::to_string(pt->ilon * 1e-6) + "," + std::to_string(pt->ilat * 1e-6) + "]"));
    jsoncons::ojson geometry;
    geometry["coordinates"] = coordinates;
    geometry["type"] = "LineString";
    jsoncons::ojson properties;
    properties["poly_lid"] = p.lid;
    properties["poly_cid"] = p.cid;
    properties["poly_length"] = p.length;
    properties["poly_nF"] = p.nF->cid;
    properties["poly_nT"] = p.nT->cid;
    feature["type"] = "Feature";
    feature["properties"] = properties;
    feature["geometry"] = geometry;
    features.push_back(feature);
  }
  geojson["features"] = features;

  std::ofstream outgeoj(basename + ".geojson");
  if (!outgeoj)
    throw std::runtime_error("Unable to create file : " + basename + ".geojson");
  //outgeoj << jsoncons::pretty_print(geojson);
  outgeoj << geojson;
  outgeoj.close();
}

void cart::dump_polyusage()
{
  std::ofstream usage_out("poly_usage_subnet.csv");
  usage_out << "p.cid;p.lid;nF.cid;nT.cid;FT.cnt;TF.cnt;tot.cnt;length" << std::endl;
  for (const auto &p : poly)
  {
    usage_out << p.cid << ";" << p.lid << ";" << p.nF->cid << ";" << p.nT->cid << ";";
    if (p.oneway == 6)
      usage_out << 0 << ";" << 0 << ";" << 0 << ";";
    else
      usage_out << p.tot_count / 2 << ";" << p.tot_count / 2 << ";" << p.tot_count << ";";
    usage_out << p.length << std::endl;
  }
  usage_out.close();
}

void cart::dump_bpmatrix()
{
  for (auto &us_bm : bpmatrix){
    std::cout << "BP MATRIX OF USER TYPE " << us_bm.first << std::endl;
    for (auto &r : bpmatrix[us_bm.first])
    {
      std::cout << "ROUTES TO NODE " << r.first << std::endl;
      for (int i = 0; i < (int)r.second.size(); ++i)
        std::cout << "bp[" << r.first << "][" << i << "] = "
                  << bpmatrix[us_bm.first][r.first][i]
                  << " " << node2poly(bpmatrix[us_bm.first][r.first][i], i)->lid << " " << node2poly(i, bpmatrix[us_bm.first][r.first][i])->lid
                  << std::endl;
    }
  }
}

void cart::patch_cart()
{
  if( jconf.has_member("file_patch") )
  {
    std::ifstream patchfile(jconf["file_patch"].as<std::string>());
    if( !patchfile ) throw std::runtime_error("Patch file not found");

    std::string header; std::getline(patchfile, header);
    int pcid, ow;
    while( patchfile >> pcid >> ow) patch_poly(get_poly_cid(pcid)->lid, ow);
    patchfile.close();
  }
}

void cart::patch_poly(const int &plid, const int &ow)
{
  poly[plid].oneway = ow;
  std::map<std::string, pawn_param> p_type_init;
  pawn_param pp(1.0, 0.5, 0.0,0.0);
  p_type_init["locals"] = pp;
  update_weight(p_type_init);
  update_bpmatrix("locals");

  /*
  // remove
  if ( ow == ONEWAY_TF || ow == ONEWAY_CLOSED )
  {
    auto nF = poly[plid].nF;
    nF->link.erase(
      std::remove_if(
        nF->link.begin(),
        nF->link.end(),
        [plid](std::pair<node_it, poly_it> &p) { return p.second->lid == plid; }
      ),
      nF->link.end()
    );
  }

  if ( ow == ONEWAY_FT || ow == ONEWAY_CLOSED )
  {
    auto nT = poly[plid].nT;
    nT->link.erase(
      std::remove_if(
        nT->link.begin(),
        nT->link.end(),
        [plid](std::pair<node_it, poly_it> &p) { return p.second->lid == plid; }
      ),
      nT->link.end()
    );
  }

  // add
  if ( ow == ONEWAY_TF || ow == ONEWAY_BOTH )
  {
    auto nT = poly[plid].nT;
    if ( std::find_if( nT->link.begin(),
                       nT->link.end(),
                       [plid](std::pair<node_it, poly_it> &p) { return p.second->lid == plid; } )
        == nT->link.end() )
    {
      nT->link.emplace_back(std::make_pair(poly[plid].nF, poly.begin() + plid));
    }
  }

  if ( ow == ONEWAY_FT || ow == ONEWAY_BOTH )
  {
    auto nF = poly[plid].nF;
    if ( std::find_if( nF->link.begin(),
                       nF->link.end(),
                       [plid](std::pair<node_it, poly_it> &p) { return p.second->lid == plid; } )
        == nF->link.end() )
    {
      nF->link.emplace_back(std::make_pair(poly[plid].nT, poly.begin() + plid));
    }
  }

  poly[plid].oneway = ow;
  */
}

pawn_param::pawn_param(double alpha_we_, double beta_bp_, double hight_, double alpha_speed_) {
  alpha_we = alpha_we_;
  beta_bp = beta_bp_;
  hight = hight_;
  alpha_speed = alpha_speed_;
}
