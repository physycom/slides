/*!
 *  \file   carto.h
 *  \author A. Fabbri (alessandro.fabbri27@unibo.it), C. Mizzi (chiara.mizzi2@unibo.it), S. Sinigardi (stefano.sinigardi@unibo.it)
 *  \brief  Declaration file for #cart.
 *  \details This file contains the definitions of classes which constitute the bulk of the cartography object.
 */

#ifndef _MINIMOCAS_CARTO_H_
#define _MINIMOCAS_CARTO_H_

#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <queue>
#include <list>
#include <set>

#include <jsoncons/json.hpp>

#include <physycom/geometry.hpp>

#define VERBOSE_DEBUG 0 // 0 OFF, 1 ON
#if VERBOSE_DEBUG
#define MESSAGE(...)     std::fprintf(stdout, __VA_ARGS__)
#define MESSAGE_ERR(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define MESSAGE(...)
#define MESSAGE_ERR(...)
#endif

// Forward Declaration
struct poly_base;
struct node_base;
struct point_base;
struct arc_base;
struct cart;

// Handy typedefs
typedef std::vector<poly_base>::iterator  poly_it;  /*!< \brief Shorthand typedef for `std::vector<poly_base>::iterator`. */
typedef std::vector<node_base>::iterator  node_it;  /*!< \brief Shorthand typedef for `std::vector<node_base>::iterator`. */
typedef std::vector<point_base>::iterator point_it; /*!< \brief Shorthand typedef for `std::vector<point_base>::iterator`. */
typedef std::vector<arc_base>::iterator   arc_it;   /*!< \brief Shorthand typedef for `std::vector<arc_base>::iterator`. */

typedef std::map<int, std::map<int, double>> bpweight_t; /*!< \brief Shorthand typedef for BestPath Weight matrix, i.e. a sparse weighted adjacency list in the form of a `std::map<std::map<int, double>`. */
//typedef std::map<int,map<int, std::map<int, double>>> bpweight_t; /*!< \brief Shorthand typedef for BestPath Weight matrix, i.e. a sparse weighted adjacency list in the form of a `std::map<std::map<int, double>`. */

/////////////////////////////////////////////////////////////////////////////////////
// SUPPORT CLASSES
/////////////////////////////////////////////////////////////////////////////////////

// units conversion
constexpr double kmh_to_ms = 1 / 3.6;     /*!< \brief Speed conversion factor from km/h to m/s. */

// geo tools
#define IDEG_TO_RAD        1.745329e-8    /*!< \brief Conversion factor from integer degrees to radians. */
#define DSLAT              0.1110704      /*!< \brief Conversion factor from integer degrees to meters, on geodesic circle. */
#define DEFAULT_CELL_SIDE  50.0           /*!< \brief Default value for #grid_base cell side (in meters). */

/*! \brief Generic geodetic distance template function.
 *  \details Generic template for the evaluation of the geodetic distance on Earth's surface, using the local tangent plane approximation. It accepts a pair, not necessarily of the same type provided they expose an integer pair of `lat` and `lon` members.
 * Given a pair of geo-localized objects whose geo-coordinates let their (lat,lon) coordinates be \f$(\phi_1, \theta_1)\f$ and \f$(\phi_2, \theta_2)\f$.
 *
 *  \return a double representing the approximated distance in meters.
 */
template<typename T, typename U> double distance(const T &t, const U &u);

/*! \struct point_base
 *  \brief Base class for point on Earth's surface
 *  \details Base class for description of a generic point on Earth's surface, the values of coordinates are in integer lat/lon obtained by scaling the usual double value by \f$10^6\f$.
 */
struct point_base
{
  int ilat; /*!< \brief Integer latitude coordinate */
  int ilon; /*!< \brief Integer longitude coordinate */

  point_base();
  point_base(const int &ilat, const int &ilon);

  friend std::ostream& operator<< (std::ostream& stream, const point_base& pt);
  inline bool operator==(const point_base &pt) const { return (pt.ilat == ilat && pt.ilon == ilon); };
  inline bool operator!=(const point_base &pt) const { return !(pt == *this); };
};

// physycom::geometry compatibility specialization
template <>
auto physycom::to_coords_tuple<point_base>(point_base &pt)
{
  return std::make_tuple(&pt.ilon, &pt.ilat);
}

/*! \struct point_proj
 *  \brief Projected point class.
 *  \details A class for description of a generic point on Earth's surface together with its projection onto the cartography system.
 */
struct point_proj : public point_base
{
  arc_it a; /*!< \brief Iterator to the arc onto which the point is projected. */
  double s; /*!< \brief Poly's arc lenght, describing the position of the projected point. */
  point_proj();
  point_proj operator=(const point_base &pt);
};

/*! \struct arc_base
 *  \brief Base class for _arc_ objects.
 *  \details Base class containing the information for the base _arc_ object which acts as building block for polylines, see #poly_base.
 */
struct arc_base
{
  int lid;       /*!< \brief Arc local id. */
  double length; /*!< \brief Arc lenght, in meters. */
  double s;      /*!< \brief Arc lenght, along the polyline, of Front point. */
  double ux;     /*!< \brief Arc unit versor, \f$x\f$ component. */
  double uy;     /*!< \brief Arc unit versor, \f$x\f$ component. */
  point_it ptF;  /*!< \brief #point_it iterator to arc's Front point. */
  point_it ptT;  /*!< \brief #point_it iterator to arc's Tail point. */
  poly_it p;     /*!< \brief #poly_it to arc's belonging #poly_base. */
  arc_base(const int &_lid, const point_it &_ptF, const point_it &_ptT, const poly_it &_p, const double &_length, const double &_s);
};

// polyline base class
enum
{
  ONEWAY_BOTH   = 0,
  ONEWAY_TF     = 1,
  ONEWAY_FT     = 2,
  ONEWAY_CLOSED = 3
};

enum
{
  TF = 0,
  FT
};

struct poly_base
{
  unsigned long long int cid;
  unsigned long long int nFcid, nTcid;
  int lid;
  int type, oneway;
  node_it nF, nT;
  double length;
  double speed;
  std::string name;
  std::vector<point_it> point;
  std::vector<poly_it> poly;
  std::vector<arc_it> arc;
  int cntFT, cntTF;
  bool ok_edit;

  // poly stats
  int tot_count;
  std::map<int, std::pair<std::set<int>, std::set<int>>> uniq_pwn; //<time_idx, <cnt_FT, cnt_TF>>

  // pip stuff
  double speed_t;
  int counter;
  double density;

  // water level PS (Punta Salute) [cm]
  double lvl_ps;

  // constructors
  poly_base();
  poly_base(const unsigned long long int &_cid, const unsigned long long int &_nFcid, const unsigned long long int &_nTcid);
  point_base get_point(const double &s);

  // methods
  template<int ORIENTATION> double weight() const;
};

// Node
struct node_base
{
  int lid;                          // local_id
  unsigned long long int cid;       // cartography_id
  int ilat, ilon;                   // deg * 1e6
  std::vector<std::pair<node_it, poly_it>> link;
  bool stops = false;
  double score = 1.0;
  std::string city_membership;

  template<typename T> bool isF(const T &t) const;
  template<typename T> bool isT(const T &t) const;
  inline bool operator==(const node_base &n) const { return lid == n.lid; }
  inline bool operator!=(const node_base &n) const { return !(*this == n); }
};

// Cell
struct cell_base
{
  std::vector<arc_it> arc;
  std::vector<node_it> node;
  int ilatcen, iloncen, ilatbot, ilonleft;
  cell_base();
};

// Grid
struct grid_base
{
  int grow, gcol;   // grid size
  double gside;     // grid cell size [m]
  int gdlat, gdlon; // grid cell size [deg]
  int gilatmax, gilatmin, gilonmax, gilonmin;

  grid_base();
  grid_base(const int &ilat_max, const int &ilat_min, const int &ilon_max, const int &ilon_min, const double &cell_side, std::vector<node_base> &node, std::vector<arc_base> &arc);

  inline void coord_to_grid(const int &ilat, const int &ilon, int &row, int &col)
  {
    row = int((ilat - gilatmin) / gdlat);
    col = int((ilon - gilonmin) / gdlon);
    if( row < 1 ) row = 1;
    if( row > grow - 2 ) row = grow - 2;
    if( col < 1 ) col = 1;
    if( col > gcol - 2 ) col = gcol - 2;
  }

  void dump_geojson(const std::string &filename);

  ///////////
  inline cell_base& operator()(const int &ilat, const int &ilon)
  {
    int row, col;
    coord_to_grid(ilat, ilon, row, col);
    return grid[row][col];
  }
  inline cell_base& operator()(const point_base &pt) { return operator()(pt.ilat, pt.ilon); }

  // operator[][] overload
  class proxy
  {
  public:
    proxy(std::vector<cell_base> _row) : row(_row) {}
    cell_base operator[](int col) { return row[col]; }
  private:
    std::vector<cell_base> row;
  };
  proxy operator[](int row) { return proxy(grid[row]); }

  // to enable range for
  inline std::vector<std::vector<cell_base>>::iterator begin() { return grid.begin(); }
  inline std::vector<std::vector<cell_base>>::iterator end() { return grid.end(); }
  inline std::vector<cell_base> front() { return grid.front(); }

  std::vector<std::vector<cell_base>> grid;
};

/////////////////////////////////////////////////////////////////////////////////////
// BEST PATH SUPPORT CLASSES
/////////////////////////////////////////////////////////////////////////////////////

constexpr int NO_ROUTE = -1;
constexpr double VMIN_BP = 20 / 3.6;

enum
{
  BP_PATHFOUND   = 1,
  BP_MAXITER     = 2,
  BP_HEAPEMPTY   = 3,
  BP_GEOLOCFAIL  = 4,
  BP_NO_POINTS   = 5,
  BP_MAXWEIGHT   = 6
};

struct miniseed
{
  int id;
  double x;
  miniseed() : id(0), x(0.) {};
  miniseed(int _id, double _x) : id(_id), x(_x) {};
  inline bool operator<(const miniseed &s) const { return x > s.x; }   // ascending sorted queue
};

class seed
{
public:
  double distance;
  int node, node_back, link_back;
  seed();
  seed(int _node, int _node_back, int _link_back, double _distance) :
    distance(_distance),
    node(_node),
    node_back(_node_back),
    link_back(_link_back) {}
};

/////////////////////////////////////////////////////////////////////////////////////
// ROUTE GENERATOR
/////////////////////////////////////////////////////////////////////////////////////

constexpr int node_origin = 0;
constexpr int node_poi = 1;

constexpr int default_max_poi = 5;

struct route_t
{
  double dist;
  std::string id;
  std::vector<int> dest;
};

struct path_base
{
  std::vector<poly_it> poly;

  inline double length()
  {
    double len = 0;
    for(const auto &p : poly) len += p->length;
    return len;
  }
};

struct path_point : public path_base
{
  point_proj ptstart, ptend;

  inline double len()
  {
    return path_base::length();
  }
};

/////////////////////////////////////////////////////////////////////////////////////
// PAWN PARAM CLASS
/////////////////////////////////////////////////////////////////////////////////////

struct pawn_param {
  double alpha_we;
  double alpha_speed;
  double beta_bp;
  double hight;
  pawn_param() {};
  pawn_param(double alpha_we_, double beta_bp_, double hight_, double alpha_speed_);
};

/////////////////////////////////////////////////////////////////////////////////////
// CONSISTENCY CHECKS
/////////////////////////////////////////////////////////////////////////////////////

class check_network_error : public std::runtime_error
{
public:
  check_network_error(std::string msg) : std::runtime_error(msg) {}
};

class check_network_error_loopnull : public check_network_error
{
public:
  check_network_error_loopnull(std::string msg) : check_network_error(msg) {}
};

class check_network_error_loop : public check_network_error
{
public:
  check_network_error_loop(std::string msg) : check_network_error(msg) {}
};

class check_network_error_double : public check_network_error
{
public:
  check_network_error_double(std::string msg) : check_network_error(msg) {}
};

class check_network_error_multi : public check_network_error
{
public:
  check_network_error_multi(std::string msg) : check_network_error(msg) {}
};


/////////////////////////////////////////////////////////////////////////////////////
// CARTOGRAPHY CLASS
/////////////////////////////////////////////////////////////////////////////////////
constexpr double MAX_WEIGHT = 1e8;
constexpr int NO_LVL_PS = -1000;

struct cart
{
  jsoncons::json jconf;
  std::vector<poly_base> poly;
  std::vector<node_base> node;
  std::vector<point_base> point;
  std::vector<arc_base> arc;
  grid_base grid;

  // poly counters
  int max_cnt;
  int max_cntFT;
  int min_cntFT;
  int max_cntTF;
  int min_cntTF;

  // grid
  int grid_cell_side;

  // forbidden turn
  std::map<int,std::vector<int>> noturns;

  // water level
  int global_lvl_ps;
  double alpha_we;

  // control points coordinates
  int ilat_min;
  int ilat_max;
  int ilat_center;
  int ilon_min;
  int ilon_max;
  int ilon_center;

  // toggle for network consistency checks and bp-related init operations
  bool skip_check;

  // methods
  cart();
  cart(jsoncons::json _jconf, bool skip_check = false);

  // utilities
  inline poly_it get_poly_cid(const unsigned long long int &cid) { return ( poly.begin() + poly_cid[cid] ); }
  inline node_it get_node_cid(const unsigned long long int &cid) { return ( node.begin() + node_cid[cid] ); }
  inline void get_node_cell(const node_base &n, int &row, int &col)
  {
    row = (n.ilat - grid.gilatmin) / grid.gdlat;
    col = (n.ilon - grid.gilonmin) / grid.gdlon;
  }
  inline void get_node_cell(const node_it &n, int &row, int &col) { get_node_cell(*n, row, col); }
  inline bool is_in_bbox(const point_base &pt)
  {
    return pt.ilat >= ilat_min && pt.ilat <= ilat_max && pt.ilon >= ilon_min && pt.ilon <= ilon_max;
  }
  template<typename T> node_it get_nearest_node(const T &t);
  template<typename T> arc_it get_nearest_arc(const T &t);
  point_proj project(const point_base &pt, const arc_it &a);
  std::list<unsigned long long> get_poly_insquare(const int &lat, const int &lon, const double &side);
  inline poly_it node2poly(const int &n1, const int &n2)
  {
    return poly.begin() + poly_cid.at(node_poly.at(node[n1].cid).at(node[n2].cid));
  }
  template <int ORIENTATION> double get_weight(const poly_base &p, const double alpha_we, const double alpha_s) const;

  // poly usage
  std::vector<int> poly_usage;
  void dump_polyusage();
  void dump_poly_geojson(const std::string &basename);

  // best path
  int n_thread;
  std::map<std::string, std::map<int, std::map<int, double>>> bpweight;       // map < user_type, < node_lid, node_lid, weight >>>
  std::unordered_map<int, std::unordered_map<int, double>> trans;             // < poly_lid, poly_lid, weight >
  std::map<std::string, std::unordered_map<int, std::vector<int>>> bpmatrix;  // map <user_type, < node_lid dest, [ node_lid ] >>
  void update_weight(std::map<std::string, pawn_param> pawn_types);
  void update_weight_lvlps(std::string user_tag, const double alpha_we_tag, const double height_tag, const double alpha_s_tag);
  void add_bpmatrix(const std::set<int> &nodedest);
  void update_bpmatrix(std::string user_tag);
  void dump_bpmatrix();

  template<typename T, typename U, typename W> int bestpath(const T &start, const U &end, W &path);
  template<typename T, typename U> int bestpath(const T &points, U &path);
  template<typename T, typename U> int bestpath(const T &start, const T &end, std::vector<U> &vec_path);

  template<typename T, typename U> int dijkstra_engine(const T &t, const U &u);
  template<typename T, typename U> int dijkstra_engine(const T &start, const U &end, bpweight_t &bpw);
  template<typename T, typename U, typename W> int dijkstra_engine(const T &start, const U &end, W &path);

  template<typename T> std::vector<int> dijkstra_explore(const T &nstart, bpweight_t &bpweight);
  template<typename T> std::vector<int> dijkstra_explore(const T &nstart);

  std::string bestpath_err(const int &err_code);

  //route
  std::map<int, char> origin_nA, poi_nA;
  std::map<char, int> origin_An, poi_An;
  std::map<std::string, route_t> routes;
  void dump_routes(const std::string &filename);

  // subgraph variables
  std::map<int, std::vector<int>> subgraph;
  std::map<int, int> node_subgra;

  // edit methods
  enum
  {
    NODE_PENDING = 10,
    NODE_DEG2,
    NODE_SUBGRAPH
  };
  std::map<int, std::vector<int>> degnode;
  std::vector<bool> edit_polylist;
  std::map<int, int> edit_nodelist;
  void identify_nodes();
  void edit_degree2();
  void remove_degree2();
  void update_degree();
  void attach_nodes();
  void remove_shortp();
  void remove_pending();
  void merge_subgraph();
  int find_intersection(point_base p0, point_base p1, point_base b0, point_base b1);
  void assign_level_ps(const std::string &grid_filename);
  void assign_level_bridge(const std::string &bridge_filename);
  void dump_noturn_file(const std::string &noturn_filename);
  void reduce_area(const jsoncons::json &jconf);
  void remove_subgraph(const jsoncons::json &jconf);
  void remove_doubleconnections();
  void dump_edited();
  void dump_test_config();
  void dump_edit_config(std::string &cartout);

  std::string edit_node_code(const int &code);

  // init core functions
  void load_poly();
  void make_node();
  void make_arc();
  void link_poly();
  void link_node();
  void make_grid();
  void make_noturn();
  void check_network();
  void make_tmatrix();
  void make_bpmatrix();
  void make_polyusage();
  void make_subgraph();
  std::string info();

  // internal functions
  void patch_cart();
  void patch_poly(const int &plid, const int &ow);

  // best path vars
  int num_iter;
  double dist_cov, dist_eff, a_eff;
  std::vector<seed> list;
  std::vector<int> prev;
  std::priority_queue<miniseed> heap;
  bool first = true;
  std::string bpm_perf;

  // name and connection maps
  std::unordered_map<unsigned long long int, int> node_cid;                                                                 // < cid, lid >
  std::unordered_map<unsigned long long int, int> poly_cid;                                                                 // < cid, lid >
  std::unordered_map<unsigned long long int, std::unordered_set<unsigned long long int>> node_map;                          // < cid, list of connected cids >
  std::unordered_map<unsigned long long int, std::unordered_map<unsigned long long int, unsigned long long int>> node_poly; // < node cid, < node cid, poly cid > >
};

#endif // _MINIMOCAS_CARTO_H_
