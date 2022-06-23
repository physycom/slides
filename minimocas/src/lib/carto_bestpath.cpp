#include <numeric>

#include <carto.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// UTILS AND DEFS
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int MAX_ITER   = 1000000000;
constexpr int STOP_VALUE = 99999;

std::string cart::bestpath_err(const int &err_code)
{
  std::string errs;
  switch(err_code)
  {
  case BP_PATHFOUND: errs = "path found (" + std::to_string(num_iter) + ")"; break;
  case BP_MAXITER:   errs = "max iter (" + std::to_string(MAX_ITER) + ")";   break;
  case BP_HEAPEMPTY: errs = "heap empty";                                    break;
  case BP_NO_POINTS: errs = "not enough points in input";                    break;
  case BP_MAXWEIGHT: errs = "max weight";                                    break;
  default:           errs = "unknown error code" + std::to_string(err_code); break;
  }
  return errs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// NODE TO NODE
////////////////////////////////////////////////////////////////////////////////////////////////////

// weighted engine
template <>
int cart::dijkstra_engine(const int &nstart, const int &nend, bpweight_t &bpweight)
{
  // initial clean up
  list.clear();
  prev.clear();
  heap = std::priority_queue<miniseed>();
  prev.resize((int)node.size(), 0);
  num_iter = 0;
  dist_cov = 0.0;
  dist_eff = dist_cov + a_eff * distance(node[nstart], node[nend]);

  list.emplace_back(0, 0, STOP_VALUE, 0.0);
  list.emplace_back(nstart, 0, STOP_VALUE, 0.0);
  heap.emplace(int(list.size() - 1), dist_eff);
  bool goal = false;
  // core
  miniseed s;
  int n = -1;
  while (heap.size())
  {
    //std::cout << "BP: " << num_iter << " heap " << heap.size() << " dist_cov: " << dist_cov << std::endl;

    ++num_iter;
    if (num_iter > MAX_ITER) return BP_MAXITER;

    s = heap.top();
    heap.pop();
    n = list[s.id].node;

    if (prev[n] > 0 && list[prev[n]].distance < list[s.id].distance) continue;
    prev[n] = s.id;

    if (n == nend) { goal = true; break; }

    int lid_polybv = list[s.id].link_back;

    // std::cout << "Node " << n << " link " << node[n].link.size() << std::endl;
    for (const auto &i : node[n].link)
    {
      //avoid choosing no turns poly
      if (noturns.find(lid_polybv)!=noturns.end())
        if(std::find(noturns[lid_polybv].begin(), noturns[lid_polybv].end(), i.second->lid) != noturns[lid_polybv].end())
          continue;

      dist_cov = list[prev[n]].distance + bpweight[node[n].lid][i.first->lid];
      if (prev[i.first->lid] > 0 && list[prev[i.first->lid]].distance < dist_cov) continue;

      list.emplace_back(i.first->lid, n, i.second->lid, dist_cov);
      dist_eff = dist_cov + a_eff * distance(*i.first, node[nend]);
      heap.emplace(int(list.size() - 1), dist_eff);
    }
  }
  if (heap.size() == 0 && !goal) return BP_HEAPEMPTY;
  if (list[prev[n]].distance >= MAX_WEIGHT) return BP_MAXWEIGHT;

  return BP_PATHFOUND;
}

template <>
int cart::dijkstra_engine(const int &nstart, const int &nend)
{
  // initial clean up
  list.clear();
  prev.clear();
  heap = std::priority_queue<miniseed>();
  prev.resize((int)node.size(), 0);
  num_iter = 0;
  dist_cov = 0.0;
  dist_eff = dist_cov + a_eff * distance(node[nstart], node[nend]);

  list.emplace_back(0, 0, STOP_VALUE, 0.0);
  list.emplace_back(nstart, 0, STOP_VALUE, 0.0);
  heap.emplace(int(list.size()-1), dist_eff);

  // core
  miniseed s;
  int n = -1;
  while( heap.size() )
  {
    //std::cout << "BP: " << num_iter << " heap " << heap.size() << std::endl;

    ++num_iter;
    if (num_iter > MAX_ITER) return BP_MAXITER;

    s = heap.top();
    heap.pop();
    n = list[s.id].node;

    if (prev[n] > 0 && list[prev[n]].distance < list[s.id].distance) continue;
    prev[n] = s.id;

    if (n == nend) break;

    //std::cout << "Node " << n << " link " << node[n].link.size() << std::endl;
    for (const auto &i : node[n].link)
    {
      dist_cov = list[prev[n]].distance + i.second->length;
      //dist_cov = list[prev[n]].distance + (i.second->length*pow(max_cnt / double(i.second->cntFT + i.second->cntTF), 0.25));

      if (prev[i.first->lid] > 0 && list[prev[i.first->lid]].distance < dist_cov) continue;

      list.emplace_back(i.first->lid, n, i.second->lid, dist_cov);
      dist_eff = dist_cov + a_eff * distance(*i.first, node[nend]);
      heap.emplace(int(list.size()-1), dist_eff);
    }
  }
  if (heap.size() == 0) return BP_HEAPEMPTY;

  return BP_PATHFOUND;
}

template <>
int cart::dijkstra_engine(const point_base &start, const point_base &end, path_point &path)
{
  // initial clean up
  list.clear();
  prev.clear();
  heap = std::priority_queue<miniseed>();
  prev.resize((int)node.size(), 0);
  num_iter = 0;

  // geometry for start point
  path.ptstart.a = get_nearest_arc(start);
  path.ptstart = project(start, path.ptstart.a);
  double dF1, dT1, deu;
  deu = distance(path.ptstart, end);
  dF1 = path.ptstart.a->s + path.ptstart.s;
  dT1 = path.ptstart.a->p->length - dF1;

  // geometry for end point
  path.ptend.a = get_nearest_arc(end);
  path.ptend = project(end, path.ptend.a);
  int n2F = path.ptend.a->p->nF->lid;
  int n2T = path.ptend.a->p->nT->lid;
  double dF2 = path.ptend.a->s + path.ptend.s;
  double dT2 = path.ptend.a->p->length - dF1;

  // init
  list.emplace_back(0, 0, STOP_VALUE, 0.0);
  list.emplace_back(path.ptstart.a->p->nF->lid, 0, STOP_VALUE, dF1);
  heap.emplace(int(list.size()-1), dF1 + deu);
  list.emplace_back(path.ptstart.a->p->nT->lid, 0, STOP_VALUE, dT1);
  heap.emplace(int(list.size()-1), dT1 + deu);

  // core
  miniseed s;
  int n = -1;
  while( heap.size() )
  {
    ++num_iter;
    if (num_iter > MAX_ITER) return BP_MAXITER;

    s = heap.top(); heap.pop();
    n = list[s.id].node;

    if (prev[n] > 0 && list[prev[n]].distance < list[s.id].distance) continue;
    prev[n] = s.id;

    if (prev[n2F] > 0 && prev[n2T] > 0) break;

    for (const auto &i : node[n].link)
    {
      dist_cov = list[prev[n]].distance + i.second->length;
      list.emplace_back(i.first->lid, n, i.second->lid, dist_cov);

      dist_eff = dist_cov + a_eff * distance(*i.first, end);
      heap.emplace(int(list.size()-1), dist_eff);
    }

    if (heap.size() == 0) return BP_HEAPEMPTY;
  }

  // final distance adjust and actual best path selection
  dF2 += list[prev[n2F]].distance;
  dT2 += list[prev[n2T]].distance;
  int nend = (dF2 < dT2) ? n2F : n2T;

  // fill path
  int nlid = nend;
  int plid = list[prev[nend]].link_back;
  path_point lpath;
  while (plid != STOP_VALUE)
  {
    lpath.poly.push_back( poly.begin() + plid );
    nlid = list[prev[nlid]].node_back;
    plid = list[prev[nlid]].link_back;
  }
  std::reverse(lpath.poly.begin(), lpath.poly.end());
  path.poly.insert(path.poly.end(), lpath.poly.begin(), lpath.poly.end());

  return BP_PATHFOUND;
}

// explore
template <>
std::vector<int> cart::dijkstra_explore(const int &nstart, bpweight_t &bpweight)
{
  int num_iter;
  double dist_cov;
  std::vector<seed> list;
  std::vector<int> prev;
  std::priority_queue<miniseed> heap;

  // initial clean up
  list.clear();
  prev.clear();
  heap = std::priority_queue<miniseed>();
  prev.resize((int)node.size(), 0);

  num_iter = 0;
  dist_cov = 0.0;

  list.emplace_back(0, 0, STOP_VALUE, 0.0);
  list.emplace_back(nstart, 0, STOP_VALUE, 0.0);
  heap.emplace(int(list.size() - 1), dist_cov);

  // core
  miniseed s;
  int n = -1;
  while (heap.size())
  {
    ++num_iter;
    //std::cout << nstart << ") size " << heap.size() << " node " << n << std::endl;
    if (num_iter > MAX_ITER) { 
      std::cout << "Dijkstra Explore: Max iter reached" << std::endl; 
      break; 
    }

    s = heap.top();
    heap.pop();
    n = list[s.id].node;
    if (prev[n] > 0 && list[prev[n]].distance < list[s.id].distance) continue;
    prev[n] = s.id;

    int lid_polybv = list[s.id].link_back;
    
    for (const auto &i : node[n].link)
    {
      //avoid choosing no turns poly
      if (noturns.find(lid_polybv)!=noturns.end())
        if(std::find(noturns[lid_polybv].begin(), noturns[lid_polybv].end(), i.second->lid) != noturns[lid_polybv].end())
          continue;

      //dist_cov = list[prev[n]].distance + bpweight[node[n].lid][i.first->lid];
      dist_cov = list[prev[n]].distance + bpweight[i.first->lid][node[n].lid]; // probable fix for reverse logic in explore logic (from end to start, instead of from start to end)
      if (prev[i.first->lid] > 0 && list[prev[i.first->lid]].distance < dist_cov) continue;
      list.emplace_back(i.first->lid, n, i.second->lid, dist_cov);
      heap.emplace(int(list.size() - 1), dist_cov);
    }
  }

  // collect path as node_lid sequence
  std::vector<int> nodeseq(node.size(), NO_ROUTE);
  for (int i = 0; i < (int)node.size(); ++i)
  {
    if (prev[i] != 0 && list[prev[i]].distance < MAX_WEIGHT)
    {
      nodeseq[i] = list[prev[i]].node_back;
    }
  }

  return nodeseq;
}
template <> std::vector<int> cart::dijkstra_explore(const node_base &n, bpweight_t &bpweight) { return dijkstra_explore(n.lid, bpweight); }
template <> std::vector<int> cart::dijkstra_explore(const node_it &n, bpweight_t &bpweight) { return dijkstra_explore(n->lid, bpweight); }

template <>
std::vector<int> cart::dijkstra_explore(const int &nstart)
{
  int num_iter;
  double dist_cov;
  //double dist_eff, a_eff;
  std::vector<seed> list;
  std::vector<int> prev;
  std::priority_queue<miniseed> heap;

  // initial clean up
  list.clear();
  prev.clear();
  heap = std::priority_queue<miniseed>();
  prev.resize((int)node.size(), 0);

  num_iter = 0;
  dist_cov = 0.0;

  list.emplace_back(0, 0, STOP_VALUE, 0.0);
  list.emplace_back(nstart, 0, STOP_VALUE, 0.0);
  heap.emplace(int(list.size() - 1), dist_cov);

  // core
  miniseed s;
  int n = -1;
  while (heap.size())
  {
    ++num_iter;
    //std::cout << nstart << ") size " << heap.size() << " node " << n << std::endl;
    if (num_iter > MAX_ITER) { std::cout << "Dijkstra Explore: Max iter reached" << std::endl; break; }

    s = heap.top();
    heap.pop();
    n = list[s.id].node;

    if (prev[n] > 0 && list[prev[n]].distance < list[s.id].distance) continue;
    prev[n] = s.id;

    for (const auto &i : node[n].link)
    {
      dist_cov = list[prev[n]].distance + i.second->length;
      //dist_cov = list[prev[n]].distance + (i.second->length*std::pow(max_cnt / double(i.second->cntFT + i.second->cntTF), 0.25));

      if (prev[i.first->lid] > 0 && list[prev[i.first->lid]].distance < dist_cov) continue;

      list.emplace_back(i.first->lid, n, i.second->lid, dist_cov);
      heap.emplace(int(list.size() - 1), dist_cov);
    }
  }

  // collect path as node_lid sequence
  std::vector<int> nodeseq(node.size(), NO_ROUTE);
  for (int i = 0; i < (int)node.size(); ++i)
  {
    if (prev[i] != 0)
      nodeseq[i] = list[prev[i]].node_back;
  }

  return nodeseq;
}
template <> std::vector<int> cart::dijkstra_explore(const node_base &n) { return dijkstra_explore(n.lid); }
template <> std::vector<int> cart::dijkstra_explore(const node_it &n) { return dijkstra_explore(n->lid); }

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// BESTPATH INTERFACES
////////////////////////////////////////////////////////////////////////////////////////////////////

// best path node_lid to node_lid populating path as std::vector<int> poly_lid
template <>
int cart::bestpath(const int &nstart, const int &nend, std::vector<int> &path)
{
  int ret = dijkstra_engine(nstart, nend, bpweight["locals"]);
  if( ret == BP_PATHFOUND )
  {
    int n_lid = nend;
    int poly_lid = list[prev[nend]].link_back;
    while (poly_lid != STOP_VALUE)
    {
      path.push_back(poly_lid);
      n_lid = list[prev[n_lid]].node_back;
      poly_lid = list[prev[n_lid]].link_back;
    }
    std::reverse(path.begin(), path.end());
  }
  return ret;
}
template<> int cart::bestpath(const node_base &n_start, const node_base &n_end, std::vector<int> &polybp) { return bestpath(n_start.lid, n_end.lid, polybp); }
template<> int cart::bestpath(const node_it &n_start, const node_it &n_end, std::vector<int> &polybp) { return bestpath(n_start->lid, n_end->lid, polybp); }

template<>
int cart::bestpath(const node_base &start, const node_base &end, path_base &path)
{
  int ret = dijkstra_engine(start.lid, end.lid, bpweight["locals"]);
  if( ret == BP_PATHFOUND )
  {
    int n_lid = end.lid;
    int poly_lid = list[prev[end.lid]].link_back;
    while (poly_lid != STOP_VALUE)
    {
      path.poly.push_back( poly.begin() + poly_lid );
      n_lid = list[prev[n_lid]].node_back;
      poly_lid = list[prev[n_lid]].link_back;
    }
    std::reverse(path.poly.begin(), path.poly.end());
  }
  return ret;
}

// oriented output (true FT, false TF) < ( lid, bool ) ... >
template <>
int cart::bestpath(const int &nstart, const int &nend, std::vector<std::pair<int, bool>> &polybp)
{
  int ret = dijkstra_engine(nstart, nend, bpweight["locals"]);
  if( ret == BP_PATHFOUND )
  {
    int nlid = nend;
    int plid = list[prev[nend]].link_back;
    while (plid != STOP_VALUE)
    {
      if ( node[nlid].isT(poly[plid]) ) polybp.push_back(std::make_pair(plid, true));
      else                              polybp.push_back(std::make_pair(plid, false));
      nlid = list[prev[nlid]].node_back;
      plid = list[prev[nlid]].link_back;
    }
    std::reverse(polybp.begin(), polybp.end());
  }
  return ret;
}
template<> int cart::bestpath(const node_base &start, const node_base &end, std::vector<std::pair<int, bool>> &polybp) { return bestpath(start.lid, end.lid, polybp); }

// bestpath point_base to point_base
template<>
int cart::bestpath(const point_base &start, const point_base &end, path_point &path)
{
  return dijkstra_engine(start, end, path);
}

// list of input node_lid
template<>
int cart::bestpath(const std::vector<int> &node_seq, std::vector<int> &path)
{
  int ret = BP_NO_POINTS;
  for(int i=0; i<int(node_seq.size())-1; ++i)
  {
    std::vector<int> path_chunk;
    ret = bestpath(node_seq[i], node_seq[i+1], path_chunk);
    if (ret == BP_PATHFOUND)
    {
      path.insert(path.end(), path_chunk.begin(), path_chunk.end());
    }
  }
  return ret;
}

// output as std::vector<path_t> wrapper
template<typename T, typename path_t>
int cart::bestpath(const T &start, const T &end, std::vector<path_t> &path_container)
{
  path_t path;
  int ret = bestpath(start, end, path);
  if( ret == BP_PATHFOUND ) path_container.push_back(path);
  return ret;
}
template<> int cart::bestpath(const node_it &start, const node_it &end, std::vector<path_base> &pathv) { return bestpath(*start, *end, pathv);}
template int cart::bestpath(const point_base &start, const point_base &end, std::vector<path_point> &pathv);
template int cart::bestpath(const node_base &start, const node_base &end, std::vector<path_base> &pathv);

template<>
int cart::bestpath(const std::vector<point_base> &points, std::vector<path_point> &pathv)
{
  int ret;
  path_point path;
  ret = bestpath(points[0], points[1], path);
  auto ptstart = path.ptstart;
  path.poly.push_back(path.ptend.a->p);
  for(int i = 1; i < (int)points.size()-2; ++i)
  {
    ret = bestpath(points[i], points[i+1], path);
    if ( ret == BP_PATHFOUND ) path.poly.push_back(path.ptend.a->p);
  }
  ret = bestpath(points[(int)points.size() - 2], points[(int)points.size() - 1], path);
  path.ptstart = ptstart;
  pathv.push_back(path);
  return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////// ROUTE GENERATOR
////////////////////////////////////////////////////////////////////////////////////////////////////

void cart::dump_routes(const std::string &filename)
{
  char sep = ';', nsep = '-';
  std::ofstream outroute(filename);
  outroute << "#origin" << std::endl;
  for(const auto i : origin_An) outroute << i.first << sep << i.second << std::endl;
  outroute << "#poi" << std::endl;
  for(const auto i : poi_An) outroute << i.first << sep << i.second << std::endl;
  outroute << "route_id;node_lid_seq;dist[m]" << std::endl;
  for(const auto &r : routes)
  {
    outroute << r.first << sep;
    for(int i=0; i< int(r.second.dest.size()); ++i)
      outroute << r.second.dest[i] << ( i == int(r.second.dest.size()-1) ? sep : nsep );
    outroute << r.second.dist << std::endl;
  }
  outroute.close();
}
