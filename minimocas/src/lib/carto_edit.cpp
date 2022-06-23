#include <carto.h>

#include <physycom/geometry.hpp>
#include <physycom/string.hpp>


constexpr double pending_len_thresh = 10.; // meters

std::string cart::edit_node_code(const int &code)
{
  std::string s;
  switch (code)
  {
  case NODE_PENDING:
    s = "Pending(" + std::to_string(int(pending_len_thresh)) + "m)";
    break;
  case NODE_DEG2:
    s = "Degree2";
    break;
  case NODE_SUBGRAPH:
    s = "Subgraph";
    break;
  default:
    s = "Unknown";
    break;
  }
  return s;
}

void cart::update_degree()
{
  degnode.clear();
  for (const auto &p : poly)
  {
    if (!p.ok_edit)
    {
      degnode[p.nF->lid].push_back(p.lid);
      degnode[p.nT->lid].push_back(p.lid);
    }
  }
  //std::cout << "Size of degnodes: " << degnode.size() << std::endl;
}

void cart::attach_nodes()
{
  std::map<int, std::pair<int, double>> close_nodes;
  double dist_thresh = 12.5;
  int r, c;
  for (const auto &d : degnode)
  {
    if (d.second.size() == 1)
    {
      auto n1 = node.begin() + d.first;
      get_node_cell(n1, r, c);
      for (const auto &n2 : grid.grid[r][c].node)
        if (degnode[n2->lid].size() == 1 && n1 != n2)
        {
          double dist = ::distance(n1, n2);

          if (dist < dist_thresh)
          {
            if (n1->lid < n2->lid)
              close_nodes[n1->lid] = std::make_pair(n2->lid, dist);
            else
              close_nodes[n2->lid] = std::make_pair(n1->lid, dist);
          }
        }
    }
  }

  for (const auto &c1 : close_nodes)
  {
    //std::cout << node[c1.first].cid << "  " << node[c1.second.first].cid << ". Dist: " << c1.second.second << std::endl;
    auto n1 = node.begin() + c1.first;
    auto n2 = node.begin() + c1.second.first;
    poly_it p1, p2;
    for (const auto &p : poly) {
      if (!p.ok_edit)
      {
        if (n1 == p.nF || n1 == p.nT)
          p1 = poly.begin() + p.lid;
        if (n2 == p.nF || n2 == p.nT)
          p2 = poly.begin() + p.lid;
      }
    }
    if (p1->cid == p2->cid) {
      continue;
    }
    if (n1 == p1->nF && n2 == p2->nF) //FF
    {
      p1->nF = n2;
      p1->point.front() = p2->point.front();
    }
    else if (n1 == p1->nF && n2 == p2->nT) //FT
    {
      p1->nF = n2;
      p1->point.front() = p2->point.back();
    }
    else if (n1 == p1->nT && n2 == p2->nF) //TF
    {
      p1->nT = n2;
      p1->point.back() = p2->point.front();
    }
    else if (n1 == p1->nT && n2 == p2->nT) //TT
    {
      p1->nT = n2;
      p1->point.back() = p2->point.back();
    }
  }
  update_degree();
}

void cart::remove_shortp()
{
  std::map<int, std::pair<int, double>> close_nodes;
  double lenght_thresh = 10.0;
  int r, c;
  for (const auto &d : degnode)
  {
    if (d.second.size() == 1)
    {
      auto n1 = node.begin() + d.first;
      get_node_cell(n1, r, c);
      for (const auto &n2 : grid.grid[r][c].node)
        if (degnode[n2->lid].size() == 1 && n1 != n2)
        {
          double dist = ::distance(n1, n2);

          if (dist < lenght_thresh && n1->link.front().second->cid == n2->link.front().second->cid)
          {
            if (!n1->link.front().second->ok_edit)
              n1->link.front().second->ok_edit = true;
          }
        }
    }
  }
  update_degree();
}

void cart::remove_doubleconnections()
{
  std::map<int, std::map<int, std::set<int>>> connections;
  for (const auto &n: node){
    for (const auto &nl:n.link)
      connections[n.lid][nl.first->lid].insert(nl.second->lid);
  }
  for (const auto &cn: connections){
    for (const auto &nn: cn.second){
      if (nn.second.size()==2)
      {
        if (poly[*nn.second.begin()].length != poly[*nn.second.end()].length) 
          continue;

        poly[*nn.second.begin()].ok_edit=true;
      }
      else if (nn.second.size()>2)
      {
        double min_l = 10e6;
        int short_p;
        std::set<double> lenght_list;
        for (auto &p:nn.second){
          // find shorter poly
            lenght_list.insert(poly[p].length);
          if (poly[p].length < min_l){
            min_l = poly[p].length;
            short_p = p;
          }
        }
        if (lenght_list.size()==nn.second.size()) 
          continue;
        
        for (auto &p:nn.second)
          if (p!=short_p) poly[p].ok_edit=true;
      }
    }
  }
}

void cart::remove_degree2()
{
  //std::cout << "size node before: " << degnode.size() << std::endl;
  int counter_2 = 0;
  int counter_n = 0;
  int poly_degremoved = 0;
  for (const auto &d : degnode)
  {
    if (d.second.size() == 2)
    {
      counter_2++;
      int n = d.first;
      int p1_lid = d.second.front();
      int p2_lid = d.second.back();
      while (p1_lid != poly[p1_lid].lid) p1_lid = poly[p1_lid].lid;
      while (p2_lid != poly[p2_lid].lid) p2_lid = poly[p2_lid].lid;

      int n1_oth, n2_oth;
      if (poly[p1_lid].nF->lid == node[n].lid)
        n1_oth = poly[p1_lid].nT->lid;
      else if (poly[p1_lid].nT->lid == node[n].lid)
        n1_oth = poly[p1_lid].nF->lid;
      else
        throw std::runtime_error("remove_degree2 : n1_oth unexpected value.");

      if (poly[p2_lid].nF->lid == node[n].lid)
        n2_oth = poly[p2_lid].nT->lid;
      else if (poly[p2_lid].nT->lid == node[n].lid)
        n2_oth = poly[p2_lid].nF->lid;
      else
        throw std::runtime_error("remove_degree2 : n2_oth unexpected value.");

      bool avoid_loop = false;
      for (const auto &p : poly)
      {
        if (p.ok_edit == true) continue;

        if (p.point.front()->ilat == node[n1_oth].ilat && p.point.front()->ilon == node[n1_oth].ilon
          && p.point.back()->ilat == node[n2_oth].ilat && p.point.back()->ilon == node[n2_oth].ilon)
          avoid_loop = true;

        else if (p.point.back()->ilat == node[n1_oth].ilat && p.point.back()->ilon == node[n1_oth].ilon
          && p.point.front()->ilat == node[n2_oth].ilat && p.point.front()->ilon == node[n2_oth].ilon)
          avoid_loop = true;
      }

      if (avoid_loop) continue;

      poly[p2_lid].ok_edit = true;
      poly_degremoved++;
      std::vector<point_it> point_f;

      if (poly[p1_lid].nF->lid == node[n].lid && poly[p2_lid].nF->lid == node[n].lid)      // FF
      {
        poly[p1_lid].nF = poly[p2_lid].nT;
        point_f.insert(point_f.end(), poly[p2_lid].point.rbegin(), poly[p2_lid].point.rend());
        point_f.insert(point_f.end(), poly[p1_lid].point.begin() + 1, poly[p1_lid].point.end());
      }
      else if (poly[p1_lid].nF->lid == node[n].lid && poly[p2_lid].nT->lid == node[n].lid) // FT
      {
        poly[p1_lid].nF = poly[p2_lid].nF;
        point_f.insert(point_f.end(), poly[p2_lid].point.begin(), poly[p2_lid].point.end());
        point_f.insert(point_f.end(), poly[p1_lid].point.begin() + 1, poly[p1_lid].point.end());
      }
      else if (poly[p1_lid].nT->lid == node[n].lid && poly[p2_lid].nF->lid == node[n].lid) // TF
      {
        poly[p1_lid].nT = poly[p2_lid].nT;
        point_f.insert(point_f.end(), poly[p1_lid].point.begin(), poly[p1_lid].point.end());
        point_f.insert(point_f.end(), poly[p2_lid].point.begin() + 1, poly[p2_lid].point.end());
      }
      else if (poly[p1_lid].nT->lid == node[n].lid && poly[p2_lid].nT->lid == node[n].lid) // TT
      {
        poly[p1_lid].nT = poly[p2_lid].nF;
        point_f.insert(point_f.end(), poly[p1_lid].point.begin(), poly[p1_lid].point.end());
        point_f.insert(point_f.end(), poly[p2_lid].point.rbegin() + 1, poly[p2_lid].point.rend());
      }
      poly[p1_lid].point = point_f;
      poly[p2_lid].lid = poly[p1_lid].lid;
    }
    else counter_n++;
  }
  update_degree();
}

void cart::merge_subgraph()
{
  int bigsubg_idx = -1, max_size = -1;
  double dist_threshold = 60;
  int cnt_subremoved = 0;
  for (const auto &g : subgraph)
  {
    if (int(g.second.size()) > max_size)
    {
      max_size = int(g.second.size());
      bigsubg_idx = g.first;
    }
  }

  for (const auto &g : subgraph)
  {
    if (g.first != bigsubg_idx)
    {
      //std::cout << "subg " << g.first << std::endl;
      std::pair<arc_it, double> arc_2connect = std::make_pair(arc.end(), 0.);
      std::pair<poly_it, node_base> poly_main;
      double dist = std::numeric_limits<double>::max();

      for (const auto &n : g.second)
      {
        if (node[n].link.size() == 1)
        {
          //std::cout<<"   --------------------------------------------------   "<<std::endl;
          //std::cout<< " poly " << node[n].link.front().second->lid << std::endl;
          //std::cout << "node pending: " << node[n].lid << " altro: " << node[n].link.front().first->lid << std::endl;

          auto ptn = point_base(node[n].ilat, node[n].ilon);
          //auto a = get_nearest_arc(ptn);

          int r, c;
          grid.coord_to_grid(ptn.ilat, ptn.ilon, r, c);
          //double d;
          //double dmin = std::numeric_limits<double>::max(), dmin2 = std::numeric_limits<double>::max();
          //arc_it nearest = arc.end(), nearest2 = arc.end();
          std::vector<std::pair<arc_it, double>> arclist;
          for (auto a = grid.grid[r][c].arc.begin(); a != grid.grid[r][c].arc.end(); ++a)
          {
            std::pair<arc_it, double> arc_pair = std::make_pair(*a, distance(ptn, **a));
            if (node_subgra[arc_pair.first->p->nF->lid] == bigsubg_idx && arc_pair.second < dist_threshold)
              arclist.push_back(arc_pair);
          }
          std::sort(arclist.begin(), arclist.end(), [](std::pair<arc_it, double> p1, std::pair<arc_it, double> p2) {
            return p1.second < p2.second;
          });
          if (arclist.size() && arclist[0].second < dist)
          {
            poly_main.first = node[n].link.front().second;
            poly_main.second = node[n];
            arc_2connect = arclist[0];
            dist = arc_2connect.second;
          }
        }
      }

      // connect subgraph pending node to main graph
      if (arc_2connect.first == arc.end())
      {
        std::cout << "WARNING: Skip merging and do remove of subgraph #" << g.first << std::endl;
        cnt_subremoved++;
        for (const auto &n : g.second)
        {
          for (const auto &l : node[n].link)
          {
            l.second->ok_edit = true;
          }
        }
        continue;
      }

      int ilat_node = poly_main.second.ilat;
      int ilon_node = poly_main.second.ilon;

      int dplat, dplon;
      double dslon, px, py;

      dplon = ilon_node - arc_2connect.first->ptF->ilon;
      dplat = ilat_node - arc_2connect.first->ptF->ilat;
      dslon = DSLAT * std::cos(arc_2connect.first->ptF->ilat * IDEG_TO_RAD);
      px = dslon * dplon;
      py = DSLAT * dplat;
      double a_p = arc_2connect.first->ux * px + arc_2connect.first->uy * py; // projection of p on a

      if (a_p < (arc_2connect.first->length / 2))
      {
        //caso 1: attacco a Tail
        if (poly_main.second.isF(poly_main.first))
        {
          poly_main.first->nF = arc_2connect.first->p->nF;
          poly_main.first->point.front() = arc_2connect.first->p->point.front();
        }
        else if (poly_main.second.isT(poly_main.first))
        {
          poly_main.first->nT = arc_2connect.first->p->nF;
          poly_main.first->point.back() = arc_2connect.first->p->point.front();
        }
      }
      else
      {
        //caso 2: attacco a Front
        if (poly_main.second.isF(poly_main.first))
        {
          poly_main.first->nF = arc_2connect.first->p->nT;
          poly_main.first->point.front() = arc_2connect.first->p->point.back();
        }
        else if (poly_main.second.isT(poly_main.first))
        {
          poly_main.first->nT = arc_2connect.first->p->nT;
          poly_main.first->point.back() = arc_2connect.first->p->point.back();
        }
      }
      //std::cout << "Merged subgraph #" << g.first << std::endl;
    }
  }
  std::cout << "Removed subgraph: " << cnt_subremoved << std::endl;
  update_degree();
}

void cart::assign_level_ps(const std::string &grid_file)
{
  try
  {
    jsoncons::json jgrid = jsoncons::json::parse_file(grid_file);

    auto dilat = jgrid["grid_data"]["dilat"].as<int>();
    auto dilon = jgrid["grid_data"]["dilon"].as<int>();
    auto gilatmin = jgrid["grid_data"]["ilat_min"].as<int>();
    auto gilonmin = jgrid["grid_data"]["ilon_min"].as<int>();
    auto grow = jgrid["grid_data"]["grow"].as<int>();
    auto gcol = jgrid["grid_data"]["gcol"].as<int>();
    auto gside = jgrid["grid_data"]["gside"].as<double>();
    double dist_thresh = gside / std::sqrt(2);
    auto gvalues = jgrid["grid_values"].as<std::vector<std::vector<int>>>();

    double ave_lvl_ps = 0.;
    int n_lvl_ps = 0;
    for (auto &p : poly)
    {
      std::vector<int> lvl_ps;
      p.ok_edit = false;

      int Fr = (p.nF->ilat - gilatmin) / dilat;
      int Tr = (p.nT->ilat - gilatmin) / dilat;
      int Fc = (p.nF->ilon - gilonmin) / dilon;
      int Tc = (p.nT->ilon - gilonmin) / dilon;

      if (Tr < Fr) { Fr = Fr ^ Tr; Tr = Fr ^ Tr; Fr = Fr ^ Tr; }  // swap content
      if (Tc < Fc) { Fc = Fc ^ Tc; Tc = Fc ^ Tc; Fc = Fc ^ Tc; }

      // protect from out of grid nodes
      if (Fr < 0) { Fr = 0; } if (Fr >= grow) { Fr = grow - 1; }
      if (Tr < 0) { Tr = 0; } if (Tr >= grow) { Tr = grow - 1; }
      if (Fc < 0) { Fc = 0; } if (Fc >= gcol) { Fr = gcol - 1; }
      if (Tc < 0) { Tc = 0; } if (Tc >= gcol) { Tr = gcol - 1; }

      for (int r = Fr; r <= Tr; ++r)
      {
        for (int c = Fc; c <= Tc; ++c)
        {
          auto pt = point_base(gilatmin + r * dilat + dilat / 2, gilonmin + c * dilon + dilon / 2);
          if (distance(pt, p) < dist_thresh)
          {
            lvl_ps.push_back(gvalues[r][c]);
          }
        }
      }

      if (lvl_ps.size())
      {
        std::sort(lvl_ps.begin(), lvl_ps.end());
        p.lvl_ps = lvl_ps.front();
        if (p.lvl_ps > 0.)
        {
          ave_lvl_ps += p.lvl_ps;
          ++n_lvl_ps;
        }
        else
        {
          p.lvl_ps = NO_LVL_PS;
        }
      }
      else
      {
        std::cout << "[assign_lvl_ps] WARNING Poly " << p.lid << "(" << p.cid << ")" << " has no lvl in grid" << std::endl;
        p.lvl_ps = NO_LVL_PS;
      }
    }
    ave_lvl_ps /= n_lvl_ps;
    std::cout << "[assign_lvl_ps] INFO Average water level PS : " << ave_lvl_ps << " [cm]" << std::endl;

    // filling with average value
    for (auto &p : poly)
      if (p.lvl_ps == NO_LVL_PS)
        p.lvl_ps = int(ave_lvl_ps);
  }
  catch (std::exception &e)
  {
    std::cerr << "EXC: Assign water level PS -> " << e.what() << std::endl;
  }
}

int cart::find_intersection(point_base p0, point_base p1, point_base b0, point_base b1)
{
  //b x p
  int cross = (b1.ilon - b0.ilon)*(p1.ilat - p0.ilat) - (p1.ilon - p0.ilon)*(b1.ilat - b0.ilat);
  if (!cross)  return 0;
  float inv_cross = 1.f / cross;

  float sb = ((p1.ilon - b0.ilon)*(b1.ilat - b0.ilat) - (b1.ilon - b0.ilon)*(p1.ilat - b0.ilat))*inv_cross;
  float sp = ((p1.ilon - b0.ilon)*(p1.ilat - p0.ilat) - (p1.ilon - p0.ilon)*(p1.ilat - b0.ilat))*inv_cross;

  if (sb > 0 && sb < 1 && sp > 0 && sp < 1)
    return 1;

  return 0;
}

void cart::assign_level_bridge(const std::string &bridge_filename)
{
  jsoncons::json geodata = jsoncons::json::parse_file(bridge_filename);
  std::map<std::string, double> bridge_height;
  std::map<std::string, std::vector<point_base>> bridge_geom;
  for (const auto &f : geodata["features"].array_range())
  {
    auto fp = f["properties"];
    std::string id_bridge = fp["ID"].as<std::string>();
    bridge_height[id_bridge] = fp["FRECCIA"].as<double>();
    if (f["geometry"]["type"].as<std::string>() == "Polygon")
    {
      for (const auto &g : f["geometry"]["coordinates"].array_range())
        for (const auto &h : g.array_range())
        {
          point_base pt(int(h[1].as<double>()*1e6), int(h[0].as<double>()*1e6));
          bridge_geom[id_bridge].push_back(pt);
        }
    }
    else if (f["geometry"]["type"].as<std::string>() == "MultiPolygon") {
      for (const auto &g : f["geometry"]["coordinates"].array_range())
        for (const auto &h : g.array_range())
          for (const auto &mh : h.array_range()) {
            point_base pt(int(mh[1].as<double>()*1e6), int(mh[0].as<double>()*1e6));
            bridge_geom[id_bridge].push_back(pt);
          }
    }
  }

  std::cout << "Number of bridges: " << bridge_geom.size() << std::endl;
  for (auto &bg : bridge_geom) {
    //select set of poly closest
    std::set<int> poly_closest;
    for (const auto &cord : bg.second) {
      int r, c;
      grid.coord_to_grid(cord.ilat, cord.ilon, r, c);
      arc_it nearest = arc.end();
      for (auto a = grid.grid[r][c].arc.begin(); a != grid.grid[r][c].arc.end(); ++a) {
        nearest = *a;
        poly_closest.insert(nearest->p->lid);
      }
    }

    //for each poly find intersection
    for (auto &pc : poly_closest) {
      auto poly_iter = poly.begin() + pc;
      int n_intersect = 0;
      for (int bc = 0; bc < int(bg.second.size() - 1); ++bc) {
        for (int i = 0; i < int(poly_iter->point.size() - 1); ++i) {
          n_intersect += find_intersection(*poly_iter->point[i], *poly_iter->point[i + 1], bg.second[bc], bg.second[bc + 1]);
        }
      }
      if (n_intersect > 0 && (poly_iter->lvl_ps == 0 || poly_iter->lvl_ps > bridge_height[bg.first])) poly_iter->lvl_ps = bridge_height[bg.first];
    }
  }

  int cnt_p = 0;
  for (const auto &i : poly)
    if (i.lvl_ps != 0)
      cnt_p++;

  std::cout << "number of poly crossing bridge: " << cnt_p << std::endl;
}

void cart::dump_noturn_file(const std::string &noturn_filename){
    std::ofstream out_noturn(noturn_filename+".file_noturn");
    out_noturn<<"cid1;cid2"<<std::endl;
    std::ifstream noturnf(noturn_filename);
    if( !noturnf ) throw std::runtime_error("File LatLon forbidden turn not found");
    std::map<std::string, point_base> nocrossings;
    std::string line;
    std::vector<std::string> tokens;
    std::getline(noturnf, line); // skip header
    while (std::getline(noturnf, line))
    {
      physycom::split(tokens, line, std::string(";"), physycom::token_compress_off);
      int ilat_c = int(stod(tokens[1])*1e6);
      int ilon_c = int(stod(tokens[2])*1e6);
      point_base point_c(ilat_c, ilon_c);
      nocrossings[tokens[0]]=point_c;
    }

    for (const auto &c: nocrossings)
    {
      node_it node_c = get_nearest_node(c.second);
      if (int(node_c->link.size())!=4){
        std::cout<<"Skip noturn crossings: "<<c.first<<" Size link: "<<node_c->link.size()<<std::endl;
      }
      else{
        node_it node_piv = node_c->link.front().first;
        poly_it poly_piv = node_c->link.front().second;
        point_it point_piv;
        if (node_piv->isF(poly_piv))
          point_piv = poly_piv->point[int(poly_piv->point.size()-2)];
        else if(node_piv->isT(poly_piv))
          point_piv = poly_piv->point[1];

        double min_cos = 1.0;
        int opposite_lid;
        for (int i=1; i<int(node_c->link.size()); ++i){
          node_it node_w = node_c->link[i].first;
          poly_it poly_w = node_c->link[i].second;
          point_it point_w;
          if (node_w->isF(poly_w))
            point_w = poly_w->point[int(poly_w->point.size()-2)];
          else if(node_w->isT(poly_w))
            point_w = poly_w->point[1];

          double dslon = DSLAT * std::cos(node_c->ilat * IDEG_TO_RAD);
          double dx_p = dslon * (point_piv->ilon-node_c->ilon);
          double dy_p = DSLAT * (point_piv->ilat-node_c->ilat);
          double dx_w = dslon * (point_w->ilon-node_c->ilon);
          double dy_w = DSLAT * (point_w->ilat-node_c->ilat);
          double cos_theta = (dx_p*dx_w+dy_p*dy_w)/(sqrt(dx_p*dx_p+dy_p*dy_p)*sqrt(dx_w*dx_w+dy_w*dy_w));
          if (cos_theta < min_cos){
            min_cos = cos_theta;
            opposite_lid = poly_w->lid;
          }
        }
        std::vector<int> connected={poly_piv->lid, opposite_lid};
        std::vector<int> connected2;
        for (const auto &l: node_c->link)
          if ( (l.second->lid!=poly_piv->lid) & (l.second->lid!=opposite_lid) )
            connected2.push_back(l.second->lid);

        for (const auto &i:connected)
          for (const auto &j:connected2){
            out_noturn<<poly[i].cid<<";"<<poly[j].cid<<std::endl;
            out_noturn<<poly[j].cid<<";"<<poly[i].cid<<std::endl;
            }
      }
    }
}

void cart::reduce_area(const jsoncons::json &jconf)
{
  int lat_max = int(jconf["lat_max"].as<double>()*1e6);
  int lat_min = int(jconf["lat_min"].as<double>()*1e6);
  int lon_max = int(jconf["lon_max"].as<double>()*1e6);
  int lon_min = int(jconf["lon_min"].as<double>()*1e6);
  for (auto &p : poly)
  {
    if (!p.ok_edit)
      for (const auto &pp : p.point)
        if (pp->ilat > lat_max || pp->ilat < lat_min || pp->ilon > lon_max || pp->ilon < lon_min)
        {
          p.ok_edit = true;
          break;
        }
  }
}

void cart::remove_subgraph(const jsoncons::json &jconf)
{
  double frac = jconf.has_member("fraction_thresh") ? jconf["fraction_thresh"].as<double>() : 0.05;

  std::cout << "Removal fraction threshold: " << frac << std::endl;

  for (const auto &g : subgraph)
  {
    double sfrac = g.second.size() / double(node.size());
    std::cout << "subgraph #" << g.first << " nodes " << g.second.size() << " fraction " << sfrac << std::endl;

    if (sfrac < frac)
    {
      std::cout << "Removing subgraph #" << g.first << std::endl;

      // collect poly lid's
      std::set<int> poly_remove;
      for (const auto &nlid : g.second)
      {
        for (const auto &link : node[nlid].link)
        {
          poly_remove.insert(link.second->lid);
        }
      }
      std::cout << "Poly to remove: " << poly_remove.size() << std::endl;

      for (const auto &plid : poly_remove)
      {
        poly[plid].ok_edit = true;
      }
    }
  }

}

void cart::dump_edited()
{
  std::string basename;
  if (jconf.has_member("cartout_basename"))
    basename = jconf["cartout_basename"].as<std::string>();
  else
    basename = "carto_edit";
  std::ofstream editpro(basename + ".pro");
  std::ofstream editpnt(basename + ".pnt");
  editpro << "#poly_cid  node_front_cid  node_tail_cid  length  lvl_ps  ?  ?  ?  speed  ?  name" << std::endl;
  for (const auto &p : poly)
  {
    if (!p.ok_edit)
    {
      if (p.nF==p.nT) continue;
      editpnt << p.cid << " " << p.lid << " " << p.point.size() << std::endl;
      for (const auto &pt : p.point)
        editpnt << pt->ilat << " " << pt->ilon << std::endl;

      editpro << std::setw(15) << p.cid << " "
        << std::setw(15) << p.nF->cid << " "
        << std::setw(15) << p.nT->cid << " "
        << std::fixed << std::setprecision(1) << p.length << " "
        << p.lvl_ps << " "
        << "5  0  0  "
        << p.speed<<" "
        << "0 "
        << p.name
        << std::endl;
    }
  }
  editpro.close();
  editpnt.close();
}

void cart::dump_test_config()
{
  std::string basename;
  if (jconf.has_member("cartout_basename"))
    basename = jconf["cartout_basename"].as<std::string>();
  else
    basename = "carto_edit";

  std::ofstream editcfg(basename + ".test");
  editcfg << R"({)" << std::endl;
  editcfg << R"(  "file_pnt"          : ")" << basename << R"(.pnt",)" << std::endl;
  editcfg << R"(  "file_pro"          : ")" << basename << R"(.pro",)" << std::endl;
  editcfg << R"(  "verbose"           : false,)" << std::endl;
  editcfg << R"(  "enable_histo"      : false,)" << std::endl;
  editcfg << R"(  "enable_gui"        : true,)" << std::endl;
  editcfg << R"(  "enable_bp"         : false,)" << std::endl;
  editcfg << R"(  "enable_geojson"    : false,)" << std::endl;
  editcfg << R"(  "enable_explore"    : false,)" << std::endl;
  editcfg << R"(  "bp_matrix"         : false,)" << std::endl;
  editcfg << R"(  "state_grid_cell_m" : 1000)" << std::endl;
  editcfg << R"(})" << std::endl;
  editcfg.close();
}

void cart::dump_edit_config(std::string &cartout)
{
  std::string basename;
  if (jconf.has_member("cartout_basename"))
    basename = jconf["cartout_basename"].as<std::string>();
  else
    basename = "carto_edit";

  std::ofstream editcfg(basename + ".edit");
  editcfg << R"({)" << std::endl;
  editcfg << R"(  "file_pnt"              : ")" << basename << R"(.pnt",)" << std::endl;
  editcfg << R"(  "file_pro"              : ")" << basename << R"(.pro",)" << std::endl;
  editcfg << R"(  "cartout_basename"      : ")" << cartout << R"(",)" << std::endl;
  editcfg << R"(  "verbose"               : false,)" << std::endl;
  editcfg << R"(  "explore_node"          : [0],)" << std::endl;
  editcfg << R"(  "enable_merge_subgraph" : true,)" << std::endl;
  editcfg << R"(  "enable_remove_degree2" : true,)" << std::endl;
  editcfg << R"(  "enable_attach_nodes"   : true)" << std::endl;
  editcfg << R"(})" << std::endl;
  editcfg.close();
}
