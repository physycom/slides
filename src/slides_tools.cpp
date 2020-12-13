/*!
 *  \file   minitest.cpp
 *  \author A. Fabbri (alessandro.fabbri27@unibo.it), C. Mizzi (chiara.mizzi2@unibo.it)
 *  \brief  OpenGL based methods implementations for app `miniplex`.
 *  \details This file contains the implementation of the graphic method, based on the FLTK wrapper of OpenGL, realated to `miniplex` app.
 */

#include <iostream>
#include <carto.h>
#include <simulation.h>

using namespace std;
using namespace jsoncons;
using namespace physycom;

constexpr int MAJOR = 1;
constexpr int MINOR = 0;

#define forit(x, y) for(auto x = y.begin(); x != y.end(); ++x)
#define foritr(x, y) for(auto x = y.rbegin(); x != y.rend(); ++x)
#define each(x, y) auto x = y.rbegin(); x != y.rend(); ++x

void usage(const char* progname)
{
  string pn(progname);
  cerr << "Usage: " << pn.substr(pn.find_last_of("/\\") + 1) << " path/to/json/config" << endl;
}

int main(int argc, char** argv)
{
  cout << "slides_tools v" << MAJOR << "." << MINOR << endl;

  string conf;
  if (argc == 2)
  {
    conf = argv[1];
  }
  else
  {
    usage(argv[0]);
    exit(1);
  }

  try
  {
    json jconf = json::parse_file(conf);

    cart c(jconf);
    std::cout << c.info() << std::endl;

    simulation s(jconf, &c);
    std::cout << s.info() << std::endl;

    double dist_near = 50.0;
    //finde near nodes
    std::map<int, std::vector<int>> near_nodes;
    for (const auto &ai : s.attractions) {
      for (const auto &ni : c.node)
        if (distance(c.node[ai.node_lid], ni) < dist_near)
          near_nodes[ai.node_lid].push_back(ni.lid);
    }

    // make container and initialize
    std::map<int, int> poly_counter;
    for (const auto &pi : c.poly) {
      poly_counter[pi.lid] = 0;
    }
    std::vector<int> path;
    for (const auto &ni : near_nodes) {
      for (const auto &nj : near_nodes) {
        if (ni != nj) {
          for (const auto &ai : ni.second)
            for (const auto &aj : nj.second) {
              path.clear();
              if (c.bestpath(c.node[ai], c.node[aj], path) == BP_PATHFOUND)
                for (const auto &pi : path)
                  poly_counter[pi]++;
            }
        }
      }


      for (const auto &si : s.sources) {
        for (const auto &bi:ni.second){
        path.clear();
        if (c.bestpath(c.node[bi], c.node[si.node_lid], path) == BP_PATHFOUND)
          for (const auto &pi : path)
            poly_counter[pi]++;
        }
      }
    }
    
    //dump
    std::string outname;
    if (jconf.has_member("file_cnt_out")) {
      std::ofstream out_fluxes(jconf["file_cnt_out"].as<std::string>());
      out_fluxes << "#lid cid n_FT  n_TF" << std::endl;
      for (const auto &pc : poly_counter)
        out_fluxes << pc.first << " " << c.poly[pc.first].cid << " " << pc.second << " " << pc.second << std::endl;
      out_fluxes.close();
    }
    else
      std::cout << "file_cnt name unspecified in json config!" << std::endl;

  }
  catch (exception &e)
  {
    cerr << "EXC: " << e.what() << endl;
    exit(1);
  }

  return 0;
}
