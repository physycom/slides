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
  }
  catch (exception &e)
  {
    cerr << "EXC: " << e.what() << endl;
    exit(1);
  }

  return 0;
}
