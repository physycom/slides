#include <pybind11/pybind11.h>

#include <simulation.h>

struct pysimulation
{
  cart *c;
  simulation *s;
  bool valid;

  pysimulation(const std::string &conf_json)
  {
    jsoncons::json jconf;
    c = nullptr;
    s = nullptr;
    try
    {
      jconf = jsoncons::json::parse(conf_json);
      c = new cart(jconf);
      s = new simulation(jconf, c);
      valid = true;
    }
    catch (std::exception &e)
    {
      std::cerr << "[pysimulation] Generic EXC : " << e.what() << std::endl;
      valid = false;
    }
  }

  ~pysimulation()
  {
    delete c;
    delete s;
  }

  bool is_valid()
  {
    return valid;
  }

  void run()
  {
    if (s != nullptr)
    {
      s->run([this]() {
        s->dump_net_state();
        s->dump_population();
        s->dump_influxgrid();
      }, [] () {});
    }
  }

  void dump_poly_geojson(const std::string &basename)
  {
    if (c != nullptr)
    {
      c->dump_poly_geojson(basename);
    }
  }

  void dump_grid_geojson(const std::string &filename)
  {
    if (s != nullptr)
    {
      s->grid.dump_geojson(filename);
    }
  }

  std::string poly_outfile()
  {
    return (s != nullptr) ? s->poly_outfile : std::string("null");

  }
  std::string grid_outfile()
  {
    return (s != nullptr) ? s->grid_outfile : std::string("null");
  }

  std::string sim_info()
  {
    std::string ret;
    if (s != nullptr && c != nullptr)
    {
      ret = "\n" + c->info() + s->info();
    }
    else
    {
      ret = "sim init failed";
    }
    return ret;
  }

};

PYBIND11_MODULE(pysim, m)
{
  pybind11::class_<pysimulation>(m, "simulation")
    .def(pybind11::init<const std::string &>())
    .def("run", &pysimulation::run)
    .def("poly_outfile", &pysimulation::poly_outfile)
    .def("grid_outfile", &pysimulation::grid_outfile)
    .def("dump_poly_geojson", &pysimulation::dump_poly_geojson)
    .def("dump_grid_geojson", &pysimulation::dump_grid_geojson)
    .def("is_valid", &pysimulation::is_valid)
    .def("sim_info", &pysimulation::sim_info);
}
