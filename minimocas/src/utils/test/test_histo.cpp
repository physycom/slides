/* Copyright 2017 - Alessandro Fabbri */

// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt

#include <iostream>
#include <random>
#include <algorithm>

#include <physycom/histo.hpp>

using namespace std;
using namespace physycom;

int main()
{
  std::mt19937 mt(12);

  // exponential
  auto lambdas = std::vector<double>({0.1, 0.5, 1.0, 2.0, 10.0});
  auto exps = std::vector<std::exponential_distribution<double>>();
  for(auto l : lambdas) exps.push_back(std::exponential_distribution<double>(l));

  // gaussian
  auto norm_p = std::vector<std::pair<double,double>>({
    { 0.5, 0.2 },
    { 0.5, 0.6 },
    { 1.0, 0.1 },
    { 1.0, 0.4 },
  });
  auto norms = std::vector<std::normal_distribution<double>>();
  for(auto p : norm_p) norms.push_back(std::normal_distribution<double>(p.first, p.second));

  // gamma
  auto gamma_p = std::vector<std::pair<double,double>>({
    { 3.0, 2.0 },
    { 7.0, 1.0 },
    { 5.0, 1.0 },
  });
  auto gammas = std::vector<std::gamma_distribution<double>>();
  for(auto p : gamma_p) gammas.push_back(std::gamma_distribution<double>(p.first, p.second));

  // discrete exp
  int nval = 20, start_val = 100;
  std::vector<int> values(nval);
  std::iota(values.begin(), values.end(), start_val);
  double binw = 500.;
  std::vector<double> distc({ 1500., 3100., 5000. });
  std::vector<std::vector<double>> probs(distc.size());
  std::vector<std::discrete_distribution<int>> dexps;
  for(int n=0; n<(int)distc.size(); ++n)
  {
    for(int i=0; i<nval; ++i) probs[n].emplace_back(std::exp(-i*binw/distc[n])/distc[n]);
    dexps.emplace_back(std::discrete_distribution<int>(probs[n].begin(), probs[n].end()));
  }

  // histograms
  multihisto<double> mh;
  mh.add_histo("exp", 0., 20., 25, "x", "Counter", "Exponential Distribution");
  mh.add_histo("norm", -2., 3., 25, "x", "Counter", "Normal Distribution");
  mh.add_histo("gamma", 0., 20., 25, "x", "Counter", "Gamma Distribution");
  mh.add_histo("dexp", start_val - 1, start_val + nval + 1, nval + 2, "x", "Counter", "Discrete Exp Distribution");
  int Nrolls = 10000;
  for(int i=0; i<Nrolls; ++i)
  {
    for(int n=0; n < exps.size(); ++n)
    {
      auto x = exps[n](mt);
      mh.count("exp","l = " + std::to_string(lambdas[n]), x);
    }
    for(int n=0; n < norms.size(); ++n)
    {
      auto x = norms[n](mt);
      mh.count("norm","m=" + std::to_string(norm_p[n].first) + " s=" + std::to_string(norm_p[n].second), x);
    }
    for(int n=0; n < gammas.size(); ++n)
    {
      auto x = gammas[n](mt);
      mh.count("gamma","k=" + std::to_string(gamma_p[n].first) + " th=" + std::to_string(gamma_p[n].second), x);
    }
    for(int n=0; n < dexps.size(); ++n)
    {
      auto x = dexps[n](mt);
      mh.count("dexp","dc = " + std::to_string(distc[n]), double(values[x]));
    }
  }
  mh.dump("histo/");
  mh.gnuplot("histo/");
  mh.gnuplot_cdf("histo/");

	return 0;
}

