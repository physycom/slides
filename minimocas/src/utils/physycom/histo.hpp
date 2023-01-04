/* Copyright 2017 - Alessandro Fabbri */

// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt


#ifndef PHYSYCOM_UTILS_HISTO_HPP
#define PHYSYCOM_UTILS_HISTO_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <fstream>
#include <numeric>
#include <limits>
#include <algorithm>

namespace physycom
{
  template<typename T>
  struct histo
  {
    std::map<std::string, std::vector<T>> data;
    std::map<std::string, std::vector<int>> counter;
    std::map<std::string, T> mean, stddev;
    T min, max, binw;
    int nmin = std::numeric_limits<int>::max();
    int nmax = std::numeric_limits<int>::min();
    int nbin;
    std::string xlabel, ylabel, title;

    histo() {};

    histo(T min_, T max_, T binw_, std::string xl = "Xlabel", std::string yl = "Counter", std::string t = "Title")
    {
      min = min_;
      max = max_;
      binw = binw_;
      nbin = int((max - min) / binw);
      xlabel = xl;
      ylabel = yl;
      title = t;
    }

    // this won't compile with int data
    // but it's handy for floating point
    histo(T min_, T max_, int nbin_, std::string xl = "Xlabel", std::string yl = "Counter", std::string t = "Title")
    {
      min = min_;
      max = max_;
      nbin = nbin_;
      binw = (max - min) / nbin;
      xlabel = xl;
      ylabel = yl;
      title = t;
    }

    void count(const std::string &tag, const T &t)
    {
      if (int(counter[tag].size()) != nbin) counter[tag].resize(nbin, 0);
      int idx = int((t - min) / binw);
      if (idx < 0) return;
      counter[tag][(idx > nbin - 1) ? nbin - 1 : idx]++;
    }

    void evaluate_mean()
    {
      for (auto i : data)
      {
        counter[i.first].resize(nbin, 0);
        for (const auto &d : i.second)
        {
          int idx = int((d - min) / binw);
          if (idx < 0) continue;
          counter[i.first][(idx > nbin - 1) ? nbin - 1 : idx]++;
        }
      }
    }

    void populate()
    {
      for (auto i : data)
      {
        counter[i.first].resize(nbin, 0);
        for (const auto &d : i.second)
        {
          int idx = int((d - min) / binw);
          if (idx < 0) continue;
          counter[i.first][(idx > nbin - 1) ? nbin - 1 : idx]++;
        }
      }
    }

    void dump(const std::string &filename)
    {
      // evaluate totals for normalization
      std::map<std::string, int> tot;
      std::map<std::string, double> cumulate;
      std::for_each(counter.begin(), counter.end(), [&tot](const std::pair<std::string, std::vector<int>> &p)
      {
        tot[p.first] = std::accumulate(p.second.begin(), p.second.end(), 0);
      });

      char sep = '\t';
      std::ofstream outhisto(filename);
      outhisto << "bin";
      for (const auto &label : counter) outhisto << sep << label.first << "-cnt" << sep << label.first << "-cdf";
      outhisto << std::endl;
      for (int i = 0; i < nbin; ++i)
      {
        outhisto << min + i * binw;
        for (const auto &label : counter)
        {
          int cnt = counter[label.first][i];
          nmin = (nmin < cnt) ? nmin : cnt;
          nmax = (nmax > cnt) ? nmax : cnt;

          outhisto << sep << cnt << sep << (cumulate[label.first] += cnt / (double)tot[label.first]);
        }
        outhisto << std::endl;
      }
      outhisto.close();
    }

    std::string gnuplot_style = R"(# Border
set style line 101 lc rgb '#808080' lt 1 lw 1
set border 11 front ls 101
set tics nomirror out scale 0.75
set format y '%.0s %c'
set format y2 '%.0s %%'
set border linewidth 1.5
# Styles
linew = 1.2
ptsize = 1.5
set style line 11 lc rgb '#0072bd' lt 1 lw linew # blue
set style line 12 lc rgb '#d95319' lt 1 lw linew # orange
set style line 13 lc rgb '#edb120' lt 1 lw linew # yellow
set style line 14 lc rgb '#7e2f8e' lt 1 lw linew # purple
set style line 15 lc rgb '#77ac30' lt 1 lw linew # green
set style line 16 lc rgb '#4dbeee' lt 2 lw linew # light - blue
set style line 17 lc rgb '#a2142f' lt 1 lw linew # red
set style line 21 lc rgb '#0072bd' pointtype 7 lw linew ps ptsize # blu
set style line 22 lc rgb '#d95319' pointtype 7 lw linew ps ptsize # orange
set style line 23 lc rgb '#edb120' pointtype 7 lw linew ps ptsize # yellow
set style line 24 lc rgb '#7e2f8e' pointtype 7 lw linew ps ptsize # purple
set style line 25 lc rgb '#77ac30' pointtype 7 lw linew ps ptsize # green
set style line 26 lc rgb '#4dbeee' pointtype 7 lw linew ps ptsize # light - blue
set style line 27 lc rgb '#a2142f' pointtype 7 lw linew ps ptsize # red
)";

    void gnuplot(const std::string &filename) const
    {
      std::string basename = filename.substr(0, filename.find_last_of("."));
      basename = basename.substr(filename.find_last_of("/\\") + 1);
      std::ofstream outplt(filename);
      outplt << R"(set terminal pngcairo  transparent enhanced font "Verdana,20" fontscale 0.8 size 960, 720 background rgb 'white'
set output ')" << basename << R"(.png')" << gnuplot_style << R"(# Grid and Ticks
set ytics 0, )" << 1.2 * nmax / 5 << ", " << 10 * nmax << R"( nomirror out scale 0.75
set style line 102 lc rgb '#d6d7d9' lt 1 lw 1
set grid xtics ytics back ls 102
# More options
set style fill solid 1.00 border lt - 1
set style histogram clustered gap 1 title textcolor lt - 1
set style data histograms
set key top center autotitle columnhead
set xrange[-1:)" << nbin << R"(]
set yrange[)" << nmin << ":" << 1.2 * nmax << R"(]
set title ')" << title << R"('
set xlabel ')" << xlabel << R"('
set ylabel ')" << ylabel << R"('
set xtics border in scale 0, 0 nomirror rotate by - 45
label_undersampling = 2
plot ')" << basename << R"(.txt')";

      int cnt = 0, column = 2, style = 11;
      for (const auto &p : counter)
      {
        outplt << ((cnt == 0) ? "" : "     ''") << " using " << column << ":xtic(int($0) % label_undersampling == 0 ? stringcolumn(1) : '') title '" << p.first << "' ls " << style++ << ((cnt != int(counter.size() - 1)) ? " ,\\" : "") << std::endl;
        column += 2;
        ++cnt;
      }
      outplt.close();
    }

    void gnuplot_cdf(const std::string &filename) const
    {
      std::string basename = filename.substr(0, filename.find_last_of("."));
      std::ofstream outplt(basename + ".cdf.plt");
      basename = basename.substr(filename.find_last_of("/\\") + 1);
      outplt << R"(set terminal pngcairo  transparent enhanced font "Verdana,20" fontscale 0.8 size 960, 720 background rgb 'white'
set output ')" << basename << R"(.cdf.png')" << gnuplot_style << R"(# Grid and Ticks
set ytics 0, )" << 1.2 * nmax / 5 << ", " << 10 * nmax << R"( nomirror out scale 0.75
set y2tics 0, 10, 110 nomirror out scale 0.35
set style line 102 lc rgb '#d6d7d9' lt 1 lw 1
set grid xtics ytics back ls 102
# More options
set style fill solid 1.00 border lt - 1
set style histogram clustered gap 1 title textcolor lt - 1
set style data histograms
set key top left autotitle columnhead
set xrange[-1:)" << nbin << R"(]
set yrange[)" << nmin << ":" << 1.2 * nmax << R"(]
set y2range[0:105]
set title ')" << title << R"('
set xlabel ')" << xlabel << R"('
set ylabel ')" << ylabel << R"('
set y2label 'Percentage'
set xtics border in scale 0, 0 nomirror rotate by - 45
label_undersampling = 2
plot ')" << basename << R"(.txt')";

      int cnt = 0, column = 2, hstyle = 11, lpstyle = 21;
      for (const auto &p : counter)
      {
        outplt << ((cnt == 0) ? "" : "     ''") << " using " << column++ << ":xtic(int($0) % label_undersampling == 0 ? stringcolumn(1) : '') title '" << p.first << "' ls " << hstyle++ << ((cnt != int(2 * counter.size() - 1)) ? " ,\\" : "") << std::endl; cnt++;
        outplt << ((cnt == 0) ? "" : "     ''") << " using (column(0)):($" << column++ << "*100) title '" << p.first << "-cdf' with linespoints ls " << lpstyle++ << " axes x1y2 " << ((cnt != int(2 * counter.size() - 1)) ? " ,\\" : "") << std::endl; cnt++;
      }
      outplt.close();
    }
  };

  template<typename T>
  struct multihisto
  {
    std::map<std::string, histo<T>> hs;
    void add_histo(std::string name, T min, T max, T binw, std::string xl = "Xlabel", std::string yl = "Counter", std::string t = "Title") { hs[name] = histo<T>(min, max, binw, xl, yl, t); }
    void add_histo(std::string name, T min, T max, int nbin, std::string xl = "Xlabel", std::string yl = "Counter", std::string t = "Title") { hs[name] = histo<T>(min, max, nbin, xl, yl, t); }
    void count(std::string name, std::string tag, T t) { hs[name].count(tag, t); }
    void push(std::string name, std::string tag, T t) { hs[name].data[tag].push_back(t); }
    void populate() { for (auto &h : hs) h.second.populate(); }
    void dump(std::string path_prefix = "") { for (auto &h : hs) h.second.dump(path_prefix + "histo_" + h.first + ".txt"); }
    void gnuplot(std::string path_prefix = "") { for (auto &h : hs) h.second.gnuplot(path_prefix + "histo_" + h.first + ".plt"); }
    void gnuplot_cdf(std::string path_prefix = "") { for (auto &h : hs) h.second.gnuplot_cdf(path_prefix + "histo_" + h.first + ".plt"); }
  };

  template<typename T>
  struct covstats
  {
    multihisto<T> * mh;
    std::map<std::string, std::map<std::string, std::map<std::string, double>>> cov;     // < tag, column, column, value>
    std::map<std::string, std::map<std::string, std::map<std::string, double>>> quad;    // < tag, column, column, value>
    std::map<std::string, std::map<std::string, double>> mean;                           // < tag, column, value>
    std::map<std::string, int> ndata;                                                    // < tag, value >

    covstats(multihisto<T> &mh_)
    {
      mh = &mh_;
    }

    void populate()
    {
      // sample numbers
      for (auto x : mh->hs)
      {
        for (auto tag : x.second.data)
        {
          ndata[tag.first] = (int)tag.second.size();
        }
        break;
      }

      // mean
      for (auto &x : mh->hs)
        for (auto &tag : x.second.data)
          for (int i = 0; i < tag.second.size(); ++i)
            mean[tag.first][x.first] += x.second.data[tag.first][i];

      // quad mean
      for (auto &x : mh->hs)
        for (auto &y : mh->hs)
          for (auto &tag : x.second.data)
            for (int i = 0; i < tag.second.size(); ++i)
            {
              quad[tag.first][x.first][y.first] += x.second.data[tag.first][i] * y.second.data[tag.first][i];
              //cov[tag.first][x.first][y.first] += (x.second.data[tag.first][i] - mean[tag.first][x.first] / ndata[tag.first]) * (y.second.data[tag.first][i] - mean[tag.first][y.first] / ndata[tag.first]) / ndata[tag.first];
            }

      // covariance
      for (auto tag : quad)
        for (auto i : tag.second)
          for (auto j : i.second)
          {
            std::cout << tag.first << " " << i.first << " " << j.first << std::endl;
            cov[tag.first][i.first][j.first] = quad[tag.first][i.first][j.first] / double(ndata[tag.first]) - mean[tag.first][i.first] * mean[tag.first][j.first] / double(ndata[tag.first] * ndata[tag.first]);
          }
    }
  };

}   // end namespace physycom

#endif // PHYSYCOM_UTILS_HISTO_HPP
