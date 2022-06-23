/* Copyright 2016-2017 - Alessandro Fabbri, Chiara Mizzi, Stefano Sinigardi */

// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt


#ifndef PHYSYCOM_UTILS_TIME_HPP
#define PHYSYCOM_UTILS_TIME_HPP

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <ctime>
#include <sstream>
#include <map>

#include <physycom/string.hpp>

namespace physycom
{
  // Convert full LOCAL date 'YYYY-MM-DD hh:mm:ss' to unix time (N.B. unix time is DEFINED as GMT)
  inline size_t date_to_unix(const std::string &date, const char* format = "%Y-%m-%d %H:%M:%S")
  {
    std::tm t = {};
    t.tm_isdst = -1;              // this enforce the CORRECT recalculation by mktime
    std::stringstream ss(date);
    ss >> std::get_time(&t, format);
    return std::mktime(&t);
  }

  // Convert unix time to LOCAL date
  inline std::string unix_to_date(const size_t &t_unix, const char* format = "%Y-%m-%d %H:%M:%S")
  {
    struct tm * t = std::localtime((time_t *)&t_unix);
    char tcs[100];
    std::strftime(tcs, sizeof(tcs), format, t);
    return std::string(tcs);
  }


  // Assign time slot
  // 1) Auto mode, based on minutes
  inline std::string find_slot_auto_ranges_minutes(const std::string &date, const std::string &time_min = "00:15",const std::string &time_max = "23:59", int dtmin = 30) {     // 3/10/2016 12:14:34
    // extract date components and convert hour to minutes since 00:00:00
    std::vector<std::string> tok;
    physycom::split(tok, date, std::string("- :"), physycom::token_compress_off);
    std::stringstream slot;
    slot << tok[0] << tok[1] << tok[2] << "_";
    size_t tmin_now = stoi(tok[3])*60+stoi(tok[4]);

    // convert time_min and time_max to minutes since 00:00:00
    physycom::split(tok, time_min, std::string(":"), physycom::token_compress_off);
    size_t tmin_min = stoi(tok[0])*60 + stoi(tok[1]);
    physycom::split(tok, time_max, std::string(":"), physycom::token_compress_off);
    size_t tmin_max = stoi(tok[0])*60 + stoi(tok[1]);

    // prepare the output string
    if ( tmin_now >= tmin_min && tmin_now < tmin_max ){
      size_t slot_index = (tmin_now - tmin_min) / dtmin;
      size_t tmin_slot = tmin_min + slot_index*dtmin;

      size_t hi = tmin_slot/60;
      size_t mi = tmin_slot - hi*60;
      size_t hf, mf;
      if (tmin_max > tmin_slot + dtmin) {
        hf = (tmin_slot + dtmin)/60;
        mf = tmin_slot + dtmin - hf*60;
      }
      else {
        hf = tmin_max / 60;
        mf = tmin_max - hf * 60;
      }

      slot << std::setw(2) << std::setfill('0') << hi << "."
        << std::setw(2) << std::setfill('0') << mi << "-"
        << std::setw(2) << std::setfill('0') << hf << "."
        << std::setw(2) << std::setfill('0') << mf;
    }
    else{
      slot << "residuals";
    }

    return slot.str();
  }

  // 2) Manual mode
  inline std::string find_slot_manual_ranges(const std::string &date) {
    std::string slot;
    std::vector<std::string> date_v;
    physycom::split(date_v, date, std::string("- :"), physycom::token_compress_off);
    slot = date_v[0] + date_v[1] + date_v[2] + "_";
    int h = stoi(date_v[3]);
    if (h < 8 || h >= 23) {
      slot += "23-08";
    }
    else if (h >= 8 && h < 12) {
      slot += "08-12";
    }
    else if (h >= 12 && h < 16) {
      slot += "12-16";
    }
    else if (h >= 16 && h < 20) {
      slot += "16-20";
    }
    else if (h >= 20 && h < 23) {
      slot += "20-23";
    }
    else {
      slot += "residuals";     // you should not see this...
    }

    return slot;
  }

  // 3) Mini manual mode
  inline std::string find_slot_mini_ranges(const std::string &date) {
    std::string slot;
    std::vector<std::string> date_v;
    physycom::split(date_v, date, std::string("- :"), physycom::token_compress_off);
    int h = stoi(date_v[3]);
    if (h >= 7 && h < 12) {
      slot = "07-12";
    }
    else if (h >= 12 && h < 17) {
      slot = "12-17";
    }
    else if (h >= 17 && h < 22) {
      slot = "17-22";
    }
    else {
      slot = "night";
    }

    return slot;
  }

  // Returns a vector of string with the given ranges
  // E.G. "10.00-10.30" "10.30-11.00" ...
  inline std::vector<std::string> get_slot_auto_ranges_minutes(const std::string &time_min = "00:15", const std::string &time_max = "23:59", int dtmin = 30)
  {
    std::vector<std::string> tok;
    physycom::split(tok, time_min, std::string(":"), physycom::token_compress_off);
    int tmin_min = stoi(tok[0]) * 60 + stoi(tok[1]);
    physycom::split(tok, time_max, std::string(":"), physycom::token_compress_off);
    int tmin_max = stoi(tok[0]) * 60 + stoi(tok[1]);

    // prepare the output string
    std::vector<std::string> slots;
    int slot_num = (tmin_max - tmin_min) / dtmin;
    for (int i = 0; i<slot_num; ++i)
    {
      int tmin_slot = tmin_min + i*dtmin;
      int hi = tmin_slot / 60;
      int mi = tmin_slot - hi * 60;
      int hf, mf;
      if (tmin_max > tmin_slot + dtmin)
      {
        hf = (tmin_slot + dtmin) / 60;
        mf = tmin_slot + dtmin - hf * 60;
      }
      else
      {
        hf = tmin_max / 60;
        mf = tmin_max - hf * 60;
      }

      std::stringstream slot;
      slot
        << std::setw(2) << std::setfill('0') << hi << "."
        << std::setw(2) << std::setfill('0') << mi << "-"
        << std::setw(2) << std::setfill('0') << hf << "."
        << std::setw(2) << std::setfill('0') << mf;
      slots.push_back(slot.str());
    }

    return slots;
  }

  inline std::string get_little_easter(const int &year)
  {
    int day,month;
    int a, b, c, d, e, m, n;

    switch(year/100)
    {
      case 15:  // 1583 - 1599
      case 16:  // 1600 - 1699
        m=22; n=2;  break;
      case 17:  // 1700 - 1799
        m=23; n=3; break;
      case 18:  // 1800 - 1899
        m=23; n=4; break;
      case 19:  // 1900 - 1999
      case 20:  // 2000 - 2099
        m=24; n=5;break;
      case 21:  // 2100 - 2199
        m=24; n=6; break;
      case 22:  // 2200 - 2299
        m=25; n=0; break;
      case 23:  // 2300 - 2399
        m=26; n=1; break;
      case 24:  // 2400 - 2499
        m=25; n=1; break;
    }

    a=year%19;
    b=year%4;
    c=year%7;
    d=(19*a+m)%30;
    e=(2*b+4*c+6*d+n)%7;
    day=d+e;

    if (d+e<10)
    {
      day += 22;
      month = 3;
    }
    else
    {
      day -= 9;
      month = 4;

      if ((day==26)||((day==25)&&(d==28)&&(e==6)&&(a>10)))
      {
        day -= 7;
      }
    }

    auto t = date_to_unix(std::to_string(day) + "/" + std::to_string(month), "%d/%m") + 24*60*60;

    return unix_to_date(t, "%d/%m");
  }

  static std::map<std::string, std::string> IT_holiday({
    { "01/01", "Capodanno"},
    { "06/01", "Epifania"},
    { "25/04", "Liberazione"},
    { "01/05", "Festa del Lavoro"},
    { "02/06", "Festa della Repubblica"},
    { "15/09", "Ferragosto"},
    { "01/11", "Ognissanti"},
    { "08/12", "Immacolata Concezione"},
    { "25/12", "Natale"},
    { "26/12", "Santo Stefano"},
  });

  inline bool is_holiday_IT(const size_t &t_unix)
  {
    // sundays
    auto dm = unix_to_date(t_unix, "%w");
    if( dm == "0" )
      return true;

    // holidays
    dm = unix_to_date(t_unix, "%d/%m");
    for(const auto &p : physycom::IT_holiday)
      if(p.first == dm)
        return true;

    // little-easter
    auto y = std::stoi(unix_to_date(t_unix, "%Y"));
    if( dm == get_little_easter(y) )
      return true;

    return false;
  }

} // end namespace physycom

#endif //PHYSYCOM_UTILS_TIME_HPP
