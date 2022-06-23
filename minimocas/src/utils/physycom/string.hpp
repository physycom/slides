/* Copyright 2017 - Alessandro Fabbri */

// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt


#ifndef PHYSYCOM_UTILS_SPLIT_HPP
#define PHYSYCOM_UTILS_SPLIT_HPP

#include <string>
#include <vector>
#include <cctype>
#include <algorithm>

namespace physycom{

  enum
  {
    token_compress_on,
    token_compress_off
  };

  template<typename E, typename C> inline bool belongs_to(const E& e, const C& c) {
    for (size_t i = 0; i < c.size(); i++) { if (e == c.at(i)) return true; }
    return false;
  }

  template<typename T>
  void split(std::vector<T>& tok, const T &str, const T &sep, const int &mode = token_compress_off){
    tok.clear();
    if( str.size() ){
      auto start = str.begin();
      for( auto it = str.begin(); it != str.end(); it++ ){
        if( belongs_to(*it, sep) ){
          tok.push_back( std::string(start, it) );
          start = it + 1;
        }
      }
      tok.push_back( std::string(start, str.end() ) );

      if( mode == token_compress_on ){
        auto it = tok.begin();
        while( it != tok.end() ){
          if( *it == "" ) it = tok.erase(it);
          else it++;
        }
      }
    }
  }

  template<typename T> inline T stoa(const std::string &s){}

  template<> inline double stoa(const std::string &s)
  {
    return stod(s);
  }

  template<> inline int stoa(const std::string &s)
  {
    return stoi(s);
  }

  template<typename T>
  inline std::vector<T> stov(const std::string &s, const std::string &sep = ",")
  {
    std::vector<T> v;
    std::vector<std::string> tok;
    physycom::split(tok, s, sep, physycom::token_compress_on);
    for(const auto &t : tok) v.push_back(stoa<T>(t));
  }

  static inline void trim_inplace(std::string &s)
  {
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(), [](char c){
              return !std::isspace(c);
            }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](char c){
              return !std::isspace(c);
            }).base(),
            s.end());
  }

  static inline std::string trim(const std::string &s)
  {
    std::string trims(s);
    trim_inplace(trims);
    return trims;
  }

}   // end namespace physycom

#endif // PHYSYCOM_SPLIT_HPP
