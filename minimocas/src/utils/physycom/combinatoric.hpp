#ifndef PHYSYCOM_UTILS_COMBINATORIC_HPP
#define PHYSYCOM_UTILS_COMBINATORIC_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

namespace physycom
{

  inline int factorial(const int &n)
  {
    if(n<0) throw std::runtime_error("physycom::factorial argument must be non-negative");
    int res = 1;
    for(int i=n; i>0; --i) res *= i;
    return res;
  }

  inline int binom(const int &n, const int &k)
  {
    if(n<0 && k<0 && n<k) throw std::runtime_error("physycom::binom argument must be non-negative with n >= k");
    int res = 1;
    for(int i=n; i>n-k; --i) res *= i;
    return res/factorial(k);
  }

  static struct
  {
    template<typename T>
    std::vector<T> operator()(T elem)
    {
      std::vector<T> tuple;
      std::sort(elem.begin(), elem.end());
      do
      {
        tuple.push_back(elem);
      }
      while( std::next_permutation(elem.begin(),elem.end()) );
      return tuple;
    }

  } permutations;

  static struct
  {
    std::vector<unsigned long> pos;

    template<typename T>
    std::vector<T> operator()(const T &elems, unsigned long k)
    {
      assert(k > 0 && k <= elems.size());
      std::vector<T> tuple;
      pos.resize(k);
      combinations_recursive(elems, k, 0, 0, tuple);
      return tuple;
    }

    template<typename T>
    void combinations_recursive(const T &elems, unsigned long k, unsigned long depth, unsigned long margin, std::vector<T> &tuple)
    {
      if (depth >= k)
      {
        tuple.push_back(T());
        for (unsigned long i = 0; i < pos.size(); ++i) tuple.back().push_back(elems[pos[i]]);
        return;
      }

      // magic condition which prevents from useless recursion
      if ((elems.size() - margin) < (k - depth))
        return;

      for (unsigned long i = margin; i < elems.size(); ++i)
      {
        pos[depth] = i;
        combinations_recursive(elems, k, depth + 1, i + 1, tuple);
      }

      return;
    }
  } combinations;

} // end of namespace physycom

#endif // PHYSYCOM_UTILS_COMBINATORIC_HPP
