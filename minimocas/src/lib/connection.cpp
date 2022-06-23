/*!
 *  \file   connection.cpp
 *  \author A. Fabbri (alessandro.fabbri27@unibo.it)
 *  \brief  BRIEF
 *  \details DETAILS
 */

#include <sstream>

#include <connection.h>

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Exception.hpp>

std::string curlpp_get(const std::string &url)
{
  try
  {
    curlpp::Cleanup cleaner;
    curlpp::Easy request;
    request.setOpt(new curlpp::options::Url(url));
    std::stringstream ss;
    ss << request;
    return ss.str();
  }
  catch ( curlpp::LogicError & e )
  {
    std::cout << e.what() << std::endl;
  }
  catch ( curlpp::RuntimeError & e )
  {
    std::cout << e.what() << std::endl;
  }
  return std::string("");
}
