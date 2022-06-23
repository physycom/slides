/*!
 *  \file   connection.hpp
 *  \author A. Fabbri (alessandro.fabbri27@unibo.it)
 *  \brief  A basic boost websocket server wrapper.
 *  \details This file contains the implementation of a basic Websocket server based on the boost/beast library.
 */

#ifndef _MINIMOCAS_CONNECTION_HPP_
#define _MINIMOCAS_CONNECTION_HPP_

#ifdef _WIN32
#define _WIN32_WINNT _WIN32_WINNT_WIN10
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#endif

#include <iostream>
#include <thread>

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <jsoncons/json.hpp>

#include <carto.h>
#include <simulation.hpp>

using tcp = boost::asio::ip::tcp;
namespace websocket = boost::beast::websocket;
using namespace jsoncons;

struct connection
{
  connection();
  connection(const jsoncons::json &jconf, cart *c);
};

void handle_connection(tcp::socket& socket, cart *c);  // forward declaration

connection::connection(const jsoncons::json &jconf, cart *c)
{
  auto const address = boost::asio::ip::make_address(jconf["server"]["ip"].as<std::string>());
  auto const port = jconf["server"]["port"].as<unsigned short>();
  std::cout << "Listening at " << address << ":" << port << std::endl;

  boost::asio::io_context ioc{1};
  tcp::acceptor acceptor{ioc, {address, port}};
  for(;;)
  {
    tcp::socket socket{ioc};
    acceptor.accept(socket);
    std::thread{std::bind(&handle_connection, std::move(socket), c)}.detach();
  }
}

void handle_connection(tcp::socket& socket, cart *c)
{
  try
  {
    websocket::stream<tcp::socket> ws{std::move(socket)};
    ws.accept();

    for(;;)
    {
      boost::beast::multi_buffer buffer;
      ws.read(buffer);

      std::ostringstream os;
      os << boost::beast::buffers(buffer.data());
      std::string msg = os.str();
      std::cout << "Message received : " << std::endl << msg << std::endl;

      json jconf = json::parse(msg);
      simulation s(jconf, c);
      s.run([&s, &ws, &buffer]()
      {
        buffer.consume(buffer.size());

        s.net_state = std::vector<int>(s.c->poly.size(), 0);
        for (const auto &type : s.pawns)
          for(const auto &p : type)
            ++s.net_state[p.current_poly];

        boost::beast::ostream(buffer) << s.sim_time << s.separator;
        for (const auto &p : s.net_state)
          boost::beast::ostream(buffer) << p << s.separator;
        boost::beast::ostream(buffer) << std::endl;

        ws.text(ws.got_text());
        ws.write(buffer.data());
      });

      buffer.consume(buffer.size());
      boost::beast::ostream(buffer) << "Simulation done";
      ws.text(ws.got_text());
      ws.write(buffer.data());
    }
  }
  catch(boost::system::system_error const& se)
  {
    // This indicates that the session was closed
    if(se.code() != websocket::error::closed)
      std::cerr << "Error: " << se.code().message() << std::endl;
  }
  catch(std::exception const& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

#endif // _MINIMOCAS_CONNECTION_HPP_
