// Copyright 2013 Daniel Parker
// Distributed under Boost license

#include <jsoncons/json_event_reader.hpp>
#include <string>
#include <sstream>

using namespace jsoncons;

void pull_parser_example1()
{
    std::string s = R"(
    [
        {
            "enrollmentNo" : 100,
            "firstName" : "Tom",
            "lastName" : "Cochrane",
            "mark" : 55              
        },
        {
            "enrollmentNo" : 101,
            "firstName" : "Catherine",
            "lastName" : "Smith",
            "mark" : 95              
        },
        {
            "enrollmentNo" : 102,
            "firstName" : "William",
            "lastName" : "Skeleton",
            "mark" : 60              
        }
    ]
    )";

    std::istringstream is(s);

    json_event_reader event_reader(is);

    for (; !event_reader.done(); event_reader.next())
    {
        const auto& event = event_reader.current();
        switch (event.event_type())
        {
            case json_event_type::name:
                std::cout << event.as<std::string>() << ": ";
                break;
            case json_event_type::string_value:
                std::cout << event.as<std::string>() << "\n";
                break;
            case json_event_type::int64_value:
            case json_event_type::uint64_value:
                std::cout << event.as<std::string>() << "\n";
                break;
        }
    }
}

void pull_parser_examples()
{
    std::cout << "\nPull parser examples\n\n";
    pull_parser_example1();

    std::cout << "\n";
}

