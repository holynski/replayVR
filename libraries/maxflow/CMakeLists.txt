project(maxflow)

include_directories(./)
set(maxflow_sources
  graph.cpp
  maxflow.cpp)

add_library(maxflow STATIC ${maxflow_sources})
