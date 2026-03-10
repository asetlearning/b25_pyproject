pybind11_add_module(tcgraph_ext bindings.cpp)
target_link_libraries(tcgraph_ext PRIVATE tc_graph)

install(TARGETS tcgraph_ext DESTINATION .)
