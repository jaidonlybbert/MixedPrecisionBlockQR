include(CTest)
add_executable(test_mmult ${CMAKE_HOME_DIRECTORY}/tests/test_mmult.cpp)
add_test(NAME test_mmult COMMAND test_mmult)
