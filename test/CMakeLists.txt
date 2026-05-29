find_package(GTest CONFIG REQUIRED)

file(GLOB TEST_SOURCES "*.cpp")

add_executable(unit_tests ${TEST_SOURCES})

target_link_libraries(unit_tests PRIVATE 
    GTest::gtest_main 
    mor_core
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

add_test(NAME MOR_Unit_Tests COMMAND unit_tests)