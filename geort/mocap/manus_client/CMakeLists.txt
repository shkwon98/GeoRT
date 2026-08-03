cmake_minimum_required(VERSION 3.8)
project(manus_client)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# add_compile_options(-Wall -Wextra -Wpedantic) endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)
include_directories("$ENV{CONDA_PREFIX}/include")

add_executable(manus_left src/SDKMinimalClient.cpp
                          src/ClientPlatformSpecific.cpp)
add_executable(manus_right src/right_hand_ros.cpp
                           src/ClientPlatformSpecific.cpp)

# Link Manus SDK library to executable targets
find_library(MANUS_SDK ManusSDK HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib REQUIRED)

target_link_libraries(manus_left ${MANUS_SDK})
target_link_libraries(manus_right ${MANUS_SDK})

# Specify target dependencies
ament_target_dependencies(manus_left rclcpp std_msgs sensor_msgs)
ament_target_dependencies(manus_right rclcpp std_msgs sensor_msgs)

# Install targets
install(TARGETS manus_left manus_right DESTINATION lib/${PROJECT_NAME})

ament_package()
