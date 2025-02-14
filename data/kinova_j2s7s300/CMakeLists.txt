cmake_minimum_required(VERSION 3.5)
project(kinova_j2s7s300_ign)

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY urdf kinova_j2s7s300/meshes DESTINATION share/${PROJECT_NAME})

# Setup the project
ament_package()
