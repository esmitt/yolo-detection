cmake_minimum_required(VERSION 3.8)

project("Cvedia-Esmitt")

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/model)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories( ${CMAKE_BINARY_DIR}/bin)

# Source and header files
set(NAME_HEADER
    include/inferenceengine.h
    include/detectedobject.h
    include/guiengine.h
    include/opencvinferenceengine.h
    include/opencvguiengine.h
)

set(NAME_SRC
    main.cpp  
    src/opencvinferenceengine.cpp
    src/opencvguiengine.cpp
)

# Output directories
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Create executable
add_executable(Cvedia-Esmitt ${NAME_HEADER} ${NAME_SRC})

# Link libraries
target_link_libraries(Cvedia-Esmitt ${OpenCV_LIBS})
