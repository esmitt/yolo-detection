# CMakeList.txt : CMake project for Cvedia-Esmitt, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Cvedia-Esmitt")
set(CMAKE_CXX_STANDARD 17) 
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

#if (CMAKE_VERSION VERSION_GREATER 3.12)
#  set_property(TARGET Cvedia-Esmitt PROPERTY CXX_STANDARD 17)
#endif()


FIND_PACKAGE( OpenCV REQUIRED )  
set( NAME_SRC
    Cvedia-Esmitt.cpp  
)
                           
#INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
# Add source to this project's executable.
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( Cvedia-Esmitt ${NAME_SRC} )

TARGET_LINK_LIBRARIES (Cvedia-Esmitt ${OpenCV_LIBS})

# TODO: Add tests and install targets if needed.
