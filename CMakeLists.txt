cmake_minimum_required(VERSION 3.10)
#set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})

project(rtx)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DRTX_EXPORTS)

SET(SOURCE_DIR "crtx")

set(HEADERS
	${SOURCE_DIR}/common.h
	${SOURCE_DIR}/internal.h
	${SOURCE_DIR}/rtx.h
)

set(SOURCES
	${SOURCE_DIR}/dllmain.cpp
	${SOURCE_DIR}/cuew/cuew.c
)

#message(INFO "--- ${HEADERS}")
#message(INFO "--- ${SOURCES}")

add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES})

#set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "rtx.h")

target_include_directories(${PROJECT_NAME} PUBLIC
	${SOURCE_DIR}/optix_7.1
	${SOURCE_DIR}/cuew
)

#install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION bin)
#install(TARGETS ${PROJECT_NAME} PUBLIC_HEADER DESTINATION bin)
