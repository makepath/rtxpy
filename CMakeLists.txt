cmake_minimum_required(VERSION 3.10)

project(rtxpy)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DRTX_EXPORTS)

if (WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

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

add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
	${SOURCE_DIR}/optix_7.1
	${SOURCE_DIR}/cuew
)
