cmake_minimum_required(VERSION 3.16)

project(vk_hwrt)

message("Compiling Shaders:")

set(GLSL_COMPILER "$ENV{VULKAN_SDK}/Bin/glslc.exe")

file(GLOB_RECURSE GLSL_SRC_FILES "src/shaders/*")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/shaders/")

source_group("Shader Files" FILES ${GLSL_SRC_FILES})

foreach(GLSL_FILE ${GLSL_SRC_FILES})
	get_filename_component(GLSL_FILE_NAME ${GLSL_FILE} NAME)
	set(SPIRV_FILE "${PROJECT_BINARY_DIR}/shaders/${GLSL_FILE_NAME}.spv")
	execute_process(COMMAND ${GLSL_COMPILER} "-g" ${GLSL_FILE} -o ${SPIRV_FILE} "--target-spv=spv1.4")
endforeach(GLSL_FILE)

message("Building Project:")

include_directories(src)
file(GLOB_RECURSE SRC_FILES "src/*.h" "src/*.cpp")
add_executable(${PROJECT_NAME} ${SRC_FILES} ${GLSL_SRC_FILES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:DEBUG_>)
target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:RELEASE_>)

target_precompile_headers(${PROJECT_NAME} PRIVATE src/pch.h)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /wd26812)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/"
		COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/"
		COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_BINARY_DIR}/shaders" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders")
endif()

message("Adding glfw:")

set(GLFW_DIR 3rdparty/glfw)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
set(GLFW_INSTALL OFF CACHE BOOL "")

add_subdirectory(${GLFW_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

set_property(TARGET glfw PROPERTY FOLDER "3rdparty/glfw")
set_property(TARGET update_mappings PROPERTY FOLDER "3rdparty/glfw")

message("Adding glm:")

set(GLM_DIR 3rdparty/glm)

target_include_directories(${PROJECT_NAME} PRIVATE ${GLM_DIR})

message("Adding volk:")

set(VOLK_DIR 3rdparty/volk)

add_subdirectory(${VOLK_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${VOLK_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE volk)

set_property(TARGET volk PROPERTY FOLDER "3rdparty")

message("Adding stb:")

set(STB_DIR 3rdparty/stb)

target_include_directories(${PROJECT_NAME} PRIVATE ${STB_DIR})

message("Adding tinyobjloader:")

set(TOL_DIR 3rdparty/tinyobjloader)

add_subdirectory(${TOL_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${TOL_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE tinyobjloader)

set_property(TARGET tinyobjloader PROPERTY FOLDER "3rdparty")
