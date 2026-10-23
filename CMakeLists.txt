cmake_minimum_required(VERSION 3.16)
project(task5_armor_detection LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)
find_package(Threads REQUIRED)

add_executable(armor_detection
    apps/main.cpp
    src/VideoReader.cpp
    src/Vision.cpp
    include/VideoReader.h
    include/Vision.h
)

add_executable(preprocess_viewer
    apps/preprocess_viewer.cpp
    src/Vision.cpp
    include/Vision.h
)

foreach(target_name IN ITEMS armor_detection preprocess_viewer)
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
    )

    target_compile_options(${target_name} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wsuggest-override
    )

    target_link_libraries(${target_name} PRIVATE
        ${OpenCV_LIBS}
    )
endforeach()

target_link_libraries(armor_detection PRIVATE Threads::Threads)
