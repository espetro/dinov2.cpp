# Tier-2 HTTP embeddings microservice. Reachable only with
# -DDINOV2_BUILD_SERVER=ON (see the root CMakeLists). Consumes the public C
# API plus the internal src/image.h decode helper; nothing here is linked or
# referenced by tier-1 targets. httplib.h is a vendored single-header copy of
# cpp-httplib v0.57.1 (MIT), same pattern as the stb headers in src/.

add_executable(dinov2-server dinov2-server.cpp)
target_link_libraries(dinov2-server PRIVATE dinov2)
target_include_directories(dinov2-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src)
dinov2_target_flags(dinov2-server)

# httplib's thread pool needs a threads library on unix
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(dinov2-server PRIVATE Threads::Threads)
