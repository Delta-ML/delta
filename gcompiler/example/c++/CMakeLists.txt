
delta_fetch_files_with_suffix(${DELTA_INFER_ROOT}/example/c++ "cc" DELTA_INFER_EXAMPLES)
delta_fetch_files_with_suffix(${DELTA_INFER_ROOT}/example/c++/basic "cc" DELTA_INFER_EXAMPLES_BASIC)

message("tf libs:" ${DELTA_INFER_LINK_LIBS})

foreach(SRC_NAME ${DELTA_INFER_EXAMPLES})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
    list(GET SEXY_LIST -1 EXAMPLE_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${EXAMPLE_CASE_NAME})
    list(GET SEXY_LIST 0 EXAMPLE_CASE_NAME)
    find_package(Threads)

    message("xxx" ${DELTA_INFER_LINK_LIBS})

    add_executable(${EXAMPLE_CASE_NAME} ${SRC_NAME} ${DELTA_INFER_EXAMPLES_BASIC})
    #target_link_libraries(${EXAMPLE_CASE_NAME} "-Wl,--allow-multiple-definition -Wl,--whole-archive -Wl,--no-as-needed" ${DELTA_INFER_LINK_LIBS} -Wl,--no-whole-archive ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${EXAMPLE_CASE_NAME} ${DELTA_INFER_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    #target_link_options(${EXAMPLE_CASE_NAME} INTERFACE -Wl,--allow-multiple-definition -Wl,--whole-archive)
    #set_target_properties(${EXAMPLE_CASE_NAME} PROPERTIES LINK_FLAGS "-Wl,--allow-multiple-definition -Wl,--whole-archive")

    #cc_binary(${EXAMPLE_CASE_NAME} SRCS ${SRC_NAME} ${DELTA_INFER_EXAMPLES_BASIC} 
    #                               DEPS delta_infer custom_ops 
    #                               LINK_LIBS "-Wl,--allow-multiple-definition -Wl,--whole-archive" ${DELTA_INFER_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT}
    #                               )
endforeach()

