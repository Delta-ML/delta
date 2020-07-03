
include_directories(/nfs/ml-k80-ser017/cuichaowen/work/pkgtool/elf)
delta_fetch_files_with_suffix(${DELTA_INFER_ROOT}/example/c++ "cc" DELTA_INFER_EXAMPLES)
delta_fetch_files_with_suffix(${DELTA_INFER_ROOT}/example/c++/basic "cc" DELTA_INFER_EXAMPLES_BASIC)


foreach(SRC_NAME ${DELTA_INFER_EXAMPLES})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
    list(GET SEXY_LIST -1 EXAMPLE_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${EXAMPLE_CASE_NAME})
    list(GET SEXY_LIST 0 EXAMPLE_CASE_NAME)
    find_package(Threads)
    cc_binary(${EXAMPLE_CASE_NAME} SRCS ${SRC_NAME} ${DELTA_INFER_EXAMPLES_BASIC} 
                                   DEPS delta_infer custom_ops 
                                   LINK_LIBS ${DELTA_INFER_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endforeach()

