
delta_fetch_files_with_suffix(${DELTA_INFER_DELTA_INFER}/test "cc" DELTA_INFER_TESTS)

foreach(SRC_NAME ${DELTA_INFER_TESTS})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
    list(GET SEXY_LIST -1 TEST_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
    list(GET SEXY_LIST 0 TEST_CASE_NAME)
    find_package(Threads)
    cc_binary(${TEST_CASE_NAME} SRCS ${SRC_NAME} DEPS delta_infer pywrap_tf_internal LINK_LIBS ${DELTA_INFER_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endforeach()

