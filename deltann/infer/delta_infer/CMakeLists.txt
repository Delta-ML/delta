
set(DELTA_INFER_SRC "")
delta_fetch_include_recursively(${DELTA_INFER_DELTA_INFER})

add_subdirectory(${DELTA_INFER_DELTA_INFER}/custom_grappler/)

if(BUILD_DELTA_INFER_CUSTOM_OPS)
    add_subdirectory(${DELTA_INFER_DELTA_INFER}/custom_ops/)
endif()

add_subdirectory(${DELTA_INFER_DELTA_INFER}/core/)

add_subdirectory(${DELTA_INFER_DELTA_INFER}/scheduler/)

#delta_msg(INFO STR "Get src: ${DELTA_INFER_SRC}")

if(BUILD_SHARED)
    cc_library(delta_infer SHARED SRCS ${DELTA_INFER_SRC} DEPS pywrap_tf_internal LINK_LIBS ${DELTA_INFER_LINK_LIBS})
else()
    cc_library(delta_infer STATIC SRCS ${DELTA_INFER_SRC} DEPS pywrap_tf_internal LINK_LIBS ${DELTA_INFER_LINK_LIBS})
endif()

# putting test after lib is compiled.
#add_subdirectory(${DELTA_INFER_DELTA_INFER}/test/)
