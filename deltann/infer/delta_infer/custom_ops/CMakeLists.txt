
set(DELTA_INFER_CUSTOM_OPS_SRC "")
delta_fetch_include_recursively(${DELTA_INFER_DELTA_INFER}/custom_ops)

delta_fetch_files_with_suffix(${DELTA_INFER_DELTA_INFER}/custom_ops "cc" DELTA_INFER_CUSTOM_OPS_SRC)

if(WITH_CUDA)
    set(DELTA_INFER_CUSTOM_OPS_CU_SRC "")
    delta_fetch_files_with_suffix(${DELTA_INFER_DELTA_INFER}/custom_ops/platform/CUDA/ "cc" DELTA_INFER_CUSTOM_OPS_SRC) 
    delta_fetch_files_with_suffix(${DELTA_INFER_DELTA_INFER}/custom_ops/platform/CUDA/kernels/ "cu" DELTA_INFER_CUSTOM_OPS_CU_SRC) 
    set(FLAGS_BACKUP ${CMAKE_CXX_FLAGS}) 
    set(CMAKE_CXX_FLAGS "")
    if(BUILD_SHARED)
        CUDA_COMPILE(DELTA_INFER_CUSTOM_OPS_CU_OBJS SHARED ${DELTA_INFER_CUSTOM_OPS_CU_SRC})
    else()
        CUDA_COMPILE(DELTA_INFER_CUSTOM_OPS_CU_OBJS STATIC ${DELTA_INFER_CUSTOM_OPS_CU_SRC})
    endif()
    set(CMAKE_CXX_FLAGS ${FLAGS_BACKUP})
endif()

if(BUILD_X86)
    delta_fetch_files_with_suffix(${DELTA_INFER_DELTA_INFER}/custom_ops/platform/X86 "cc" DELTA_INFER_CUSTOM_OPS_SRC)
endif()

delta_msg(INFO STR "Get custom ops lib src:" ITEMS ${DELTA_INFER_CUSTOM_OPS_SRC})
delta_msg(INFO STR "Get custom ops lib cu:" ITEMS ${DELTA_INFER_CUSTOM_OPS_CU_SRC})

if(BUILD_SHARED)
    cc_library(custom_ops SHARED 
                          SRCS ${DELTA_INFER_CUSTOM_OPS_SRC} ${DELTA_INFER_CUSTOM_OPS_CU_OBJS} 
                          DEPS pywrap_tf_internal LINK_LIBS ${DELTA_INFER_LINK_LIBS})
else()
    cc_library(custom_ops STATIC 
                          SRCS ${DELTA_INFER_CUSTOM_OPS_SRC} ${DELTA_INFER_CUSTOM_OPS_CU_OBJS} 
                          DEPS pywrap_tf_internal LINK_LIBS ${DELTA_INFER_LINK_LIBS})
endif()
