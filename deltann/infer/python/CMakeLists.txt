
set(DELTA_INFER_EXPORT_TO_PY_SRC "")
delta_fetch_include_recursively(${DELTA_INFER_ROOT}/python)

delta_fetch_files_with_suffix(${DELTA_INFER_ROOT}/python/delta_infer/cpp/ "cc" DELTA_INFER_EXPORT_TO_PY_SRC)

delta_fetch_files_with_suffix(${DELTA_INFER_DELTA_INFER}/custom_grappler "cc" DELTA_INFER_SRC)
delta_fetch_files_with_suffix(${DELTA_INFER_DELTA_INFER}/core "cc" DELTA_INFER_SRC)

#delta_msg(INFO STR "Get src:${DELTA_INFER_SRC}")

if(MSVC)
  set(PY_SUFFIX ".pyd")
else()
  set(PY_SUFFIX ".so")
endif()

# pybind11 is a header only lib, so we import it directly.
add_subdirectory(${DELTA_INFER_ROOT}/third_party/pybind11 binary_dir)
pybind11_add_module(export_py SHARED ${DELTA_INFER_EXPORT_TO_PY_SRC} ${DELTA_INFER_SRC})
add_dependencies(export_py pywrap_tf_internal)
target_link_libraries(export_py LINK_PRIVATE pywrap_tf_internal)
target_link_libraries(export_py LINK_PRIVATE ${DELTA_INFER_LINK_LIBS})

if(EXISTS ${DELTA_INFER_ROOT}/third_party/pybind11/include/pybind11/pybind11.h)
    target_include_directories(export_py PUBLIC ${DELTA_INFER_ROOT}/third_party/pybind11/include) 
    target_include_directories(export_py PUBLIC ${pybind11_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "cannot find pybind")
endif()

set_target_properties(export_py PROPERTIES PREFIX "")
set_target_properties(export_py PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
set_target_properties(export_py PROPERTIES SUFFIX ${PY_SUFFIX})
#set_target_properties(export_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/)
target_include_directories(export_py PRIVATE
                           $<BUILD_INTERFACE:${DELTA_INFER_ROOT}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include>
                           ${PYTHON_INCLUDE_DIR})
