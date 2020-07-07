macro(add_deps_of_tf lib_name lib_path)
    add_library(${lib_name} SHARED IMPORTED)
    set_property(TARGET ${lib_name} PROPERTY IMPORTED_LOCATION ${lib_path})
endmacro()

macro(find_python)
    find_package (Python COMPONENTS Interpreter Development)
    if(Python_FOUND)
        include_directories(${Python_INCLUDE_DIRS})
        list(APPEND DELTA_INFER_LINK_LIBS ${Python_LIBRARIES})
    else()
        find_program(PY_EXE python)
        string(FIND ${PY_EXE} python __find_out)
        if(NOT (${__find_out} EQUAL -1))
            delta_msg(INFO STR "Find python in ${PY_EXE}")
            get_filename_component(PYTHON_PATH PY_EXE DIRECTORY)    
            exec_program(ls ${PYTHON_PATH}/../lib/ ARGS "libpython*.so" 
                                           OUTPUT_VARIABLE OUTPUT 
                                           RETURN_VALUE VALUE)
            if(NOT VALUE)
		        string(REPLACE "\n" ";" OUTPUT_LIST "${OUTPUT}")
                foreach(var ${OUTPUT_LIST})
                    delta_msg(INFO STR "Find python lib: ${PYTHON_PATH}/../lib/${var}")
                    list(APPEND DELTA_INFER_LINK_LIBS ${PYTHON_PATH}/../lib/${var})
		        endforeach()
            endif()
            include_directories(${PYTHON_PATH}/../include/python*)
        else()
            find_package(PythonLibs)
            if(PYTHONLIBS_FOUND)
                delta_msg(WARN STR "Find python : ${PYTHON_INCLUDE_DIR} \n ${PYTHON_LIBRARY}")
                include_directories(${PYTHON_INCLUDE_DIR})
                list(APPEND DELTA_INFER_LINK_LIBS ${PYTHON_LIBRARY})
            else()
                delta_msg(ERROR STR "Python module not found!")
            endif()
        endif()
    endif()
endmacro()

macro(detect_deps lib_path lib_name)
    exec_program(ldd ${lib_path}
                 ARGS "${lib_name}"
                 OUTPUT_VARIABLE OUTPUT
                 RETURN_VALUE VALUE)
    if(NOT VALUE) 
        string(REPLACE " " ";" OUTPUT_LIST "${OUTPUT}")
        foreach(__str ${OUTPUT_LIST}) 
            foreach(__target ${ARGN})
                string(FIND ${__str} ${__target} __out_1)
                string(FIND ${__str} "/" __out_2)
                if((NOT (${__out_1} EQUAL -1)) AND (NOT (${__out_2} EQUAL -1)))
                    list(APPEND DELTA_INFER_LINK_LIBS ${__str})
                    delta_msg(INFO STR "Find dependent lib ${__target}:" ITEMS "path:${__str}")
                endif()
            endforeach()
        endforeach()
    else()
        delta_msg(WARN STR "error running in detect_deps")
    endif()
    unset(__str)
    unset(__target)
    unset(__out_1)
    unset(__out_2)
endmacro()

function(find_tf)
    exec_program(python .
                 ARGS "-c \"import tensorflow as tf;\
                            print(\' %s %s %s\' % (\
                                    tf.__version__,\
                                    tf.sysconfig.get_include(),\
                                    tf.sysconfig.get_lib())\
                                 );\""
                 OUTPUT_VARIABLE OUTPUT
                 RETURN_VALUE VALUE)
    if(NOT VALUE)
        string(REGEX REPLACE "[\]\r\n\'\:\;\[]+" "" NEW_OUTPUT ${OUTPUT})
        delta_msg(INFO STR "ccw: ${NEW_OUTPUT}")
        string(REPLACE " " ";" OUTPUT_LIST ${NEW_OUTPUT})
        #delta_msg(INFO ITEMS ${OUTPUT_LIST})
        list(GET OUTPUT_LIST -3 TF_VERSION)
        list(GET OUTPUT_LIST -2 TF_INCLUDE_PATH)
        list(GET OUTPUT_LIST -1 TF_LIB_PATH)
        include_directories(${TF_INCLUDE_PATH})

        delta_fetch_files_with_suffix(${TF_LIB_PATH} "so*" LIB_TF_NAMES) 
        list(GET LIB_TF_NAMES 0 LIB_TF_PATH)
        list(APPEND DELTA_INFER_LINK_LIBS ${LIB_TF_PATH})
        delta_msg(INFO STR "${LIB_TF_PATH}")
        detect_deps(/ ${LIB_TF_PATH} libiomp5.so libmklml_intel.so)
        #if(TF_VERSION VERSION_LESS "2.0.0") 
        #    list(APPEND DELTA_INFER_LINK_LIBS ${TF_LIB_PATH}/libtensorflow_framework.so.1)
        #    if(TF_VERSION VERSION_LESS "1.9.0")
        #        detect_deps(${TF_LIB_PATH}/ libtensorflow_framework.so libiomp5.so libmklml_intel.so)
        #    else()
        #        detect_deps(${TF_LIB_PATH}/ libtensorflow_framework.so.1 libiomp5.so libmklml_intel.so)
        #    endif()
        #else()
        #    # try to use old abi version
        #    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        #    list(APPEND DELTA_INFER_LINK_LIBS ${TF_LIB_PATH}/libtensorflow_framework.so.2)
        #    detect_deps(${TF_LIB_PATH}/ libtensorflow_framework.so.2 libiomp5.so libmklml_intel.so)
        #endif()

        # pywrap_tf_internal contain the registered session
        add_deps_of_tf(pywrap_tf_internal ${TF_LIB_PATH}/python/_pywrap_tensorflow_internal.so)
        #add_deps_of_tf(fast_tensor_util ${TF_LIB_PATH}/python/framework/fast_tensor_util.so)

        set(DELTA_INFER_LINK_LIBS ${DELTA_INFER_LINK_LIBS} PARENT_SCOPE)
        delta_msg(INFO STR "Detect tensorflow(${TF_VERSION}) in " 
                       ITEMS "(header)${TF_INCLUDE_PATH};(lib)${TF_LIB_PATH}")
    else()
        delta_msg(WARN STR "You should install suitable tensorflow (version >= 1.14.0)")
    endif()
endfunction()

find_tf()
find_python()
