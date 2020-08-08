include(FetchContent)

set(PYBIND11_DIR ${DELTA_INFER_ROOT}/third_party/pybind11)
FetchContent_Declare(
  pybind11 
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        master 
  DOWNLOAD_DIR  ${PYBIND11_DIR}
  SOURCE_DIR    ${PYBIND11_DIR}
)

delta_msg(INFO STR "pybind11: ${PYBIND11_DIR}")
FetchContent_MakeAvailable(pybind11)
