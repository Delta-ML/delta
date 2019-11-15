# Contributing Guide

## License

The source file should contain a license header. See the existing files as the example.

## Name style
All name in python and cpp using [snake case style](https://en.wikipedia.org/wiki/Snake_case), except for `op` for `Tensorflow`.
For Golang, using Camel-Case for `variable name` and `interface`.

## Python style
Changes to Python code should conform the [Chromium Python Style Guide](https://chromium.googlesource.com/chromium/src/+/master/styleguide/python/python.md).  
You can use [yapf](https://github.com/google/yapf) to check the style.   
The style configuration is `.style.yapf`.   
You can using `tools/format.sh` tool to format code.

## C++ style
Changes to C++ code should conform to [Google C++ Style Guide](https://github.com/google/styleguide).   
You can use [cpplint](https://github.com/google/styleguide/tree/gh-pages/cpplint) to check the style and use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to format the code.  
The style configuration is `.clang-format`.   
You can using `tools/format.sh` tool to format code.

## C++ macro 
C++ macros should start with `DELTA_`, except for most common ones like `LOG` and `VLOG`.

## Golang style
For Golang styple, please see docs below:

* [How to Write Go Code](https://golang.org/doc/code.html)
* [Effective Go](https://golang.org/doc/effective_go.html#interface-names)
* [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
* [Golang Style in Chinese](https://juejin.im/post/5c16f16c5188252dcb30ff42)

Before commit golang code, plase using `go fmt` and `go vec` to format and lint code.

## Logging guideline

For `python` using [abseil-py](https://github.com/abseil/abseil-py), [more info](https://abseil.io/docs/python/).

For C++ using [abseil-cpp](https://github.com/abseil/abseil-cpp), [more info](https://abseil.io/docs/cpp/).

For Golang using [glog](https://github.com/golang/glog).

## Unit test

For `python` using `tf.test.TestCase`, and the entrypoint for python unittest is `tools/test/python_test.sh`.   

For C++ using [googletest](https://github.com/google/googletest), and the entrypoint for C++ unittest is `tools/test/cpp_test.sh`.

For Golang using `go test` for unittest.



