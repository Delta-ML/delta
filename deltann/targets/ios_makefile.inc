# Settings for IOS.
ifeq ($(TARGET), ios)
  BUILD_FOR_IOS_SIMULATOR := false
	ifeq ($(TARGET_ARCH), x86_64)
	  BUILD_FOR_IOS_SIMULATOR := true
	endif
	ifeq ($(TARGET_ARCH), i386)
	  BUILD_FOR_IOS_SIMULATOR := true
	  CXXFLAGS += -Dthread_local=
	endif
	ifeq ($(BUILD_FOR_IOS_SIMULATOR), true)
		IPHONEOS_PLATFORM := $(shell xcrun --sdk iphonesimulator \
			--show-sdk-platform-path)
		IPHONEOS_SYSROOT := $(shell xcrun --sdk iphonesimulator \
			--show-sdk-path)
	else
		IPHONEOS_PLATFORM := $(shell xcrun --sdk iphoneos --show-sdk-platform-path)
		IPHONEOS_SYSROOT := $(shell xcrun --sdk iphoneos --show-sdk-path)
	endif
	IOS_SDK_VERSION := $(shell xcrun --sdk iphoneos --show-sdk-version)
    MIN_SDK_VERSION := 8.0
    CXX := g++
    CC := gcc
    AR := ar

    CXXFLAGS += -std=c++11 -miphoneos-version-min=${MIN_SDK_VERSION} \
            -fembed-bitcode \
	    -Wno-c++11-narrowing \
	    -mno-thumb \
	    -isysroot ${IPHONEOS_SYSROOT} \
	    -arch $(TARGET_ARCH) \
            -O3
    CCFLAGS += -miphoneos-version-min=${MIN_SDK_VERSION} \
		-isysroot ${IPHONEOS_SYSROOT} \
		-fembed-bitcode \
		-arch $(TARGET_ARCH) \
		-O3
    LDFLAGS += -fembed-bitcode \
		-miphoneos-version-min=${MIN_SDK_VERSION} \
		-arch ${TARGET_ARCH} \
		-framework Accelerate #-stdlib=libc++ -lc++ -lc++abi

endif
