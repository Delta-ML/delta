# Settings for Android.
ifeq ($(TARGET), android)
	ifeq ($(TARGET_ARCH), armv7)
        CXX := arm-linux-androideabi-g++
        CC := arm-linux-androideabi-gcc
        AR := arm-linux-androideabi-ar

        CXXFLAGS += --sysroot=/nfs/project/gaoyonghu/tools/android-toolchain-arm-android-21/sysroot \
            -isystem /nfs/project/gaoyonghu/tools/android-toolchain-arm-android-21/include/c++/4.9.x \
            -D__ANDROID_API__=21 \
            -march=armv7-a \
            -std=c++11 \
            -mfpu=neon \
            -mfloat-abi=softfp \
            -fPIC \
            -fPIE \
            -pie \
            -Wl,-plugin-opt=-emulated-tls \
            -fvisibility=default \
            -ffunction-sections \
            -fdata-sections \
            -DNDEBUG \
            -Os

        LDFLAGS += -rdynamic -pie -rdynamic -fPIE \
            -Wl,--gc-sections -llog -s

        STDLIB += -Wl,--gc-sections \
            -Wl,--as-needed \
            -Wl,--exclude-libs,ALL \
            -lc++_shared -llog -s

        EXT_LIBS += -L/nfs/project/gaoyonghu/tf-feature-extraction/gen/android_armv7/lib -lfeature_extraction \
                    -L$(TFLIT_ROOT)/bazel-bin/tensorflow/contrib/lite/experimental/c -ltensorflowlite_c
    endif

	ifeq ($(TARGET_ARCH), armv8)
        CXX := aarch64-linux-android-g++
        CC := aarch64-linux-android-gcc
        AR := aarch64-linux-android-ar

        CXXFLAGS += --sysroot=/nfs/project/gaoyonghu/tools/android-toolchain-arm64-android-21/sysroot \
            -isystem /nfs/project/gaoyonghu/tools/android-toolchain-arm64-android-21/include/c++/4.9.x \
            -march=armv8-a \
            -D__ANDROID_API__=21 \
            -std=c++11 \
            -fPIE \
            -pie \
            -fvisibility=default \
            -DNDEBUG \
            -Os

        LDFLAGS += -rdynamic -pie -rdynamic -fPIE  -Wl,--gc-sections

        EXT_LIBS += -L/nfs/project/gaoyonghu/tf-feature-extraction/gen/android_armv8/lib -lfeature_extraction \
                    -L$(TFLIT_ROOT)/bazel-bin/tensorflow/contrib/lite/experimental/c -ltensorflowlite_c

        STDLIB += -Wl,--gc-sections \
            -Wl,--as-needed \
            -Wl,--exclude-libs,ALL \
            -lc++_shared -llog -s
    endif
endif
