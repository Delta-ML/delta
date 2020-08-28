apt-get update && apt-get install -y --no-install-recommends \
        autoconf\
        automake \
        clang-format \
        curl \
        git \
        libtool \
        sudo \
        sox \
        tig \
        make \
        vim \
        zlib1g-dev \
        wget \
        subversion \
        ca-certificates \
        unzip \
        patch \
        ffmpeg \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
