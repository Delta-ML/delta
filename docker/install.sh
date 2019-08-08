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
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
