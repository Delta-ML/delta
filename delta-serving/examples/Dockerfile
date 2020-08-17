# Dockerfile References: https://docs.docker.com/engine/reference/builder/

# Start from the latest golang base image
FROM golang:latest

WORKDIR /go/delta

# Build the Go app
#RUN bash build.sh

COPY ./output ./output

# Expose port 8004 to the outside world
EXPOSE 8004

# Command to run the executable
CMD ["pushd  output/delta-service"]
CMD ["./run.sh start &"]
CMD ["popd"]
