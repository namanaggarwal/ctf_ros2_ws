ARG IMAGE_NAME
FROM ${IMAGE_NAME}:13.2.1-runtime-ubuntu24.04 AS base

FROM base AS base-amd64

ENV NV_CUDNN_VERSION=9.20.0.48-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-13
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-13=${NV_CUDNN_VERSION}

FROM base AS base-arm64

ENV NV_CUDNN_VERSION=9.20.0.48-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-13
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-13=${NV_CUDNN_VERSION}
FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

    

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*
