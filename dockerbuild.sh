#!/bin/bash

cd python_runtime || exit 1

# Default values
useProposed=""
version="v1.0"
proposed="-p"

# Parse arguments
while getopts ":pv:" opt; do
  case $opt in
    p)
      useProposed="-p"
      ;;
    v)
      version="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Compose the docker tag
dockerTag="pandeymanish93/ml-inference:alexnet-${version}"
echo "pandeymanish93/ml-inference:alexnet-${version} ${useProposed}"
./tutorials/local_build_ml.sh -r alexnet      -t pandeymanish93/ml-inference:alexnet-v1.0
# ./tutorials/local_build_ml.sh -r alexnet      -t pandeymanish93/ml-inference:alexnet-v1.0
# ./tutorials/local_build_ml.sh -r googlenet      -t pandeymanish93/ml-inference:googlenet-v1.0
# ./tutorials/local_build_ml.sh -r resnet50      -t pandeymanish93/ml-inference:resnet50-v1.0
# ./tutorials/local_build_ml.sh -r efficientnet      -t pandeymanish93/ml-inference:efficientnet-v1.0
# ./tutorials/local_build_ml.sh -r inception      -t pandeymanish93/ml-inference:inception-v1.0
# ./tutorials/local_build_ml.sh -r zygote      -t pandeymanish93/ml-inference:zygote-v1.0
# ./tutorials/local_build_ml.sh -r bert         -t pandeymanish93/ml-inference:bert-v1.0
# ./tutorials/local_build_ml.sh -r distilgpt2   -t pandeymanish93/ml-inference:distilgpt2-v1.0

# ./tutorials/local_build_ml.sh -r googlenet    -t pandeymanish93/ml-inference:googlenet-$version $useProposed
# ./tutorials/local_build_ml.sh -r resnet50     -t pandeymanish93/ml-inference:resnet50-$version $useProposed
# ./tutorials/local_build_ml.sh -r efficientnet -t pandeymanish93/ml-inference:efficientnet-$version $useProposed
# ./tutorials/local_build_ml.sh -r inception    -t pandeymanish93/ml-inference:inception-$version $useProposed
# ./tutorials/local_build_ml.sh -r bert         -t pandeymanish93/ml-inference:bert-$version $useProposed
# ./tutorials/local_build_ml.sh -r distilgpt2   -t pandeymanish93/ml-inference:distilgpt2-$version $useProposed

# ./tutorials/local_build_ml.sh -r alexnet      -t pandeymanish93/ml-inference:alexnet-v1.2 $proposed
# ./tutorials/local_build_ml.sh -r googlenet      -t pandeymanish93/ml-inference:googlenet-v1.2 $proposed
# ./tutorials/local_build_ml.sh -r resnet50      -t pandeymanish93/ml-inference:resnet50-v1.2 $proposed
# ./tutorials/local_build_ml.sh -r efficientnet      -t pandeymanish93/ml-inference:efficientnet-v1.2 $proposed
# ./tutorials/local_build_ml.sh -r inception      -t pandeymanish93/ml-inference:inception-v1.2 $proposed
# ./tutorials/local_build_ml.sh -r bert         -t pandeymanish93/ml-inference:bert-v1.2 $useProposed
# ./tutorials/local_build_ml.sh -r distilgpt2   -t pandeymanish93/ml-inference:distilgpt2-v1.2 $useProposed

# ./tutorials/local_build_ml.sh -r zygote      -t pandeymanish93/ml-inference:zygote-v1.2 $proposed
