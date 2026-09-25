#!/bin/bash

# Get the action type
ACTION=$1
shift

if [[ "$ACTION" != "update" && "$ACTION" != "invoke" && "$ACTION" != "update_all" && "$ACTION" != "invoke_all" ]]; then
    echo "Usage: $0 {update|invoke|video-update|update_all} [--device=cpu|gpu] [--multi_gpu=True|False] [--version=vX.Y]"
    exit 1
fi

# Default values
DEVICE="cpu"
MULTI_GPU="False"
VERSION="v1.0"  # default version

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device=*)
            DEVICE="${1#*=}"
            ;;
        --multi_gpu=*)
            MULTI_GPU="${1#*=}"
            ;;
        --version=*)
            VERSION="${1#*=}"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 {update|invoke|video-update|update_all} [--device=cpu|gpu] [--multi_gpu=True|False] [--version=vX.Y]"
            exit 1
            ;;
    esac
    shift
done
# Normalize multi_gpu to lowercase
MULTI_GPU=$(echo "$MULTI_GPU" | tr '[:upper:]' '[:lower:]')
echo $MULTI_GPU
echo "Device: $DEVICE"

cd python_runtime || exit

case "$ACTION" in
    update)
        echo "Running update with version=$VERSION..."
        wsk action update alexnet core/mlruntime/alexnet/alexnet.py --docker=pandeymanish93/ml-inference:alexnet-$VERSION --memory 1024
        wsk action update resnet50 core/mlruntime/resnet50/resnet50.py --docker=pandeymanish93/ml-inference:resnet50-$VERSION --memory 1024
        wsk action update efficientnet core/mlruntime/efficientnet/efficientnet.py --docker=pandeymanish93/ml-inference:efficientnet-$VERSION --memory 1524
        wsk action update googlenet core/mlruntime/googlenet/googlenet.py --docker=pandeymanish93/ml-inference:googlenet-$VERSION --memory 1024
        wsk action update inception core/mlruntime/inception/inception.py --docker=pandeymanish93/ml-inference:inception-$VERSION --memory 1024
        wsk action update bert core/mlruntime/bert/bert.py --docker=pandeymanish93/ml-inference:bert-$VERSION --memory 1024
        wsk action update distilgpt2 core/mlruntime/distilgpt2/distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-$VERSION --memory 1024
        ;;
    invoke)
        echo "Invoking actions with device=$DEVICE and multi_gpu=$MULTI_GPU..."
        wsk action invoke resnet50 --param device "$DEVICE" --param multi_gpu "$MULTI_GPU"
        wsk action invoke alexnet --param device "$DEVICE" --param multi_gpu "$MULTI_GPU"
        wsk action invoke efficientnet --param device "$DEVICE" --param multi_gpu "$MULTI_GPU"
        wsk action invoke googlenet --param device "$DEVICE" --param multi_gpu "$MULTI_GPU" 
        wsk action invoke inception --param device "$DEVICE" --param multi_gpu "$MULTI_GPU"
        wsk action invoke bert --param device "$DEVICE"
        wsk action invoke distilgpt2 --param device "$DEVICE"
        ;;
    video-update)
        echo "Running video-update with version=$VERSION..."
        wsk action update video-classification core/mlruntime/alexnet/alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-$VERSION --memory 1024
        ;;
    update_all)
        echo "Running vide and video-update for all version"
        wsk action update alexnet core/mlruntime/alexnet/alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.0 --memory 1024 --timeout 300000
        wsk action update alexnet_p core/mlruntime/alexnet/alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.2 --memory 1024 --timeout 300000

        wsk action update efficientnet core/mlruntime/efficientnet/efficientnet_video.py --docker=pandeymanish93/ml-inference:efficientnet-v1.0 --memory 1024 --timeout 300000
        wsk action update efficientnet_p core/mlruntime/efficientnet/efficientnet_video.py --docker=pandeymanish93/ml-inference:efficientnet-v1.2 --memory 1024 --timeout 300000

        wsk action update googlenet core/mlruntime/googlenet/googlenet_b64.py --docker=pandeymanish93/ml-inference:googlenet-v1.0 --memory 1024 --timeout 300000
        wsk action update googlenet_p core/mlruntime/googlenet/googlenet_b64.py --docker=pandeymanish93/ml-inference:googlenet-v1.2 --memory 1024 --timeout 300000

        wsk action update resnet50 core/mlruntime/resnet50/resnet50.py --docker=pandeymanish93/ml-inference:resnet50-v1.0 --memory 1024 --timeout 300000
        wsk action update resnet50_p core/mlruntime/resnet50/resnet50.py --docker=pandeymanish93/ml-inference:resnet50-v1.2 --memory 1024 --timeout 300000

        wsk action update inception core/mlruntime/inception/inception.py --docker=pandeymanish93/ml-inference:inception-v1.0 --memory 1024 --timeout 300000
        wsk action update inception_p core/mlruntime/inception/inception.py --docker=pandeymanish93/ml-inference:inception-v1.2 --memory 1024 --timeout 300000

        wsk action update bert core/mlruntime/bert/bert.py --docker=pandeymanish93/ml-inference:bert-v1.0 --memory 1024 --timeout 300000
        wsk action update bert_p core/mlruntime/bert/bert.py --docker=pandeymanish93/ml-inference:bert-v1.2 --memory 1024 --timeout 300000

        wsk action update distilgpt2 core/mlruntime/distilgpt2/distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-v1.0 --memory 1024 --timeout 300000
        wsk action update distilgpt2_p core/mlruntime/distilgpt2/distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-v1.2 --memory 1024 --timeout 300000







        # wsk action update alexnet core/mlruntime/alexnet/alexnet.py --docker=pandeymanish93/ml-inference:alexnet-v1.0 --memory 1024
        # wsk action update alexnet_video core/mlruntime/alexnet/alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.0 --memory 1024

        # wsk action update alexnet_p core/mlruntime/alexnet/alexnet.py --docker=pandeymanish93/ml-inference:alexnet-v1.2 --memory 1024
        # wsk action update alexnet_video_p core/mlruntime/alexnet/alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.2 --memory 1024

        # wsk action update googlenet core/mlruntime/googlenet/googlenet.py --docker=pandeymanish93/ml-inference:googlenet-v1.0 --memory 1024
        # wsk action update googlenet_video core/mlruntime/googlenet/googlenet_video.py --docker=pandeymanish93/ml-inference:googlenet-v1.0 --memory 1024

        # wsk action update googlenet_p core/mlruntime/googlenet/googlenet.py --docker=pandeymanish93/ml-inference:googlenet-v1.2 --memory 1024
        # # wsk action update googlenet_video_p core/mlruntime/googlenet/googlenet_video.py --docker=pandeymanish93/ml-inference:googlenet-v1.2 --memory 1024

        # wsk action update bert core/mlruntime/bert/bert.py --docker=pandeymanish93/ml-inference:bert-v1.0 --memory 1024
        # wsk action update distilgpt2 core/mlruntime/distilgpt2/distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-v1.0 --memory 1024

        ;;
    invoke_all)
        echo "Running vide and video-update for all version"
        wsk action update alexnet core/mlruntime/alexnet/alexnet.py --docker=pandeymanish93/ml-inference:alexnet-v1.0 --memory 1024 --timeout 300000
        wsk action update alexnet_video core/mlruntime/alexnet/alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.0 --memory 1024 --timeout 300000

        wsk action update alexnet_p core/mlruntime/alexnet/alexnet.py --docker=pandeymanish93/ml-inference:alexnet-v1.2 --memory 1024
        wsk action update alexnet_video_p core/mlruntime/alexnet/alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.2 --memory 1024

        ;;
esac
