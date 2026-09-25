cd mlfiles

#if folder python_runtime exists then copy below commands
if [ -d "../python_runtime" ]; then
    # Copy files from python_runtime/core/mlruntime/alexnet/alexnet.py to current directory
    cp ../python_runtime/core/mlruntime/alexnet/alexnet.py .
    cp ../python_runtime/core/mlruntime/resnet50/resnet50.py .
    cp ../python_runtime/core/mlruntime/efficientnet/efficientnet.py .
    cp ../python_runtime/core/mlruntime/googlenet/googlenet.py .
    cp ../python_runtime/core/mlruntime/inception/inception.py .
    cp ../python_runtime/core/mlruntime/bert/bert.py .
    cp ../python_runtime/core/mlruntime/distilgpt2/distilgpt2.py .
else
    echo "python_runtime directory does not exist. Exiting."
    # exit 1
fi


VERSION="v1.0"
wsk action update alexnet alexnet.py --docker=pandeymanish93/ml-inference:alexnet-$VERSION --memory 1024
wsk action update resnet50 resnet50.py --docker=pandeymanish93/ml-inference:resnet50-$VERSION --memory 1024
wsk action update efficientnet efficientnet.py --docker=pandeymanish93/ml-inference:efficientnet-$VERSION --memory 1524
wsk action update googlenet googlenet.py --docker=pandeymanish93/ml-inference:googlenet-$VERSION --memory 1024
wsk action update inception inception.py --docker=pandeymanish93/ml-inference:inception-$VERSION --memory 1024
wsk action update bert bert.py --docker=pandeymanish93/ml-inference:bert-$VERSION --memory 1024
wsk action update distilgpt2 distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-$VERSION --memory 1024

VERSION="v1.2"
wsk action update alexnet_p alexnet.py --docker=pandeymanish93/ml-inference:alexnet-$VERSION --memory 1024
wsk action update resnet50_p resnet50.py --docker=pandeymanish93/ml-inference:resnet50-$VERSION --memory 1024
wsk action update efficientnet_p efficientnet.py --docker=pandeymanish93/ml-inference:efficientnet-$VERSION --memory 1524
wsk action update googlenet_p googlenet.py --docker=pandeymanish93/ml-inference:googlenet-$VERSION --memory 1024
wsk action update inception_p inception.py --docker=pandeymanish93/ml-inference:inception-$VERSION --memory 1024
wsk action update bert_p bert.py --docker=pandeymanish93/ml-inference:bert-$VERSION --memory 1024
wsk action update distilgpt2_p distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-$VERSION --memory 1024




# wsk action update alexnet alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.0 --memory 1024 --timeout 300000
# wsk action update alexnet_p alexnet_video.py --docker=pandeymanish93/ml-inference:alexnet-v1.2 --memory 1024 --timeout 300000

# wsk action update efficientnet efficientnet_video.py --docker=pandeymanish93/ml-inference:efficientnet-v1.0 --memory 1024 --timeout 300000
# wsk action update efficientnet_p efficientnet_video.py --docker=pandeymanish93/ml-inference:efficientnet-v1.2 --memory 1024 --timeout 300000

# wsk action update googlenet googlenet_b64.py --docker=pandeymanish93/ml-inference:googlenet-v1.0 --memory 1024 --timeout 300000
# wsk action update googlenet_p core/mlruntime/googlenet/googlenet_b64.py --docker=pandeymanish93/ml-inference:googlenet-v1.2 --memory 1024 --timeout 300000

# wsk action update resnet50 core/mlruntime/resnet50/resnet50.py --docker=pandeymanish93/ml-inference:resnet50-v1.0 --memory 1024 --timeout 300000
# wsk action update resnet50_p core/mlruntime/resnet50/resnet50.py --docker=pandeymanish93/ml-inference:resnet50-v1.2 --memory 1024 --timeout 300000

# wsk action update inception core/mlruntime/inception/inception.py --docker=pandeymanish93/ml-inference:inception-v1.0 --memory 1024 --timeout 300000
# wsk action update inception_p core/mlruntime/inception/inception.py --docker=pandeymanish93/ml-inference:inception-v1.2 --memory 1024 --timeout 300000

# wsk action update bert core/mlruntime/bert/bert.py --docker=pandeymanish93/ml-inference:bert-v1.0 --memory 1024 --timeout 300000
# wsk action update bert_p core/mlruntime/bert/bert.py --docker=pandeymanish93/ml-inference:bert-v1.2 --memory 1024 --timeout 300000

# wsk action update distilgpt2 core/mlruntime/distilgpt2/distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-v1.0 --memory 1024 --timeout 300000
# wsk action update distilgpt2_p core/mlruntime/distilgpt2/distilgpt2.py --docker=pandeymanish93/ml-inference:distilgpt2-v1.2 --memory 1024 --timeout 300000
