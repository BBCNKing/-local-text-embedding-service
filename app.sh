#!/bin/bash

# 检查是否传入了模型路径和端口号
if [ $# -eq 2 ]; then
    model_path=$1
    port=$2
    nohup python your_app.py --model_path "$model_path" --port "$port" > app.log 2>&1 &
    echo "应用已在后台启动，日志输出到 app.log"
elif [ $# -eq 1 ]; then
    model_path=$1
    nohup python your_app.py --model_path "$model_path" > app.log 2>&1 &
    echo "应用已在后台启动，日志输出到 app.log"
else
    echo "Usage: sh app.sh <model_path> [port]"
    exit 1
fi