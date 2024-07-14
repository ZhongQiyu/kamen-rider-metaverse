#!/bin/bash

# 创建性能测试目录和子目录
mkdir -p performance/memory_usage
mkdir -p performance/response_time

# 创建系统测试目录和子目录
mkdir -p system/end_to_end

# 创建性能测试相关的文件
touch performance/memory_usage/test_database_memory.py
touch performance/memory_usage/test_processing_memory.py
touch performance/response_time/test_api_response.py
touch performance/response_time/test_loading_response.py

# 创建系统测试相关的文件
touch system/end_to_end/test_user_flow1.py
touch system/end_to_end/test_user_flow2.py

echo "Test directories and files have been created successfully."
