#!/bin/bash

# Zotero存储文件夹的路径
ZOTERO_STORAGE_PATH="/Users/qaz1214/Downloads/ref/ref-to-extract"

# 所有PDF文件将被复制到这个文件夹
OUTPUT_FOLDER_PATH="/Users/qaz1214/Downloads/ref//ref-to-extract/misc"

# 创建输出文件夹，如果它不存在的话
mkdir -p "$OUTPUT_FOLDER_PATH"

# 查找并复制所有的PDF文件
find "$ZOTERO_STORAGE_PATH" -name '*.pdf' -exec cp {} "$OUTPUT_FOLDER_PATH" \;

echo "所有的PDF文件已经被复制到 $OUTPUT_FOLDER_PATH"
