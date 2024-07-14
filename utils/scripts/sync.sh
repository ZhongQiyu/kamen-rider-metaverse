#!/bin/bash

# 检查参数个数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 PARENT_REPO_PATH EXTERNAL_REPOS_DIR"
    exit 1
fi

# 父仓库路径
PARENT_REPO_PATH=$1
# 包含外部仓库的目录
EXTERNAL_REPOS_DIR=$2

# 进入父仓库目录
cd "$PARENT_REPO_PATH" || exit

# 遍历要转换为子模块的目录中的所有项目
for repo_dir in "$EXTERNAL_REPOS_DIR"/*; do
    # 检查是否是一个 Git 仓库
    if [ -d "$repo_dir/.git" ]; then
        # 获取仓库的远程URL
        repo_url=$(git -C "$repo_dir" config --get remote.origin.url)
        if [ -z "$repo_url" ]; then
            echo "ERROR: Unable to find remote url for $repo_dir"
            continue
        fi

        # 获取相对路径
        repo_path="${repo_dir#$PARENT_REPO_PATH/}"

        # 删除旧的目录
        git rm -rf "$repo_path" && rm -rf "$repo_path"
        
        # 添加新的子模块或子树
        # 如果你想使用子树，请取消注释以下两行，并注释掉子模块相关的行
        # git subtree add --prefix="$repo_path" "$repo_url" master --squash
        # git commit -m "Added $repo_url as a new subtree at $repo_path."

        # 如果你想使用子模块，请确保以下两行没有被注释
        # git submodule add "$repo_url" "$repo_path"
    else
        echo "WARNING: $repo_dir is not a git repository."
    fi
done

# 这个commit是针对添加子模块的
# 如果你使用的是子树，请确保已经在上面的git subtree add命令中进行了commit
# git commit -m "Transformed external repos to submodules or subtrees."

# 更新所有子模块
# 如果你使用的是子模块，请取消注释以下行
# echo "Updating submodules..."
# git submodule update --init --recursive

# 如果你使用的是子树，并希望拉取更新，请编写一个额外的脚本来处理这个步骤
