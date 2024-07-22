#!/bin/bash

# 检查是否为root用户执行，因为注销其他用户通常需要管理员权限
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# 从 `/etc/passwd` 获取所有登录用户的列表
# 假设每个用户都有一个对应的会话管理进程
users=$(cut -d: -f1 /etc/passwd)

for user in $users
do
    # 这里使用 `pkill` 命令来杀死所有属于该用户的进程
    # 注意：这是非常激进的方式，因为它会关闭用户所有正在运行的进程
    # 实际使用时需要根据实际情况进行调整
    pkill -KILL -u $user
    
    # 如果你有特定的服务或进程管理用户会话，你需要调用特定的命令来注销用户
    # 例如：service session-manager logout $user
done

echo "All users have been logged off."

