#!/bin/sh
# LLMSEC 容器入口脚本——确保 .env 在应用启动前存在
#
# 优先级：
#   1. /app/.env 已存在（容器层或挂载）→ 直接用
#   2. /app/output/.env.bak 存在（上次运行持久化到 output 卷）→ 恢复
#   3. /app/.env.example 存在（镜像自带模板）→ 从模板创建
#   4. 都没有 → 创建空文件
#
# 这使得用户只需 `docker compose up` 或 `docker run`，无需预先 cp .env.example .env

set -e

if [ ! -f /app/.env ]; then
  if [ -f /app/output/.env.bak ]; then
    echo "[entrypoint] 从 output 卷恢复 .env（上次保存的配置）"
    cp /app/output/.env.bak /app/.env
  elif [ -f /app/.env.example ]; then
    echo "[entrypoint] 从 .env.example 模板创建 .env（请在看板 UI 中配置）"
    cp /app/.env.example /app/.env
  else
    echo "[entrypoint] 创建空 .env（请在看板 UI 中配置）"
    touch /app/.env
  fi
fi

exec "$@"
