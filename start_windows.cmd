@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
where py >nul 2>&1
if errorlevel 1 (
  python app.py
) else (
  py -3 app.py
)
if errorlevel 1 (
  echo.
  echo 启动失败，请检查 Python 安装和部署说明。
  pause
)
