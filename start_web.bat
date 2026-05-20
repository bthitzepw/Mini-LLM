@echo off
echo ========================================
echo   Mini LLM 网页服务启动脚本
echo ========================================
echo.
echo 正在启动网页服务器...
echo 请在浏览器中访问: http://localhost:5000
echo.
echo 按 Ctrl+C 可停止服务
echo ========================================
echo.

python web_app.py

pause
