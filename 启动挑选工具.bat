@echo off
chcp 65001 >nul
title 连拍照片挑选工具
echo ============================================
echo   连拍照片挑选工具
echo ============================================
echo.
echo 正在启动服务器...
echo 请在浏览器中打开显示的地址
echo 按 Ctrl+C 可停止服务器
echo.
"C:\Users\13417\AppData\Local\Python\bin\python.exe" "D:\TinyGiant\picker.py" --port 8888
echo.
pause