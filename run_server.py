"""启动服务器（禁用 webbrowser 以避免无界面环境卡死）"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 禁用 webbrowser
import webbrowser
webbrowser.open = lambda url: None

# 覆盖主函数的 webbrowser
import picker
original_open = webbrowser.open

port = sys.argv[2] if len(sys.argv) > 2 else '0'
directory = sys.argv[1] if len(sys.argv) > 1 else ''

sys.argv = ['picker.py']
if directory:
    sys.argv += [directory]
sys.argv += ['--port', port]

picker.main()
