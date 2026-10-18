# -*- coding: utf-8 -*-
"""兼容入口（thin shim）。

真正的实现位于 `src/lstm_dispatch_baseline.py`。保留这个文件是为了让
`experiments/` 下的实验脚本继续以 `import lstm_dispatch_baseline as B`
的方式工作（它们用 `sys.path.insert(0, '.')` 以项目根目录为工作目录）。

用法：
    python lstm_dispatch_baseline.py            # 等价于 python src/lstm_dispatch_baseline.py
"""
import importlib.util as _u
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "src" / "lstm_dispatch_baseline.py"
_spec = _u.spec_from_file_location("_bike_lstm_impl", _IMPL)
_mod = _u.module_from_spec(_spec)
sys.modules["_bike_lstm_impl"] = _mod
_spec.loader.exec_module(_mod)

# 把实现模块的公有符号暴露到本模块，供 `B.xxx` 访问
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})

if __name__ == "__main__":
    _mod.main()
