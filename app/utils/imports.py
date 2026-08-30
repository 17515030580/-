# -*- coding: utf-8 -*-
import importlib


def import_symbol(path: str):
    if ":" not in path:
        raise ValueError("类路径必须采用 module.path:ClassName 格式")
    module_name, symbol_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)
