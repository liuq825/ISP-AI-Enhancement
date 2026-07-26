"""把“所有 Python 代码具有中文说明”固化为持续集成门禁。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CHINESE = re.compile(r"[\u4e00-\u9fff]")
_PYTHON_ROOTS = (Path("src"), Path("tests"))


def _has_chinese_docstring(node: ast.AST) -> bool:
    """判断 AST 节点是否具有至少四个中文字符的说明字符串。"""

    docstring = ast.get_docstring(node)
    return bool(docstring and len(_CHINESE.findall(docstring)) >= 4)


def test_python_modules_classes_and_functions_have_chinese_docs() -> None:
    """所有模块、类和函数都必须提供可维护的中文说明。"""

    missing: list[str] = []
    for root in _PYTHON_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if not _has_chinese_docstring(tree):
                missing.append(f"{path}:module")
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not _has_chinese_docstring(node):
                        missing.append(f"{path}:{node.lineno}:{node.name}")
    assert not missing, "以下代码缺少详细中文说明：\n" + "\n".join(missing)
