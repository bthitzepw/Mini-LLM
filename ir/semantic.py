"""
中层IR — 语义结构表示层

在浅层IR（词法/语法结构）之上，增加语义信息：
  - 作用域树 (Scope Tree)
  - 符号表 (Symbol Table)：变量 → 类型绑定
  - 调用图 (Call Graph)：函数调用有向图
  - 类型标注 (Type Annotation)：参数/返回类型信息

设计原则：
  1. 与 layers.py 的神经网络层定义互补，不冲突
  2. 纯数据结构，不依赖任何框架
  3. 可序列化（to_dict / from_dict），方便作为模型输入特征

# NOTE: 这一层的目的是让模型在训练时能够感知代码的语义结构，
# 而非仅仅看到"token 序列"。对于 38M 小模型，语义信息尤其重要。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto


# ============================================================
# 基础类型定义
# ============================================================

class PrimitiveType(Enum):
    """基本类型"""
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    VOID = auto()
    ANY = auto()        # 未知/动态类型
    NULL = auto()


class TypeCategory(Enum):
    """类型分类"""
    PRIMITIVE = auto()
    ARRAY = auto()
    MAP = auto()
    FUNCTION = auto()
    CLASS = auto()
    GENERIC = auto()
    UNION = auto()
    POINTER = auto()


@dataclass
class TypeInfo:
    """
    类型信息

    示例:
      TypeInfo(category=PRIMITIVE, name="int")
      TypeInfo(category=ARRAY, name="list", type_params=[TypeInfo(category=PRIMITIVE, name="int")])
      TypeInfo(category=FUNCTION, name="add", type_params=[
          TypeInfo(category=PRIMITIVE, name="int"),  # param
          TypeInfo(category=PRIMITIVE, name="int"),  # param
      ], return_type=TypeInfo(category=PRIMITIVE, name="int"))
    """
    category: TypeCategory
    name: str
    type_params: List[TypeInfo] = field(default_factory=list)
    return_type: Optional[TypeInfo] = None
    is_nullable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "category": self.category.name,
            "name": self.name,
            "is_nullable": self.is_nullable,
        }
        if self.type_params:
            d["type_params"] = [tp.to_dict() for tp in self.type_params]
        if self.return_type:
            d["return_type"] = self.return_type.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TypeInfo":
        return cls(
            category=TypeCategory[d["category"]],
            name=d.get("name", ""),
            type_params=[cls.from_dict(tp) for tp in d.get("type_params", [])],
            return_type=cls.from_dict(d["return_type"]) if "return_type" in d else None,
            is_nullable=d.get("is_nullable", False),
        )

    @classmethod
    def primitive(cls, name: str) -> "TypeInfo":
        """快捷创建基本类型"""
        return cls(category=TypeCategory.PRIMITIVE, name=name)

    @classmethod
    def unknown(cls) -> "TypeInfo":
        """未知类型"""
        return cls(category=TypeCategory.PRIMITIVE, name="any")


# ============================================================
# 作用域系统 (Scope)
# ============================================================

@dataclass
class Symbol:
    """
    符号表条目：代码中的一个命名实体

    包含：
      - 变量绑定 (name → type)
      - 定义位置 (定义行号)
      - 是否参数/局部/全局
      - 引用的行号列表
    """
    name: str
    type_info: TypeInfo = field(default_factory=TypeInfo.unknown)
    defined_at: int = 0                      # 定义行号
    references: List[int] = field(default_factory=list)  # 引用行号列表
    is_parameter: bool = False
    is_local: bool = True
    is_mutable: bool = True
    initial_value: Optional[str] = None     # 初始值表达式（可选）

    def __repr__(self):
        kind = "param" if self.is_parameter else ("local" if self.is_local else "global")
        return f"Symbol({self.name}: {self.type_info.name}, {kind}, line={self.defined_at})"


class Scope:
    """
    作用域节点

    一个 Scope 代表代码中的一个命名空间区域（函数体、类体、if块等）。
    作用域树通过 parent/children 形成嵌套结构。

    示例:
      global_scope = Scope(name="module", kind="module")
      func_scope = Scope(name="add", kind="function", parent=global_scope)
      func_scope.add_symbol(Symbol("x", TypeInfo.primitive("int"), defined_at=2))
      func_scope.add_symbol(Symbol("y", TypeInfo.primitive("int"), defined_at=2))
    """

    def __init__(self, name: str, kind: str = "block",
                 parent: Optional["Scope"] = None):
        self.name = name
        self.kind = kind            # module / function / class / block / loop / try
        self.parent = parent
        self.children: List[Scope] = []
        self.symbols: Dict[str, Symbol] = {}   # name → Symbol
        self.start_line: int = 0
        self.end_line: int = 0

        if parent:
            parent.children.append(self)

    def add_symbol(self, symbol: Symbol):
        """向当前作用域添加符号"""
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        """
        在当前作用域及所有祖先作用域中查找符号

        从内向外逐层查找，模拟词法作用域规则。
        """
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """仅当前作用域查找"""
        return self.symbols.get(name)

    def get_all_symbols(self, recursive: bool = True) -> List[Symbol]:
        """获取所有符号（可递归子作用域）"""
        symbols = list(self.symbols.values())
        if recursive:
            for child in self.children:
                symbols.extend(child.get_all_symbols(recursive=True))
        return symbols

    def get_depth(self) -> int:
        """从根到当前作用域的深度"""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbols": {
                name: {
                    "type": sym.type_info.to_dict(),
                    "defined_at": sym.defined_at,
                    "references": sym.references,
                    "is_parameter": sym.is_parameter,
                    "is_local": sym.is_local,
                    "is_mutable": sym.is_mutable,
                }
                for name, sym in self.symbols.items()
            },
            "children": [child.to_dict() for child in self.children],
        }

    def __repr__(self):
        symbol_count = len(self.symbols)
        child_count = len(self.children)
        return f"Scope(name='{self.name}', kind='{self.kind}', symbols={symbol_count}, children={child_count})"


# ============================================================
# 调用图 (Call Graph)
# ============================================================

@dataclass
class CallEdge:
    """
    调用边：caller → callee

    表示一个函数调用了另一个函数。
    """
    caller: str          # 调用者函数名
    callee: str          # 被调用者函数名
    line: int = 0        # 调用发生的行号
    arg_count: int = 0   # 参数数量
    is_direct: bool = True  # 直接调用 vs 间接调用（函数指针/回调）

    def __repr__(self):
        return f"{self.caller} → {self.callee} (line {self.line})"


class CallGraph:
    """
    函数调用有向图

    表示模块内所有函数的调用关系，对代码理解和依赖分析至关重要。

    用法:
      cg = CallGraph()
      cg.add_call("main", "add", line=10, arg_count=2)
      cg.add_call("main", "print", line=11, arg_count=1)
      cg.get_callees("main")  # → ["add", "print"]
      cg.get_callers("add")   # → ["main"]
    """

    def __init__(self):
        self.edges: List[CallEdge] = []
        # 邻接表（优化查询性能）
        self._callees: Dict[str, Set[str]] = {}    # caller → {callees}
        self._callers: Dict[str, Set[str]] = {}    # callee → {callers}
        self._functions: Set[str] = set()

    def add_function(self, name: str):
        """注册一个函数"""
        self._functions.add(name)

    def add_call(self, caller: str, callee: str, line: int = 0,
                 arg_count: int = 0, is_direct: bool = True):
        """添加一条调用边"""
        edge = CallEdge(caller, callee, line, arg_count, is_direct)
        self.edges.append(edge)

        self._callees.setdefault(caller, set()).add(callee)
        self._callers.setdefault(callee, set()).add(caller)
        self._functions.add(caller)
        self._functions.add(callee)

    def get_callees(self, func_name: str) -> List[str]:
        """获取某函数调用的所有函数"""
        return sorted(self._callees.get(func_name, set()))

    def get_callers(self, func_name: str) -> List[str]:
        """获取调用某函数的所有函数"""
        return sorted(self._callers.get(func_name, set()))

    def get_fan_out(self, func_name: str) -> int:
        """扇出：此函数调用了多少个不同函数"""
        return len(self._callees.get(func_name, set()))

    def get_fan_in(self, func_name: str) -> int:
        """扇入：有多少个函数调用了此函数"""
        return len(self._callers.get(func_name, set()))

    @property
    def function_count(self) -> int:
        return len(self._functions)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def find_cycles(self) -> List[List[str]]:
        """
        检测调用图中的循环调用（递归/间接递归）

        使用 DFS + 回溯检测环。
        返回所有环路（函数名列表）。
        """
        cycles = []
        visited: Set[str] = set()
        in_stack: Set[str] = set()
        stack: List[str] = []

        def dfs(node: str):
            visited.add(node)
            in_stack.add(node)
            stack.append(node)

            for callee in sorted(self._callees.get(node, set())):
                if callee not in visited:
                    dfs(callee)
                elif callee in in_stack:
                    # 找到环
                    cycle_start = stack.index(callee)
                    cycles.append(stack[cycle_start:] + [callee])

            stack.pop()
            in_stack.discard(node)

        for func in sorted(self._functions):
            if func not in visited:
                dfs(func)
        return cycles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "functions": sorted(self._functions),
            "edges": [
                {
                    "caller": e.caller,
                    "callee": e.callee,
                    "line": e.line,
                    "arg_count": e.arg_count,
                    "is_direct": e.is_direct,
                }
                for e in self.edges
            ],
            "fan_out": {f: self.get_fan_out(f) for f in sorted(self._functions)},
            "fan_in": {f: self.get_fan_in(f) for f in sorted(self._functions)},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CallGraph":
        cg = cls()
        for f in d.get("functions", []):
            cg.add_function(f)
        for e in d.get("edges", []):
            cg.add_call(**e)
        return cg

    def summary(self) -> str:
        """可读摘要"""
        cycles = self.find_cycles()
        lines = [
            f"CallGraph: {self.function_count} functions, {self.edge_count} edges",
            f"  Top fan-out:",
        ]
        fan_outs = sorted(self._functions, key=lambda f: -self.get_fan_out(f))[:5]
        for f in fan_outs:
            lines.append(f"    {f}: calls {self.get_fan_out(f)} functions")
        if cycles:
            lines.append(f"  Cycles detected: {len(cycles)}")
            for i, cycle in enumerate(cycles[:3]):
                lines.append(f"    Cycle {i+1}: {' → '.join(cycle)}")
        return "\n".join(lines)


# ============================================================
# 代码语义块 (SemanticBlock)
# ============================================================

@dataclass
class SemanticBlock:
    """
    代码片段的结构化语义表示

    将一段代码的所有语义信息打包，可作为模型训练/推理的输入特征。
    包含：
      - 原始代码文本（或 token 序列）
      - 作用域树
      - 符号表（扁平化）
      - 调用图（如果包含多个函数）
      - 类型标注
    """
    code_text: str = ""
    scope_root: Optional[Scope] = None
    call_graph: Optional[CallGraph] = None
    language: str = "python"
    token_count: int = 0

    def get_symbol_count(self) -> int:
        if self.scope_root:
            return len(self.scope_root.get_all_symbols())
        return 0

    def get_function_count(self) -> int:
        if self.call_graph:
            return self.call_graph.function_count
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "token_count": self.token_count,
            "symbol_count": self.get_symbol_count(),
            "function_count": self.get_function_count(),
            "scope": self.scope_root.to_dict() if self.scope_root else None,
            "call_graph": self.call_graph.to_dict() if self.call_graph else None,
        }

    def summary(self) -> str:
        """可读摘要"""
        lines = [
            f"SemanticBlock(lang={self.language}, tokens={self.token_count})",
            f"  Symbols: {self.get_symbol_count()}",
            f"  Functions: {self.get_function_count()}",
        ]
        if self.call_graph:
            lines.append("  " + self.call_graph.summary().replace("\n", "\n  "))
        return "\n".join(lines)


# ============================================================
# 语义特征提取器接口
# ============================================================

class SemanticExtractor:
    """
    语义特征提取器基类

    从代码文本中提取 SemanticBlock（语种相关，子类实现）。
    当前仅提供接口定义，具体语言的分析器在后续版本中实现。
    """

    def extract(self, code_text: str, language: str = "python") -> SemanticBlock:
        """从代码文本中提取语义块"""
        raise NotImplementedError


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    # 类型系统
    "PrimitiveType", "TypeCategory", "TypeInfo",
    # 作用域
    "Symbol", "Scope",
    # 调用图
    "CallEdge", "CallGraph",
    # 语义块
    "SemanticBlock",
    # 提取器
    "SemanticExtractor",
]
