"""
深层IR — 代码结构与数据流分析层

在中层IR（semantic.py）之上，增加编译器级的分析结构：
  - 控制流图 (Control Flow Graph, CFG)
  - 数据流图 (Data Flow Graph, DFG)
  - 模块依赖图 (Module Dependency Graph)

这些结构适合作为模型的辅助输入，帮助小型引擎理解代码的逻辑结构。

设计原则：
  1. 纯数据结构，零框架依赖
  2. 所有图结构都可序列化（to_dict / from_dict）
  3. 与 ir/layers.py 和 ir/semantic.py 互补，不冲突

# NOTE: CFG/DFG 在小模型代码理解中非常关键——
# 模型通过 token 序列无法直接感知"if 有两个分支"或"变量在第3行定义、第8行使用"这类信息。
# 这些图结构可以作为额外的特征输入，帮助模型做出更准确的预测。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto


# ============================================================
# 控制流图 (CFG)
# ============================================================

class BlockKind(Enum):
    """基本块类型"""
    ENTRY = auto()        # 函数入口
    EXIT = auto()         # 函数出口
    STRAIGHT = auto()     # 直线代码块
    CONDITION = auto()    # 条件判断
    LOOP_HEADER = auto()  # 循环头
    LOOP_BODY = auto()    # 循环体
    LOOP_EXIT = auto()    # 循环出口
    TRY_BODY = auto()     # 异常处理体
    CATCH_BODY = auto()   # 异常捕获体
    RETURN = auto()       # 返回点


@dataclass
class BasicBlock:
    """
    基本块：一段直线代码（无分支的连续指令序列）

    基本块是控制流图的最小单元。
    一个基本块有且只有一个入口和出口。

    示例:
      block = BasicBlock(id=0, kind=BlockKind.ENTRY, label="function_entry")
      block.add_statement("x = 1")
      block.add_statement("y = 2")
    """
    id: int
    kind: BlockKind = BlockKind.STRAIGHT
    label: str = ""
    statements: List[str] = field(default_factory=list)  # 块内的代码行
    start_line: int = 0
    end_line: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_statement(self, stmt: str):
        self.statements.append(stmt)

    @property
    def size(self) -> int:
        return len(self.statements)

    def __repr__(self):
        return f"BasicBlock(id={self.id}, kind={self.kind.name}, lines={self.start_line}-{self.end_line}, stmts={self.size})"


@dataclass
class CFGEdge:
    """
    控制流边：从 source_block → target_block

    kind 分类：
      - "fallthrough": 顺序执行
      - "branch_true": 条件为真
      - "branch_false": 条件为假
      - "loop_back": 循环回边
      - "except": 异常跳转
    """
    source: int
    target: int
    kind: str = "fallthrough"
    condition: str = ""  # 条件表达式文本（可选）

    def __repr__(self):
        return f"CFGEdge({self.source} → {self.target}, {self.kind})"


class ControlFlowGraph:
    """
    控制流图

    表示函数体内的执行路径（所有可能的执行流向）。

    用法:
      cfg = ControlFlowGraph()
      entry = cfg.add_block(BlockKind.ENTRY, "start")
      body = cfg.add_block(BlockKind.STRAIGHT, "body")
      exit_block = cfg.add_block(BlockKind.EXIT, "end")
      cfg.add_edge(entry.id, body.id, kind="fallthrough")
      cfg.add_edge(body.id, exit_block.id, kind="fallthrough")
      cfg.validate()  # True
    """

    def __init__(self, function_name: str = ""):
        self.function_name = function_name
        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: List[CFGEdge] = []
        self._next_id: int = 0

    def add_block(self, kind: BlockKind = BlockKind.STRAIGHT,
                  label: str = "", start_line: int = 0,
                  end_line: int = 0) -> BasicBlock:
        """添加基本块"""
        block = BasicBlock(
            id=self._next_id, kind=kind, label=label,
            start_line=start_line, end_line=end_line
        )
        self.blocks[self._next_id] = block
        self._next_id += 1
        return block

    def add_edge(self, source: int, target: int, kind: str = "fallthrough",
                 condition: str = ""):
        """添加控制流边"""
        edge = CFGEdge(source, target, kind, condition)
        self.edges.append(edge)

    def get_successors(self, block_id: int) -> List[int]:
        """获取某块的所有后继"""
        return [e.target for e in self.edges if e.source == block_id]

    def get_predecessors(self, block_id: int) -> List[int]:
        """获取某块的所有前驱"""
        return [e.source for e in self.edges if e.target == block_id]

    def get_entry(self) -> Optional[BasicBlock]:
        """获取入口块"""
        for block in self.blocks.values():
            if block.kind == BlockKind.ENTRY:
                return block
        return None

    def get_exit(self) -> Optional[BasicBlock]:
        """获取出口块"""
        for block in self.blocks.values():
            if block.kind == BlockKind.EXIT:
                return block
        return None

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def cyclomatic_complexity(self) -> int:
        """
        圈复杂度 (McCabe)

        M = E - N + 2P
        E: 边数, N: 节点数, P: 连通分量数(单函数=1)
        """
        return self.edge_count - self.block_count + 2

    def validate(self) -> Tuple[bool, str]:
        """
        验证CFG的完整性

        检查：
          1. 是否有入口块
          2. 是否有出口块
          3. 所有边引用的块都存在
          4. 所有非出口块都有后继
        """
        if not self.get_entry():
            return False, "Missing entry block"
        if not self.get_exit():
            return False, "Missing exit block"

        for edge in self.edges:
            if edge.source not in self.blocks:
                return False, f"Edge source {edge.source} not in blocks"
            if edge.target not in self.blocks:
                return False, f"Edge target {edge.target} not in blocks"

        for block_id, block in self.blocks.items():
            if block.kind != BlockKind.EXIT:
                succs = self.get_successors(block_id)
                if not succs:
                    return False, f"Block {block_id} ({block.kind.name}) has no successors"

        return True, "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "blocks": [
                {
                    "id": b.id,
                    "kind": b.kind.name,
                    "label": b.label,
                    "start_line": b.start_line,
                    "end_line": b.end_line,
                    "size": b.size,
                }
                for b in self.blocks.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "kind": e.kind,
                    "condition": e.condition,
                }
                for e in self.edges
            ],
            "cyclomatic_complexity": self.cyclomatic_complexity,
        }

    def summary(self) -> str:
        valid, msg = self.validate()
        lines = [
            f"CFG for '{self.function_name}': "
            f"{self.block_count} blocks, {self.edge_count} edges, "
            f"CC={self.cyclomatic_complexity}, valid={valid}",
        ]
        if not valid:
            lines.append(f"  ! {msg}")
        return "\n".join(lines)


# ============================================================
# 数据流图 (DFG)
# ============================================================

@dataclass
class DefUseChain:
    """
    定义-使用链

    记录一个变量从定义点到所有使用点的路径。

    示例:
      DefUseChain(
        variable="x",
        def_line=2,           # x = 1 (定义)
        use_lines=[5, 8, 12]  # print(x) / y = x + 2 / return x
      )
    """
    variable: str
    def_line: int
    use_lines: List[int] = field(default_factory=list)
    def_expression: str = ""   # 定义的表达式文本

    @property
    def use_count(self) -> int:
        return len(self.use_lines)

    def __repr__(self):
        return f"DefUseChain({self.variable}: def@{self.def_line}, uses={self.use_lines})"


class DataFlowGraph:
    """
    数据流图

    表示变量在函数体内的定义-使用关系。

    对小型引擎尤其有价值：当模型看到变量名时，
    可以关联到它的定义和使用位置，提升代码生成的准确性。

    用法:
      dfg = DataFlowGraph()
      dfg.add_def("x", line=2, expression="x = 1")
      dfg.add_use("x", line=5)
      dfg.add_use("x", line=8)
      dfg.get_live_range("x")  # (2, 8)
    """

    def __init__(self, function_name: str = ""):
        self.function_name = function_name
        self._chains: Dict[str, DefUseChain] = {}

    def add_def(self, variable: str, line: int, expression: str = ""):
        """记录变量定义"""
        if variable not in self._chains:
            self._chains[variable] = DefUseChain(
                variable=variable, def_line=line, def_expression=expression
            )
        else:
            # 重新定义（覆盖之前的定义）
            self._chains[variable].def_line = line
            self._chains[variable].def_expression = expression
            self._chains[variable].use_lines.clear()

    def add_use(self, variable: str, line: int):
        """记录变量使用"""
        if variable in self._chains:
            if line not in self._chains[variable].use_lines:
                self._chains[variable].use_lines.append(line)

    def get_chain(self, variable: str) -> Optional[DefUseChain]:
        """获取变量的定义-使用链"""
        return self._chains.get(variable)

    def get_live_range(self, variable: str) -> Optional[Tuple[int, int]]:
        """
        获取变量的活跃范围

        返回 (定义行号, 最后使用行号)
        若变量只有定义而无使用（死代码），返回 None
        """
        chain = self._chains.get(variable)
        if chain is None:
            return None
        if not chain.use_lines:
            return None
        return (chain.def_line, max(chain.use_lines))

    def get_unused_variables(self) -> List[str]:
        """找出定义但从未使用的变量（潜在死代码）"""
        unused = []
        for var, chain in self._chains.items():
            if chain.use_count == 0:
                unused.append(var)
        return unused

    @property
    def variable_count(self) -> int:
        return len(self._chains)

    @property
    def total_uses(self) -> int:
        return sum(ch.use_count for ch in self._chains.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "variables": [
                {
                    "name": ch.variable,
                    "def_line": ch.def_line,
                    "def_expression": ch.def_expression,
                    "use_count": ch.use_count,
                    "use_lines": ch.use_lines,
                }
                for ch in self._chains.values()
            ],
            "unused": self.get_unused_variables(),
        }

    def summary(self) -> str:
        unused = self.get_unused_variables()
        lines = [
            f"DFG for '{self.function_name}': "
            f"{self.variable_count} variables, {self.total_uses} uses",
        ]
        if unused:
            lines.append(f"  Unused variables: {', '.join(unused)}")
        return "\n".join(lines)


# ============================================================
# 模块依赖图 (Module Dependency Graph)
# ============================================================

@dataclass
class ModuleImport:
    """
    模块导入记录

    示例:
      ModuleImport(source="mymodule.utils", target="os", kind="stdlib")
      ModuleImport(source="mymodule.utils", target="numpy", kind="third_party")
      ModuleImport(source="mymodule.main", target="mymodule.utils", kind="local")
    """
    source: str     # 导入者
    target: str     # 被导入者
    kind: str = "local"  # stdlib / third_party / local
    imported_names: List[str] = field(default_factory=list)  # from X import a, b, c
    line: int = 0


class ModuleGraph:
    """
    模块依赖图

    表示项目中所有模块之间的导入关系。
    对仓库级代码理解和依赖分析很关键。

    用法:
      mg = ModuleGraph()
      mg.add_import("main.py", "utils.py", kind="local")
      mg.add_import("main.py", "os", kind="stdlib")
      mg.add_import("utils.py", "math", kind="stdlib")
      mg.find_circular_deps()  # []
      mg.get_topological_order()  # ['os', 'math', 'utils.py', 'main.py']
    """

    def __init__(self, project_name: str = ""):
        self.project_name = project_name
        self.imports: List[ModuleImport] = []
        self._modules: Set[str] = set()
        self._deps: Dict[str, Set[str]] = {}   # source → {targets}

    def add_module(self, name: str):
        """注册一个模块"""
        self._modules.add(name)

    def add_import(self, source: str, target: str, kind: str = "local",
                   imported_names: Optional[List[str]] = None, line: int = 0):
        """添加一个导入关系"""
        imp = ModuleImport(source, target, kind,
                          imported_names or [], line)
        self.imports.append(imp)
        self._modules.add(source)
        self._modules.add(target)
        self._deps.setdefault(source, set()).add(target)

    def get_dependencies(self, module: str) -> List[str]:
        """获取某模块的所有依赖"""
        return sorted(self._deps.get(module, set()))

    def get_dependents(self, module: str) -> List[str]:
        """获取依赖某模块的所有模块（谁依赖它）"""
        return sorted(m for m, deps in self._deps.items() if module in deps)

    def find_circular_deps(self) -> List[List[str]]:
        """检测模块间的循环依赖"""
        cycles = []
        visited = set()
        in_stack = set()
        stack = []

        def dfs(node):
            visited.add(node)
            in_stack.add(node)
            stack.append(node)
            for dep in sorted(self._deps.get(node, set())):
                if dep not in visited:
                    dfs(dep)
                elif dep in in_stack:
                    idx = stack.index(dep)
                    cycles.append(stack[idx:] + [dep])
            stack.pop()
            in_stack.discard(node)

        for mod in sorted(self._modules):
            if mod not in visited:
                dfs(mod)
        return cycles

    def get_topological_order(self) -> List[str]:
        """
        拓扑排序：确保依赖的模块排在前面

        用于确定构建/加载顺序。
        如果存在循环依赖，无法完全排序，但仍会返回一个近似顺序。
        """
        indeg = {m: 0 for m in self._modules}
        for source, targets in self._deps.items():
            for t in targets:
                indeg[t] = indeg.get(t, 0) + 1

        queue = [m for m in self._modules if indeg.get(m, 0) == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dep in self._deps.get(node, set()):
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    queue.append(dep)

        # 剩余的是循环依赖中的模块
        remaining = [m for m in self._modules if m not in result]
        result.extend(remaining)
        return result

    @property
    def module_count(self) -> int:
        return len(self._modules)

    @property
    def import_count(self) -> int:
        return len(self.imports)

    def get_stats(self) -> Dict[str, Any]:
        """统计报告"""
        stdlib = sum(1 for imp in self.imports if imp.kind == "stdlib")
        third_party = sum(1 for imp in self.imports if imp.kind == "third_party")
        local = sum(1 for imp in self.imports if imp.kind == "local")
        return {
            "total_modules": self.module_count,
            "total_imports": self.import_count,
            "stdlib_imports": stdlib,
            "third_party_imports": third_party,
            "local_imports": local,
            "circular_deps": len(self.find_circular_deps()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            **self.get_stats(),
            "dependencies": {
                mod: self.get_dependencies(mod) for mod in sorted(self._modules)
            },
            "dependents": {
                mod: self.get_dependents(mod) for mod in sorted(self._modules)
            },
            "topological_order": self.get_topological_order(),
        }

    def summary(self) -> str:
        stats = self.get_stats()
        cycles = self.find_circular_deps()
        lines = [
            f"ModuleGraph for '{self.project_name}': "
            f"{stats['total_modules']} modules, {stats['total_imports']} imports",
            f"  stdlib={stats['stdlib_imports']}, third_party={stats['third_party_imports']}, local={stats['local_imports']}",
        ]
        if cycles:
            lines.append(f"  ! Circular dependencies: {len(cycles)}")
            for cycle in cycles[:3]:
                lines.append(f"    {' → '.join(cycle)}")
        return "\n".join(lines)


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    # CFG
    "BlockKind", "BasicBlock", "CFGEdge", "ControlFlowGraph",
    # DFG
    "DefUseChain", "DataFlowGraph",
    # ModuleGraph
    "ModuleImport", "ModuleGraph",
]
