# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import argparse
import os
from typing import List, Dict, Tuple, Set
import networkx as nx
from networkx.algorithms import isomorphism as iso
from rdkit import Chem
import sys
from tqdm import tqdm
import copy

# --- 常量定义 ---
organic_subset = {l: (i) for i, l in enumerate('N Cl * Br I P H O C S F'.split())}
bond_types = {l: (i) for i, l in enumerate([Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])}
NUM_LABELS = len(organic_subset)

# --- 路径配置 (硬编码) ---
DATA_PATH = "/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/data.smarts"
QUERY_PATH = "/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/query.smarts"
OUTPUT_DIR = "/home/hujiayue/antonio-decaro-SIGMo-304d11c/data/Test"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Graph Conversion Utils ---
def molToGraph(mol):
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), is_aromatic=atom.GetIsAromatic(), atom_symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
    return graph

def smartsToGraph(smarts: str) -> nx.DiGraph:
    mol_smart = Chem.MolFromSmarts(smarts)
    if mol_smart is None: return None
    return molToGraph(mol_smart)

def getNodeLabel(g: dict, digit: bool = True):
    if digit: return organic_subset.get(g['atom_symbol'], 0)
    return g['atom_symbol']

def getEdgeLabel(e: dict):
    if e['bond_type'] not in bond_types: bond_types[e['bond_type']] = len(bond_types)
    return bond_types[e['bond_type']]

def get_graph_list(lines, desc_text="Processing graphs"):
    graphs = []
    for el in tqdm(lines, desc=f"[*] {desc_text}", file=sys.stderr):
        el = el.strip()
        if not el: continue
        tmp = smartsToGraph(el)
        if tmp: graphs.append(tmp.to_directed())
    return graphs

# --- GraphPi Algorithm 1: 2-Cycle Based Elimination ---

def check_dag_acyclic(restrictions: List[Tuple[int, int]], num_nodes: int) -> bool:
    """
    GraphPi Paper Algorithm 1, Line 24-29: no_conflict check.
    检查约束集是否构成 DAG (无环)。如果是 DAG，返回 True (无冲突/未被消除)。
    限制 (u, v) 意味着 id(u) > id(v)。
    """
    # 构建有向图，边 u->v 代表 u > v
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(restrictions)
    return nx.is_directed_acyclic_graph(g)

def is_permutation_compatible(perm: Dict[int, int], restrictions: List[Tuple[int, int]]) -> bool:
    """
    判断一个置换 perm 是否与当前的约束集兼容。
    如果兼容（True），说明该置换还没有被约束集“消除”。
    GraphPi 论文逻辑：如果加入 permutation 映射后的约束导致成环，则该 permutation 被消除。
    这里我们简化逻辑：我们检查置换后的映射是否违反了任何现有的偏序关系。
    实际上，论文中的 no_conflict 是用来筛选剩余的 permutation group。
    
    GraphPi Algorithm 1 logic:
    For a restriction u > v (edge u->v), we also add edge perm[u] -> perm[v].
    If the graph has cycles, it's a conflict (eliminated).
    """
    g = nx.DiGraph()
    # 添加原始约束
    g.add_edges_from(restrictions)
    # 添加置换后的约束映射: 若 u > v 是必须的，且当前是置换 perm，
    # 意味着在另一个同构嵌入中，perm[u] 和 perm[v] 也就是对应位置。
    # 论文 Line 27-28: g.add_dir_edge(perm[res.first], perm[res.second])
    mapped_restrictions = [(perm[u], perm[v]) for u, v in restrictions]
    g.add_edges_from(mapped_restrictions)
    
    return nx.is_directed_acyclic_graph(g)

def find_2_cycle(perm: Dict[int, int]) -> Tuple[int, int]:
    """寻找置换中的 2-cycle (互换): u->v, v->u, u!=v"""
    for u, v in perm.items():
        if u != v and perm.get(v) == u:
            # 为了一致性，返回 ID 较大的在前，意为 u > v 约束
            return (max(u, v), min(u, v))
    return None

def generate_restrictions_graphpi(g: nx.Graph) -> List[Tuple[int, int]]:
    """
    实现 GraphPi 论文中的 Algorithm 1。
    返回一个列表 [(u, v), ...] 表示约束 u > v。
    """
    # 1. 获取所有自同构 (Permutation Group)
    nm = iso.categorical_node_match('atom_symbol', '*')
    GM = iso.GraphMatcher(g, g, node_match=nm)
    # 列表形式存储所有置换字典 {node_idx: mapped_node_idx}
    pg = list(GM.isomorphisms_iter())
    
    current_restrictions = []
    
    # 递归或迭代消除，直到只剩 Identity
    # 论文中使用递归 (generate 函数)，这里使用迭代简化
    
    # 初始 Group 包含所有自同构
    current_pg = pg
    
    # 只要 Group 大于 1 (除了恒等置换还有其他的)
    while len(current_pg) > 1:
        found_new_restriction = False
        
        # 遍历当前剩余的置换
        for perm in current_pg:
            # 寻找 2-cycle
            cycle = find_2_cycle(perm)
            
            if cycle:
                u, v = cycle # u > v
                # 检查这个约束是否已经存在或被隐含
                if (u, v) not in current_restrictions:
                    # 尝试添加这个约束
                    test_restrictions = current_restrictions + [(u, v)]
                    
                    # 验证添加后是否仍然合法 (针对 Identity 必须合法)
                    if check_dag_acyclic(test_restrictions, g.number_of_nodes()):
                        current_restrictions.append((u, v))
                        found_new_restriction = True
                        break # 找到一个有效 2-cycle 约束后，重新筛选 Group
        
        if not found_new_restriction:
            # 如果找不到 2-cycle，但 PG > 1，说明存在高阶 cycle (如 3-cycle A->B->C->A)
            # GraphPi 论文提到任何 k-cycle 可分解为 2-cycles [cite: 232]。
            # 但在实际实现中，如果找不到显式 2-cycle，可以任选一个非恒等映射 u!=p[u]，强制 u > p[u]
            # 来以此打破对称性。
            for perm in current_pg:
                is_identity = all(k==v for k,v in perm.items())
                if not is_identity:
                    # 找第一个变动的节点
                    for u, v in perm.items():
                        if u != v:
                            # 强制约束 max(u,v) > min(u,v)
                            r = (max(u,v), min(u,v))
                            if r not in current_restrictions:
                                test_restrictions = current_restrictions + [r]
                                if check_dag_acyclic(test_restrictions, g.number_of_nodes()):
                                    current_restrictions.append(r)
                                    found_new_restriction = True
                                    break
                    if found_new_restriction: break
            
            if not found_new_restriction:
                # 无法进一步消除 (极其罕见)，跳出
                break

        # 筛选：保留那些与新约束集“兼容”的置换
        # 在 GraphPi 中，目的是消除那些冲突的。
        # 兼容意味着：如果我们将置换应用到图上，它必须不违反约束。
        # 对于我们要消除的自同构 p，如果 id(u) > id(v) 是约束，
        # 而 p 导致了矛盾（即 p 对应的对称位置无法满足该约束），它就被消除了。
        # 剩下的 current_pg 是那些 *尚未* 被当前约束集打破对称性的置换。
        next_pg = []
        for perm in current_pg:
            # 如果 perm 与 restrictions 兼容 (即应用 perm 后不成环)，则它还没被消除
            if is_permutation_compatible(perm, current_restrictions):
                next_pg.append(perm)
        
        current_pg = next_pg
        
        # 此时 current_pg 应该变小了。如果只剩 1 (Identity)，循环结束。

    return current_restrictions

def compute_symmetry_data_graphpi(graphs: List[nx.DiGraph]):
    """
    使用 GraphPi 2-cycle 方法计算约束。
    """
    all_counts = []
    all_constraints = [] # 对应 C++ int* 数组

    for g in tqdm(graphs, desc="[*] GraphPi Symmetry", file=sys.stderr):
        g_undir = g.to_undirected()
        num_nodes = g.number_of_nodes()
        
        # 1. 计算自同构数量 (为了统计 match 数倍数)
        nm = iso.categorical_node_match('atom_symbol', '*')
        GM = iso.GraphMatcher(g_undir, g_undir, node_match=nm)
        count = len(list(GM.isomorphisms_iter()))
        all_counts.append(count)
        
        # 2. 使用 GraphPi 算法生成约束对列表 [(u, v), ...] 表示 u > v
        restriction_pairs = generate_restrictions_graphpi(g_undir)
        
        # 3. 将 pairs 转换为 C++ 可用的 int 数组 format (constraints[u] = v)
        # C++ 逻辑：if (candidate <= mapping[constraints[u]]) continue;
        # 这意味着我们需要存储 u > v 关系中的 v 到 constraints[u]。
        # 由于数组每个节点只能存一个约束，如果 GraphPi 生成了 (A>B) 和 (A>C)，
        # 我们只能保留一个。优先保留 GraphPi 生成靠前的，因为它们通常是基于 2-cycle 的强约束。
        g_constraints = [-1] * num_nodes
        for u, v in restriction_pairs:
            # u > v
            if g_constraints[u] == -1:
                g_constraints[u] = v
            else:
                # 如果该节点已有约束，GraphPi 可能生成了冗余约束或多重约束。
                # 由于 C++ 端数据结构限制，我们忽略后续约束。
                pass
        
        all_constraints.extend(g_constraints)
        
    return all_counts, all_constraints

# --- Parser ---
class SIGMOParser:
    def __init__(self, graphs): self.graphs = graphs
    def parse(self, file):
        for g in self.graphs:
            g = g.to_undirected()
            print(f'n#{g.number_of_nodes()} l#{NUM_LABELS}',  end=' ', file=file)
            for i in enumerate(g.nodes): print(i[0], getNodeLabel(g.nodes[i[0]]), end=' ', file=file)
            print(f'e#{g.number_of_edges()}', end=' ', file=file)
            for a, b, d in g.edges(data=True): print(a, b, getEdgeLabel(d), end=' ', file=file)
            print(file=file)

if __name__ == '__main__':
    # 读取数据
    with open(QUERY_PATH, 'r') as f: query_graphs = get_graph_list(f.readlines(), "Parsing Query")
    #with open(DATA_PATH, 'r') as f: data_graphs = get_graph_list(f.readlines(), "Parsing Data")

    # 计算对称性 (GraphPi Method)
    print("[*] Calculating Automorphisms (GraphPi 2-Cycle Method)...")
    automorphism_counts, symmetry_constraints = compute_symmetry_data_graphpi(query_graphs)

    # 保存文件
    print(f"[*] Saving to {OUTPUT_DIR}...")
    with open(os.path.join(OUTPUT_DIR, "query_graph.dat"), 'w') as f: SIGMOParser(query_graphs).parse(f)
    #with open(os.path.join(OUTPUT_DIR, "data_graph.dat"), 'w') as f: SIGMOParser(data_graphs).parse(f)
    
    # 保存计数
    with open(os.path.join(OUTPUT_DIR, "automorphism_counts.txt"), 'w') as f:
        for c in automorphism_counts: f.write(f"{c}\n")
        
    # 保存约束
    with open(os.path.join(OUTPUT_DIR, "symmetry_constraints.txt"), 'w') as f:
        for c in symmetry_constraints: f.write(f"{c}\n")
    
    print("[*] Done.")