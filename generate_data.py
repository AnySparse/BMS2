# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import networkx as nx
from rdkit import Chem
from tqdm import tqdm

# --- 定义常量：原子和键的标签映射 ---
organic_subset = {l: (i) for i, l in enumerate('N Cl * Br I P H O C S F'.split())}
bond_types = {l: (i) for i, l in enumerate([
    Chem.rdchem.BondType.UNSPECIFIED, 
    Chem.rdchem.BondType.SINGLE, 
    Chem.rdchem.BondType.DOUBLE, 
    Chem.rdchem.BondType.TRIPLE, 
    Chem.rdchem.BondType.AROMATIC
])}
NUM_LABELS = len(organic_subset)

# --- 核心转换函数 ---

def molToGraph(mol):
    """把一个 RDKit 分子对象转为 networkx.Graph"""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       is_aromatic=atom.GetIsAromatic(),
                       atom_symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())
    return graph

def smartsToGraph(smarts: str) -> nx.DiGraph:
    """SMARTS 字符串转图"""
    mol_smart = Chem.MolFromSmarts(smarts)
    if mol_smart is None:
        return None
    return molToGraph(mol_smart)

def getNodeLabel(g: dict, digit: bool = True):
    """获取节点（原子）的离散标签"""
    if digit:
        if g['atom_symbol'] in organic_subset:
            return organic_subset[g['atom_symbol']]
        else:
            return 0
    return g['atom_symbol']

def getEdgeLabel(e: dict):
    """获取边（化学键）的离散标签"""
    if e['bond_type'] not in bond_types:
        bond_types[e['bond_type']] = len(bond_types)
    return bond_types[e['bond_type']]

def get_graph_list(filepath):
    """读取文件并转换为图列表"""
    graphs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for el in tqdm(lines, desc="[*] Processing SMARTS", file=sys.stderr):
        el = el.strip()
        if not el:
            continue
        
        # 转换为图
        tmp = smartsToGraph(el)
        if tmp:
            tmp = tmp.to_directed() 
            graphs.append(tmp)
            
    return graphs

def write_bms2_format(graphs, output_file):
    """将图列表写入 BMS2 格式文件"""
    with open(output_file, 'w') as f:
        for g in tqdm(graphs, desc="[*] Writing BMS2", file=sys.stderr):
            # BMS2 格式处理无向图结构
            g_undir = g.to_undirected()
            
            # 1. 写入头信息：n#{节点数} l#{标签总数}
            print(f'n#{g_undir.number_of_nodes()} l#{NUM_LABELS}', end=' ', file=f)
            
            # 2. 写入节点信息：{ID} {Label}
            for i, n in enumerate(g_undir.nodes):
                print(i, getNodeLabel(g_undir.nodes[i]), end=' ', file=f)
            
            # 3. 写入边数量：e#{边数}
            print(f'e#{g_undir.number_of_edges()}', end=' ', file=f)
            
            # 4. 写入边信息：{u} {v} {Label}
            for a, b, d in g_undir.edges(data=True):
                print(a, b, getEdgeLabel(d), end=' ', file=f)
            
            # 换行，结束当前图
            print(file=f)

# --- 主程序 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SMARTS file to BMS2 graph format')
    
    # 定义必需的两个参数
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input file containing SMARTS strings')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path to output file for BMS2 format')
    
    args = parser.parse_args()

    # 1. 读取并处理图
    print(f"Reading from: {args.input}")
    graphs = get_graph_list(args.input)
    
    # 2. 写入 BMS2 格式
    print(f"Writing to: {args.output}")
    write_bms2_format(graphs, args.output)
    
    print("Done.")