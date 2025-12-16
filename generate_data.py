

import argparse
import sys
import networkx as nx
from rdkit import Chem
from tqdm import tqdm


organic_subset = {l: (i) for i, l in enumerate('N Cl * Br I P H O C S F'.split())}
bond_types = {l: (i) for i, l in enumerate([
    Chem.rdchem.BondType.UNSPECIFIED, 
    Chem.rdchem.BondType.SINGLE, 
    Chem.rdchem.BondType.DOUBLE, 
    Chem.rdchem.BondType.TRIPLE, 
    Chem.rdchem.BondType.AROMATIC
])}
NUM_LABELS = len(organic_subset)



def molToGraph(mol):

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

    mol_smart = Chem.MolFromSmarts(smarts)
    if mol_smart is None:
        return None
    return molToGraph(mol_smart)

def getNodeLabel(g: dict, digit: bool = True):

    if digit:
        if g['atom_symbol'] in organic_subset:
            return organic_subset[g['atom_symbol']]
        else:
            return 0
    return g['atom_symbol']

def getEdgeLabel(e: dict):

    if e['bond_type'] not in bond_types:
        bond_types[e['bond_type']] = len(bond_types)
    return bond_types[e['bond_type']]

def get_graph_list(filepath):

    graphs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for el in tqdm(lines, desc="[*] Processing SMARTS", file=sys.stderr):
        el = el.strip()
        if not el:
            continue
        
        tmp = smartsToGraph(el)
        if tmp:
            tmp = tmp.to_directed() 
            graphs.append(tmp)
            
    return graphs

def write_bms2_format(graphs, output_file):

    with open(output_file, 'w') as f:
        for g in tqdm(graphs, desc="[*] Writing BMS2", file=sys.stderr):
            g_undir = g.to_undirected()
            
            print(f'n#{g_undir.number_of_nodes()} l#{NUM_LABELS}', end=' ', file=f)
            
            for i, n in enumerate(g_undir.nodes):
                print(i, getNodeLabel(g_undir.nodes[i]), end=' ', file=f)
            
            print(f'e#{g_undir.number_of_edges()}', end=' ', file=f)
            
            for a, b, d in g_undir.edges(data=True):
                print(a, b, getEdgeLabel(d), end=' ', file=f)
            
            print(file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SMARTS file to BMS2 graph format')
    
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input file containing SMARTS strings')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path to output file for BMS2 format')
    
    args = parser.parse_args()

    print(f"Reading from: {args.input}")
    graphs = get_graph_list(args.input)
    
    print(f"Writing to: {args.output}")
    write_bms2_format(graphs, args.output)
    

    print("Done.")
