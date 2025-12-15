# -*- coding: utf-8 -*-

import sys
import os
import shutil
import itertools
import random
import time
import argparse
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

MIN_RING_NODES = 3
MIN_FINAL_SIZE = 11
SEED_ATTEMPTS = 5

organic_subset = {l: (i) for i, l in enumerate('N Cl * Br I P H O C S F'.split())}
NUM_LABELS = len(organic_subset)
bond_types = {
    l: (i) for i, l in enumerate([
        Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ])
}

def getNodeLabel(g_node_data: dict, digit: bool = True):
    if digit: return organic_subset.get(g_node_data['atom_symbol'], 0)
    return g_node_data['atom_symbol']

def getEdgeLabel(e_data: dict):
    bond_type = e_data.get('bond_type')
    if bond_type not in bond_types: bond_types[bond_type] = len(bond_types)
    return bond_types[bond_type]

def get_atom_counts(G: nx.Graph) -> Counter:
    return Counter([d['label'] for n, d in G.nodes(data=True)])

def can_be_subgraph(proto_counts: Counter, target_counts: Counter) -> bool:
    for label, count in proto_counts.items():
        if target_counts[label] < count:
            return False
    return True

def load_data(lines: List[str]):
    graphs, mols, smarts = [], [], []
    graph_atom_counts = []

    for el in tqdm(lines, desc="[1/8] Parsing and Labeling", file=sys.stderr):
        el = el.strip()
        if not el: continue
        mol = Chem.MolFromSmarts(el)
        if mol is None: continue
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            ring_info = mol.GetRingInfo()
        except:
            continue

        g = nx.Graph()
        has_ring = (ring_info.NumRings() > 0)

        for atom in mol.GetAtoms():
            g.add_node(atom.GetIdx(),
                       label=getNodeLabel({'atom_symbol': atom.GetSymbol()}),
                       atom_symbol=atom.GetSymbol(),
                       is_ring=atom.IsInRing())

        for bond in mol.GetBonds():
            btype = bond.GetBondType() if hasattr(bond, "GetBondType") else Chem.rdchem.BondType.UNSPECIFIED
            g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=btype)

        g.graph['has_ring'] = has_ring
        g.graph['num_nodes'] = g.number_of_nodes()

        graphs.append(g)
        mols.append(mol)
        smarts.append(el)
        graph_atom_counts.append(get_atom_counts(g))

    return graphs, mols, smarts, graph_atom_counts

def calibrate_adaptive_parameters(fps: List, sample_size=2000) -> List[Tuple[float, int]]:
    n = len(fps)
    if n < 2: return [(0.5, 3)]
    sample_indices = random.sample(range(n), min(n, sample_size))
    sample_fps = [fps[i] for i in sample_indices]
    sims = []
    for i in range(1, len(sample_fps)):
        s = DataStructs.BulkTanimotoSimilarity(sample_fps[i], sample_fps[:i])
        sims.extend(s)

    if not sims: return [(0.5, 3)]
    sims.sort(reverse=True)
    n_sims = len(sims)

    p05 = max(0.75, sims[int(n_sims * 0.05)])
    p20 = max(0.65, sims[int(n_sims * 0.20)])
    p50 = max(0.55, sims[int(n_sims * 0.50)])
    p70 = max(0.40, sims[int(n_sims * 0.70)])

    rounds = [(p05, 10), (p20, 8), (p50, 5), (p70, 3)]
    unique_rounds = []
    seen = set()
    for t, s in rounds:
        t = round(t, 2)
        if t not in seen:
            unique_rounds.append((t, s))
            seen.add(t)
    return unique_rounds

def find_pure_ring_mcs(G1: nx.Graph, G2: nx.Graph, min_nodes=3):
    ring_nodes = [n for n, d in G1.nodes(data=True) if d.get('is_ring', False)]
    if not ring_nodes or len(ring_nodes) < min_nodes: return None

    pure_ring_proto = G1.subgraph(ring_nodes).copy()

    if pure_ring_proto.number_of_nodes() > G2.number_of_nodes(): return None

    matcher = nx.isomorphism.GraphMatcher(
        G2, pure_ring_proto,
        node_match=lambda n1, n2: n1['label'] == n2['label'],
        edge_match=lambda e1, e2: True
    )
    if matcher.subgraph_is_isomorphic():
        return pure_ring_proto
    return None

def find_heuristic_mcs(G1: nx.Graph, G2: nx.Graph, min_nodes=3):
    if len(G1) < len(G2):
        Small, Large = G1, G2
    else:
        Small, Large = G2, G1

    if len(Small) < min_nodes: return None

    nm = lambda n1, n2: n1['label'] == n2['label']
    if nx.is_isomorphic(Small, Large, node_match=nm):
        gm = nx.isomorphism.GraphMatcher(Large, Small, node_match=nm, edge_match=lambda e1, e2: True)
        if gm.subgraph_is_isomorphic(): return Small.copy()

    best_nodes = set()

    small_nodes = sorted(Small.nodes(data=True), key=lambda x: Small.degree(x[0]), reverse=True)
    large_nodes_dict = {}
    for n, d in Large.nodes(data=True):
        l = d['label']
        if l not in large_nodes_dict: large_nodes_dict[l] = []
        large_nodes_dict[l].append(n)

    attempts = 0
    for start_node_s, data_s in small_nodes:
        label = data_s['label']
        if label not in large_nodes_dict: continue

        candidates_l = large_nodes_dict[label]
        if len(candidates_l) > 5: candidates_l = random.sample(candidates_l, 5)

        for start_node_l in candidates_l:
            current_mapping = {start_node_s: start_node_l}
            queue = []

            s_neighbors = list(Small.neighbors(start_node_s))
            l_neighbors = list(Large.neighbors(start_node_l))

            for sn in s_neighbors:
                for ln in l_neighbors:
                    if Small.nodes[sn]['label'] == Large.nodes[ln]['label']:
                        queue.append((sn, ln))

            seen_pairs = set()

            while queue:
                s_curr, l_curr = queue.pop(0)
                pair_sig = (s_curr, l_curr)
                if pair_sig in seen_pairs: continue
                seen_pairs.add(pair_sig)

                if s_curr in current_mapping or l_curr in current_mapping.values():
                    continue

                is_connected_match = False
                for existing_s, existing_l in current_mapping.items():
                    if Small.has_edge(s_curr, existing_s) and Large.has_edge(l_curr, existing_l):
                        is_connected_match = True
                        break

                if is_connected_match:
                    current_mapping[s_curr] = l_curr
                    new_s_neighbors = [n for n in Small.neighbors(s_curr) if n not in current_mapping]
                    new_l_neighbors = [n for n in Large.neighbors(l_curr) if n not in current_mapping.values()]

                    for sn in new_s_neighbors:
                        for ln in new_l_neighbors:
                            if Small.nodes[sn]['label'] == Large.nodes[ln]['label']:
                                queue.append((sn, ln))

            if len(current_mapping) > len(best_nodes):
                best_nodes = set(current_mapping.keys())
                if len(best_nodes) >= len(Small) * 0.9:
                    sub = Small.subgraph(list(best_nodes)).copy()
                    if nx.is_connected(sub): return sub

        attempts += 1
        if attempts >= SEED_ATTEMPTS: break

    if len(best_nodes) >= min_nodes:
        sub = Small.subgraph(list(best_nodes)).copy()
        if nx.is_connected(sub):
            return sub

    return None

def tune_ring_phase_parameters(ring_candidates: List[int], fps_map: Dict, graphs: List[nx.Graph]):
    print("\n--- [2/8] Ring Clustering Parameter Grid Search ---")
    if not ring_candidates: return (0.7, 5)

    fps_subset = [fps_map[i] for i in ring_candidates]
    dists = []
    n_sub = len(fps_subset)
    iter_range = range(1, n_sub)
    if n_sub > 1000: iter_range = tqdm(iter_range, desc="Calc Matrix")

    for i in iter_range:
        s = DataStructs.BulkTanimotoSimilarity(fps_subset[i], fps_subset[:i])
        dists.extend([1 - x for x in s])

    cutoffs = [0.8, 0.7, 0.6, 0.5]
    sizes = [10, 5]

    param_grid = list(itertools.product(cutoffs, sizes))
    best_score = -1
    best_params = (0.7, 5)

    for cutoff, min_size in param_grid:
        clusters = Butina.ClusterData(dists, n_sub, cutoff, isDistData=True)
        current_locked_count = 0
        check_limit = 10
        checked = 0

        for c_tuple in clusters:
            if checked >= check_limit: break

            global_members = [ring_candidates[i] for i in c_tuple]
            if len(global_members) < min_size: continue

            sorted_members = sorted(global_members, key=lambda x: graphs[x].graph['num_nodes'])
            seed_idx = sorted_members[0]
            partner_idx = sorted_members[-1]

            proto = find_pure_ring_mcs(graphs[seed_idx], graphs[partner_idx], min_nodes=MIN_RING_NODES)

            if proto:
                valid_cnt = sum(
                    1 for mid in global_members if graphs[mid].graph['num_nodes'] >= proto.number_of_nodes())
                current_locked_count += valid_cnt
            checked += 1

        if current_locked_count > best_score:
            best_score = current_locked_count
            best_params = (cutoff, min_size)

    print(f"[*] Best Params: {best_params}")
    return best_params

def merge_clusters_segregated(final_clusters):
    print("\n--- [5/8] Segregated Merging ---")
    sorted_cids = sorted(final_clusters.keys(), key=lambda k: final_clusters[k]['proto'].number_of_nodes())
    active_set = set(sorted_cids)
    cids_list = list(sorted_cids)
    n = len(cids_list)
    merged_count = 0

    proto_counts = {cid: get_atom_counts(final_clusters[cid]['proto']) for cid in cids_list}

    for i in tqdm(range(n), desc="Merging"):
        cid_a = cids_list[i]
        if cid_a not in active_set: continue
        proto_a = final_clusters[cid_a]['proto']
        source_a = final_clusters[cid_a]['source']
        counts_a = proto_counts[cid_a]

        for j in range(i + 1, n):
            cid_b = cids_list[j]
            if cid_b not in active_set: continue

            source_b = final_clusters[cid_b]['source']
            if source_a != source_b: continue

            proto_b = final_clusters[cid_b]['proto']
            if proto_a.number_of_nodes() > proto_b.number_of_nodes(): continue

            if not can_be_subgraph(counts_a, proto_counts[cid_b]): continue

            matcher = nx.isomorphism.GraphMatcher(
                proto_b, proto_a,
                node_match=lambda n1, n2: n1['label'] == n2['label'], edge_match=lambda e1, e2: True
            )
            if matcher.subgraph_is_isomorphic():
                final_clusters[cid_a]['members'].extend(final_clusters[cid_b]['members'])
                final_clusters[cid_a]['members'] = list(set(final_clusters[cid_a]['members']))
                del final_clusters[cid_b]
                active_set.remove(cid_b)
                merged_count += 1

    print(f"    -> Merged Count: {merged_count}")
    return final_clusters

def filter_final_clusters(final_clusters):
    print(f"\n--- [7/8] Final Size Filtering (Threshold >= {MIN_FINAL_SIZE}) ---")
    valid_clusters = {}
    dissolved_count = 0
    new_cid_counter = 0

    sort_key = lambda k: (0 if final_clusters[k]['source'] == 'ring_pure' else 1, -len(final_clusters[k]['members']))
    original_cids = sorted(final_clusters.keys(), key=sort_key)

    for cid in original_cids:
        data = final_clusters[cid]
        count = len(data['members'])
        if count >= MIN_FINAL_SIZE:
            new_cid_counter += 1
            valid_clusters[new_cid_counter] = data
        else:
            dissolved_count += count

    print(f"    -> Retained: {len(valid_clusters)} clusters, Dissolved: {dissolved_count} molecules")
    return valid_clusters

def make_node_order_with_proto(member_graph, proto):
    proto_nodes_sorted = sorted(list(proto.nodes()))
    gm = nx.isomorphism.GraphMatcher(member_graph, proto, node_match=lambda n1, n2: n1['label'] == n2['label'],
                                     edge_match=lambda e1, e2: True)
    try:
        m = next(gm.subgraph_isomorphisms_iter())
        mapping = {v: k for k, v in m.items()}
    except StopIteration:
        return list(member_graph.nodes())
    first = []
    for pn in proto_nodes_sorted:
        if pn in mapping:
            first.append(mapping[pn])
        else:
            return list(member_graph.nodes())
    rest = [n for n in sorted(member_graph.nodes()) if n not in set(first)]
    return first + rest

def convert_graph_to_bms2(g, node_order):
    node_mapping = {old: new for new, old in enumerate(node_order)}
    parts = [f"n#{g.number_of_nodes()} l#{NUM_LABELS}"]
    for old in node_order: parts.append(f"{node_mapping[old]} {g.nodes[old]['label']}")
    parts.append(f"e#{g.number_of_edges()}")
    for u, v, data in g.edges(data=True): parts.append(f"{node_mapping[u]} {node_mapping[v]} {getEdgeLabel(data)}")
    return " ".join(parts)

def run_pipeline(graphs, mols, smarts, graph_atom_counts, output_dir):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("--- [1/8] Fingerprint Calculation ---")
    start = time.time()
    valid_indices = []
    fps_map = {}
    fps_list_for_calib = []

    for idx, m in enumerate(mols):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            fps_map[idx] = fp
            fps_list_for_calib.append(fp)
            valid_indices.append(idx)
        except:
            pass

    classified_indices = set()
    final_clusters = {}
    cluster_counter = 0

    ring_candidates = [i for i in valid_indices if graphs[i].graph['has_ring']]
    best_ring_cutoff, best_ring_size = tune_ring_phase_parameters(ring_candidates, fps_map, graphs)

    print(f"\n>>> [Phase 1] Pure Ring Scaffold Extraction (Params: {best_ring_cutoff}, {best_ring_size})")
    if ring_candidates:
        fps_subset = [fps_map[i] for i in ring_candidates]
        dists = []
        iter_range = range(1, len(fps_subset))
        if len(fps_subset) > 1000: iter_range = tqdm(iter_range, desc="Dist Calc")

        for i in iter_range:
            dists.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps_subset[i], fps_subset[:i])])

        clusters = Butina.ClusterData(dists, len(fps_subset), best_ring_cutoff, isDistData=True)

        for c_tuple in tqdm(clusters, desc="Ring Clustering"):
            global_members = [ring_candidates[i] for i in c_tuple]
            if len(global_members) < best_ring_size: continue

            sorted_members = sorted(global_members, key=lambda x: graphs[x].graph['num_nodes'])
            seed_idx = sorted_members[0]
            partner_idx = sorted_members[-1]

            proto = find_pure_ring_mcs(graphs[seed_idx], graphs[partner_idx], min_nodes=MIN_RING_NODES)

            if proto:
                valid_members = []
                proto_counts = get_atom_counts(proto)

                for mid in global_members:
                    if not can_be_subgraph(proto_counts, graph_atom_counts[mid]): continue

                    matcher = nx.isomorphism.GraphMatcher(
                        graphs[mid], proto,
                        node_match=lambda n1, n2: n1['label'] == n2['label'],
                        edge_match=lambda e1, e2: True
                    )
                    if matcher.subgraph_is_isomorphic(): valid_members.append(mid)

                if len(valid_members) >= best_ring_size:
                    cluster_counter += 1
                    final_clusters[cluster_counter] = {
                        'proto': proto,
                        'members': valid_members,
                        'source': 'ring_pure',
                        'atom_counts': proto_counts
                    }
                    classified_indices.update(valid_members)

    print(f"    -> Locked Ring Clusters: {len(classified_indices)} molecules")

    print("\n>>> [Phase 2] Adaptive Mining (Heuristic MCS)")
    adaptive_rounds = calibrate_adaptive_parameters(fps_list_for_calib)

    for r_idx, (cutoff, min_size) in enumerate(adaptive_rounds):
        pool = [i for i in valid_indices if i not in classified_indices]
        if not pool: break
        pool_fps = [fps_map[i] for i in pool]
        if len(pool_fps) < 2: break

        dists = []
        iter_range = range(1, len(pool_fps))
        if len(pool_fps) > 2000: iter_range = tqdm(iter_range, desc="Dist Calc")
        for i in iter_range:
            dists.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(pool_fps[i], pool_fps[:i])])

        clusters = Butina.ClusterData(dists, len(pool), cutoff, isDistData=True)

        for c_tuple in tqdm(clusters, desc=f"Round {r_idx + 1}"):
            global_members = [pool[i] for i in c_tuple]
            if len(global_members) < min_size: continue

            sorted_members = sorted(global_members, key=lambda x: graphs[x].graph['num_nodes'])
            seed_node_count = graphs[sorted_members[0]].graph['num_nodes']
            dynamic_min_cs = max(3, min(6, int(seed_node_count * 0.5)))

            best_proto = None
            best_valid = []

            for seed_idx in sorted_members[:3]:
                partner_idx = sorted_members[-1]
                if seed_idx == partner_idx: continue

                proto = find_heuristic_mcs(graphs[seed_idx], graphs[partner_idx], min_nodes=dynamic_min_cs)

                if proto:
                    proto_counts = get_atom_counts(proto)
                    temp_valid = []
                    for mid in global_members:
                        if not can_be_subgraph(proto_counts, graph_atom_counts[mid]): continue

                        matcher = nx.isomorphism.GraphMatcher(
                            graphs[mid], proto,
                            node_match=lambda n1, n2: n1['label'] == n2['label'],
                            edge_match=lambda e1, e2: True
                        )
                        if matcher.subgraph_is_isomorphic(): temp_valid.append(mid)

                    if len(temp_valid) >= min_size:
                        if len(temp_valid) > len(best_valid):
                            best_proto = proto
                            best_valid = temp_valid
                        if len(temp_valid) > len(global_members) * 0.8: break
                if best_proto: break

            if best_proto:
                cluster_counter += 1
                final_clusters[cluster_counter] = {
                    'proto': best_proto,
                    'members': best_valid,
                    'source': 'adaptive',
                    'atom_counts': get_atom_counts(best_proto)
                }
                classified_indices.update(best_valid)

    final_clusters = merge_clusters_segregated(final_clusters)

    print("\n--- [6/8] Aggressive Absorption (Ring Priority) ---")
    current_other = [i for i in valid_indices if i not in classified_indices]

    sort_key = lambda k: (
        0 if final_clusters[k]['source'] == 'ring_pure' else 1,
        final_clusters[k]['proto'].number_of_nodes()
    )
    valid_cids = sorted(final_clusters.keys(), key=sort_key)

    cluster_protos = []
    for cid in valid_cids:
        cluster_protos.append({
            'cid': cid,
            'proto': final_clusters[cid]['proto'],
            'counts': final_clusters[cid].get('atom_counts', get_atom_counts(final_clusters[cid]['proto']))
        })

    reclaimed = 0
    for oid in tqdm(current_other, desc="Reclaiming"):
        g_target = graphs[oid]
        g_counts = graph_atom_counts[oid]
        g_nodes = g_target.number_of_nodes()

        for c_data in cluster_protos:
            proto = c_data['proto']

            if g_nodes < proto.number_of_nodes(): continue

            if not can_be_subgraph(c_data['counts'], g_counts): continue

            matcher = nx.isomorphism.GraphMatcher(
                g_target, proto,
                node_match=lambda n1, n2: n1['label'] == n2['label'],
                edge_match=lambda e1, e2: True
            )
            if matcher.subgraph_is_isomorphic():
                final_clusters[c_data['cid']]['members'].append(oid)
                reclaimed += 1
                break

    print(f"    -> Absorbed: {reclaimed}")

    final_clusters = filter_final_clusters(final_clusters)

    print(f"\n--- [8/8] Exporting ({len(final_clusters)} Clusters) ---")
    sorted_cids_out = sorted(final_clusters.keys(), key=sort_key)

    final_classified_set = set()
    for cid in sorted_cids_out:
        final_classified_set.update(final_clusters[cid]['members'])

    for i, cid in enumerate(sorted_cids_out):
        new_id = i + 1
        data = final_clusters[cid]
        c_bms2 = os.path.join(output_dir, f"cluster_{new_id}.dat")
        c_smarts = os.path.join(output_dir, f"cluster_{new_id}.smarts")
        proto = data['proto']
        members = data['members']

        with open(c_bms2, 'w', encoding='utf-8') as fs, open(c_smarts, 'w', encoding='utf-8') as fsm:
            proto_order = sorted(proto.nodes())
            fs.write(convert_graph_to_bms2(proto, proto_order) + "\n")
            for idx in members:
                order = make_node_order_with_proto(graphs[idx], proto)
                fs.write(convert_graph_to_bms2(graphs[idx], order) + "\n")
                fsm.write(smarts[idx] + "\n")

    other_indices = [i for i in range(len(graphs)) if i not in final_classified_set]
    if other_indices:
        with open(os.path.join(output_dir, "cluster_other.dat"), 'w', encoding='utf-8') as fs, \
                open(os.path.join(output_dir, "cluster_other.smarts"), 'w', encoding='utf-8') as fsm:
            for idx in other_indices:
                fs.write(convert_graph_to_bms2(graphs[idx], sorted(graphs[idx].nodes())) + "\n")
                fsm.write(smarts[idx] + "\n")

    end = time.time()
    print(f"Total Time: {end - start:.2f}s")
    print(f"Total: {len(graphs)}, Classified: {len(final_classified_set)}, Other: {len(other_indices)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph Clustering Pipeline")
    
    parser.add_argument('-i', '--input', required=True, type=str, help="Path to input SMARTS file")
    parser.add_argument('-o', '--output', required=True, type=str, help="Path to output directory")
    
    args = parser.parse_args()
    
    INPUT_FILE = args.input
    OUTPUT_DIR = args.output

    if os.path.exists(INPUT_FILE):
        print(f"[*] Input File: {INPUT_FILE}")
        print(f"[*] Output Dir: {OUTPUT_DIR}")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        g, m, s, c = load_data(lines)
        if g: run_pipeline(g, m, s, c, OUTPUT_DIR)
    else:
        print(f"[!] Error: File Not Found: {INPUT_FILE}")