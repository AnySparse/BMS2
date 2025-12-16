/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "candidates.hpp"
#include "device.hpp"
#include "graph.hpp"
#include "pool.hpp"
#include "signature.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <sycl/sycl.hpp>


namespace sigmo {
namespace isomorphism {
static void dumpLoadData(const std::vector<uint64_t>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    file << "UnitID,Load\n";
    for (size_t i = 0; i < data.size(); ++i) {
        file << i << "," << data[i] << "\n";
    }
    file.close();
    std::cout << "[Info] Load data dumped to " << filename << " (" << data.size() << " records)" << std::endl;
}
namespace filter {
template<CandidatesDomain D = CandidatesDomain::Query>
utils::BatchedEvent filterCandidates(sycl::queue& queue,
                                     sigmo::DeviceBatchedCSRGraph& query_graph,
                                     sigmo::DeviceBatchedCSRGraph& data_graph,
                                     sigmo::signature::Signature<>& label_signatures,  
                                     sigmo::signature::PathSignature& path_signatures,   
                                     sigmo::signature::CycleSignature& cycle_signatures,
                                     sigmo::candidates::Candidates& candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;

  sycl::range<1> local_range{device::deviceOptions.filter_work_group_size};
  sycl::range<1> global_range{((total_data_nodes + local_range[0] - 1) / local_range[0]) * local_range[0]};

  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<sigmo::device::kernels::FilterCandidatesKernel_v2>(
        sycl::nd_range<1>({global_range, local_range}),
        [=,
         candidates = candidates.getCandidatesDevice(),

         query_label_sigs = label_signatures.getDeviceQuerySignatures(), 
         data_label_sigs = label_signatures.getDeviceDataSignatures(),   
         max_labels = label_signatures.getMaxLabels(),                   

         query_path_sigs = path_signatures.getDeviceQuerySignatures(), 
         data_path_sigs = path_signatures.getDeviceDataSignatures(),   

         query_cycle_sigs = cycle_signatures.getDeviceQuerySignatures(), 
         data_cycle_sigs = cycle_signatures.getDeviceDataSignatures()   
        ](sycl::nd_item<1> item) {
          auto data_node_id = item.get_global_id(0);
          if (data_node_id >= total_data_nodes) { return; }
          
          auto query_labels = query_graph.node_labels;
          auto data_labels = data_graph.node_labels;

          auto data_row_offsets = data_graph.row_offsets;
          auto data_degree = data_row_offsets[data_node_id + 1] - data_row_offsets[data_node_id];
          auto query_row_offsets = query_graph.row_offsets;


          auto data_sig_label = data_label_sigs[data_node_id];
          auto data_sig_path = data_path_sigs[data_node_id];
          auto data_sig_cycle = data_cycle_sigs[data_node_id];


          for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {

            if (query_labels[query_node_id] != data_labels[data_node_id] && query_labels[query_node_id] != types::WILDCARD_NODE) { continue; }


            auto query_degree = query_row_offsets[query_node_id + 1] - query_row_offsets[query_node_id];
            if (query_degree > data_degree) { continue; }


            auto query_sig_path = query_path_sigs[query_node_id];
            if (!data_sig_path.contains(query_sig_path)) { continue; }


            auto query_sig_cycle = query_cycle_sigs[query_node_id];
            if (!data_sig_cycle.contains(query_sig_cycle)) { continue; }

            if constexpr (D == CandidatesDomain::Data) {
              candidates.insert(data_node_id, query_node_id);
            } else {
              candidates.atomicInsert(query_node_id, data_node_id);
            }
          }
        });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

template<CandidatesDomain D = CandidatesDomain::Query>
utils::BatchedEvent refineCandidates(sycl::queue& queue,
                                     sigmo::DeviceBatchedCSRGraph& query_graph,
                                     sigmo::DeviceBatchedCSRGraph& data_graph,
                                     sigmo::signature::Signature<>& signatures,
                                     sigmo::candidates::Candidates& candidates) {
  size_t total_query_nodes = query_graph.total_nodes;
  size_t total_data_nodes = data_graph.total_nodes;

  auto cdev = candidates.getCandidatesDevice();
  size_t total_blocks = total_query_nodes * cdev.single_node_size;

  const size_t local_range_size = 256; 
  const size_t max_safe_global_size = 1024 * 1024 * 64; 

  size_t needed_threads = ((total_blocks + local_range_size - 1) / local_range_size) * local_range_size;
  size_t clamped_global_size = std::min(needed_threads, max_safe_global_size);
  if (clamped_global_size == 0) clamped_global_size = local_range_size;

  sycl::range<1> local_range{local_range_size};
  sycl::range<1> global_range{clamped_global_size};

  auto e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<sigmo::device::kernels::RefineCandidatesKernel_v3>(
        sycl::nd_range<1>{global_range, local_range}, 
        [=,
         candidates_dev = cdev, 
         query_signatures = signatures.getDeviceQuerySignatures(),
         data_signatures = signatures.getDeviceDataSignatures(),
         max_labels = signatures.getMaxLabels()]
         (sycl::nd_item<1> item) {
          
          size_t global_id = item.get_global_linear_id();
          size_t total_threads = item.get_global_range(0);

          // Grid-Stride Loop
          for (size_t current_idx = global_id; current_idx < total_blocks; current_idx += total_threads) {
             

             types::candidates_t candidate_block = candidates_dev.candidates[current_idx];
             

             if (candidate_block == 0) continue; 

             size_t query_node_id = current_idx / candidates_dev.single_node_size;
             size_t block_idx = current_idx % candidates_dev.single_node_size;


             auto query_signature = query_signatures[query_node_id];

             types::candidates_t new_block = candidate_block;
             types::candidates_t bits_to_check = candidate_block;

             // 3. CTZ Loop
             while (bits_to_check != 0) {
               uint32_t bit = sycl::ctz(bits_to_check);
               
               size_t data_node_id = block_idx * candidates_dev.num_bits + bit;


               if (data_node_id < total_data_nodes) {

                   auto data_signature = data_signatures[data_node_id];

                   bool keep = true;

                   #pragma unroll
                   for (types::label_t l = 0; l < sigmo::signature::Signature<>::SignatureDevice::getMaxLabels(); l++) {

                        if (query_signature.getLabelCount(l) > data_signature.getLabelCount(l)) {
                            keep = false;
                            break;
                        }
                   }

                   if (!keep) {
                     new_block &= ~(static_cast<types::candidates_t>(1) << bit);
                   }
               } else {

                   new_block &= ~(static_cast<types::candidates_t>(1) << bit);
               }


               bits_to_check &= (bits_to_check - 1);
             }

             if (new_block != candidate_block) {
               candidates_dev.candidates[current_idx] = new_block;
             }
          } 
        });
  });

  utils::BatchedEvent be;
  be.add(e);
  return be;
}

} // namespace filter

namespace join {

/**
 * // TODO improve data locality and shared memory usage
 * // TODO design data structure for temporary store the candidates for each query graph in shared memory
 * // TODO improve readibility
 */

struct Stack {
  uint depth;
  size_t candidateIdx;
};

struct Mapping { // TODO: make it SOA
  size_t query_graph_id;
  size_t data_graph_id;
  types::node_t query_nodes[10];
  types::node_t data_nodes[10];
};

SYCL_EXTERNAL bool isValidMapping(types::node_t candidate,
                                  uint depth,
                                  const uint32_t* mapping,
                                  const sigmo::DeviceBatchedCSRGraph& query_graphs,
                                  uint query_graph_id,
                                  const sigmo::DeviceBatchedCSRGraph& data_graphs,
                                  uint data_graph_id) {
  for (int i = 0; i < depth; i++) {
    size_t query_nodes_offset = query_graphs.getPreviousNodes(query_graph_id);

    bool isQueryNeighbor = query_graphs.isNeighbor(i + query_nodes_offset, depth + query_nodes_offset);

    if (query_graphs.isNeighbor(i + query_nodes_offset, depth + query_nodes_offset) != data_graphs.isNeighbor(mapping[i], candidate)) {
      return false;
    }
  }

  return true;
}


constexpr size_t MAX_QUERY_NODES = 30; 

SYCL_EXTERNAL bool isValidMapping_fast(types::node_t candidate,
                                     uint depth,
                                     const types::node_t* mapping, 
                                     const bool query_adj_matrix[][MAX_QUERY_NODES], 
                                     const sigmo::DeviceBatchedCSRGraph& data_graphs,
                                     uint data_graph_id) {
  for (int i = 0; i < depth; i++) {

    bool isQueryNeighbor = query_adj_matrix[i][depth]; 


    if (isQueryNeighbor != data_graphs.isNeighbor(mapping[i], candidate)) {
      return false;
    }
  }
  return true;
}

SYCL_EXTERNAL bool isValidMapping_fast_bitwise(
    types::node_t candidate,
    uint depth,
    const types::node_t* mapping, 
    const uint32_t* query_adj_masks,
    const sigmo::DeviceBatchedCSRGraph& data_graphs,
    uint data_graph_id) 
{

  uint32_t current_adj_mask = query_adj_masks[depth];

  for (int i = 0; i < depth; i++) {

    bool isQueryNeighbor = (current_adj_mask >> i) & 1;

    if (isQueryNeighbor != data_graphs.isNeighbor(mapping[i], candidate)) {
      return false;
    }
  }
  return true;
}
// In namespace sigmo::isomorphism::join

template<int SG_SIZE = 32>
utils::BatchedEvent joinCandidates(
    sycl::queue& queue,
    sigmo::DeviceBatchedCSRGraph& query_graphs,
    sigmo::DeviceBatchedCSRGraph& data_graphs,
    sigmo::candidates::Candidates& candidates,
    sigmo::isomorphism::mapping::GMCR& gmcr,
    size_t* num_matches,
    bool find_first = true) 
{
  utils::BatchedEvent e;
  const size_t total_tasks = gmcr.getGMCRDevice().total_query_indices;
  const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;
  
  constexpr size_t TASKS_PER_WORKGROUP = 256; 
  const size_t num_workgroups = (total_tasks + TASKS_PER_WORKGROUP - 1) / TASKS_PER_WORKGROUP;
  const size_t global_size = num_workgroups * preferred_workgroup_size;

  auto e1 = queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> shared_task_idx_accessor(sycl::range<1>(1), cgh);

    cgh.parallel_for<class JoinCandidates3Optimized>(
        sycl::nd_range<1>{global_size, preferred_workgroup_size},
        [=, query_graphs = query_graphs, data_graphs = data_graphs, 
         candidates = candidates.getCandidatesDevice(), gmcr = gmcr.getGMCRDevice()]
        (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {

          const auto wg = item.get_group();
          const auto sg = item.get_sub_group();
          
          const size_t wgid = wg.get_group_linear_id();
          const size_t wglid = wg.get_local_linear_id();
          const size_t sglid = sg.get_local_linear_id();
          const size_t sg_size = sg.get_local_range()[0];

          sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> 
              num_matches_ref{num_matches[0]};

          types::node_t mapping[MAX_QUERY_NODES]; 
          size_t private_num_matches = 0;

          uint32_t query_adj_masks[MAX_QUERY_NODES];
          uint32_t candidate_counts[MAX_QUERY_NODES];

          const size_t task_chunk_start = wgid * TASKS_PER_WORKGROUP;
          if (task_chunk_start >= total_tasks) return;
          
          const size_t num_tasks_in_chunk = sycl::min(TASKS_PER_WORKGROUP, total_tasks - task_chunk_start);

          if (wglid == 0) { shared_task_idx_accessor[0] = 0; }
          sycl::group_barrier(wg);

          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                           sycl::access::address_space::local_space>
              atomic_task_idx(shared_task_idx_accessor[0]);

          while (true) {
            uint32_t my_task_idx_in_chunk;

            if (sg.leader()) {
                my_task_idx_in_chunk = atomic_task_idx.fetch_add(1);
            }
            my_task_idx_in_chunk = sycl::select_from_group(sg, my_task_idx_in_chunk, 0);

            if (my_task_idx_in_chunk >= num_tasks_in_chunk) break; 
            
            const size_t global_task_idx = task_chunk_start + my_task_idx_in_chunk;
            
            const uint32_t query_graph_id = gmcr.query_graph_indices[global_task_idx];
            size_t data_graph_id = utils::binarySearch(
                gmcr.data_graph_offsets, data_graphs.num_graphs + 1, global_task_idx);

            const uint32_t start_data_graph = data_graphs.graph_offsets[data_graph_id];
            const uint32_t end_data_graph   = data_graphs.graph_offsets[data_graph_id + 1];
            const uint32_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
            const uint16_t num_query_nodes    = query_graphs.getGraphNodes(query_graph_id);

            bool possible = true; // Optimization flag


            for(uint16_t i = 0; i < num_query_nodes; ++i) {
                uint32_t count = candidates.getCandidatesCount(
                    i + offset_query_nodes, start_data_graph, end_data_graph);
                
                candidate_counts[i] = count;


                if (count == 0) {
                    possible = false;
                    break; 
                }

                uint32_t mask = 0;
                for(uint16_t j = 0; j < i; ++j) {
                    if (query_graphs.isNeighbor(offset_query_nodes + i,
                                                offset_query_nodes + j)) {
                        mask |= (1u << j);
                    }
                }
                query_adj_masks[i] = mask;
            }


            if (!possible) continue;


            const uint32_t root_candidates_count = candidate_counts[0];
            size_t current_task_matches = 0;
            

            for (uint32_t c_idx = sglid; c_idx < root_candidates_count; c_idx += sg_size) {

                auto root_candidate = candidates.getCandidateAt(
                    offset_query_nodes + 0, c_idx, start_data_graph, end_data_graph);
                
                Stack stack[MAX_QUERY_NODES];
                utils::detail::Bitset<uint64_t> visited{start_data_graph};
                
                mapping[0] = root_candidate;
                visited.set(root_candidate);
                uint top = 0;
                stack[top++] = {1, 0};
                
                while (top > 0) {
                    auto frame = stack[top - 1];
                    auto query_node = frame.depth;

                    if (frame.depth == num_query_nodes) { 
                        current_task_matches++;
                        top--;
                        if (find_first) {
                            break; 
                        } else {
                            continue;
                        }
                    }

                    if (frame.candidateIdx >= candidate_counts[query_node]) {
                        top--;
                        visited.unset(mapping[query_node]);
                        continue;
                    }

                    auto candidate = candidates.getCandidateAt(
                        query_node + offset_query_nodes, frame.candidateIdx,
                        start_data_graph, end_data_graph);
                    stack[top - 1].candidateIdx++; 

                    if (visited.get(candidate)) { continue; }

                    if (isValidMapping_fast_bitwise(
                            candidate, frame.depth, mapping,
                            query_adj_masks, data_graphs, data_graph_id)) {
                        mapping[frame.depth] = candidate;
                        visited.set(candidate);
                        stack[top++] = {static_cast<uint>(frame.depth + 1), 0};
                    }
                } 
                
                if (find_first && current_task_matches > 0) break;
            } 

            if (find_first) {
                bool any_found = sycl::any_of_group(sg, current_task_matches > 0);
                if (any_found && sg.leader()) {
                    private_num_matches++;
                }
            } else {
                private_num_matches += current_task_matches;
            }
          } 

          sycl::group_barrier(wg);
          private_num_matches = sycl::reduce_over_group(wg, private_num_matches, sycl::plus<>());

          if (wg.leader()) num_matches_ref += private_num_matches;
        });
  });

  e.add(e1);
  return e;
}



SYCL_EXTERNAL void defineMatchingOrder(sycl::nd_item<1> item,
                                       types::node_t* mapping,
                                       size_t& max_candidates,
                                       const sigmo::candidates::Candidates::CandidatesDevice& candidates,
                                       uint query_graph_offsets,
                                       uint total_query_nodes,
                                       uint data_graph_start,
                                       uint data_graph_end) {
  uint sgid = item.get_sub_group().get_local_linear_id();
  int current_node = -1;
  int current_candidates = 0;
  if (sgid < total_query_nodes) {
    current_node = sgid;
    current_candidates = candidates.getCandidatesCount(current_node + query_graph_offsets, data_graph_start, data_graph_end);
  }

  max_candidates = sycl::reduce_over_group(item.get_sub_group(), current_candidates, sycl::maximum<>());

  if (current_candidates != max_candidates) { current_node = -1; }

  current_node = sycl::reduce_over_group(item.get_sub_group(), current_node, sycl::maximum<>());
  mapping[0] = current_node;
  for (uint i = 0, j = 1; i < total_query_nodes; ++i) {
    if (i == current_node) continue;
    mapping[j++] = i;
  }
}

//reuse
utils::BatchedEvent joinPartialCandidates(sycl::queue& queue,
                                          sigmo::DeviceBatchedCSRGraph& query_graphs,
                                          sigmo::DeviceBatchedCSRGraph& data_graphs,
                                          sigmo::candidates::Candidates& candidates,
                                          const size_t end,
                                          int* partial_matches,
                                          size_t* num_partial_matches) {
    utils::BatchedEvent e;
    const size_t total_data_graphs = data_graphs.num_graphs;
    const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;

    constexpr size_t MAX_QUERY_NODES = 30;
    if (end > MAX_QUERY_NODES) {
        throw std::runtime_error("Partial match depth 'end' exceeds MAX_QUERY_NODES");
    }

    sycl::nd_range<1> nd_range{total_data_graphs * preferred_workgroup_size, preferred_workgroup_size};

    auto e1 = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<device::kernels::JoinPartialCandidatesKernel>(
            nd_range,
            [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()](sycl::nd_item<1> item) {
                const auto  g      = item.get_group();
                const size_t gid   = g.get_group_linear_id();
                const size_t lid   = item.get_local_linear_id();
                const size_t lsize = g.get_local_range()[0];
                const size_t gcount= g.get_group_linear_range();

                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    global_match_counter{num_partial_matches[0]};


                for (uint32_t data_graph_id = gid; data_graph_id < total_data_graphs; data_graph_id += gcount) {
                    const uint32_t qid = 0; 
                    const uint32_t qnode_off = query_graphs.getPreviousNodes(qid);
                    const uint16_t qnodes    = query_graphs.getGraphNodes(qid);
                    if (end > qnodes) continue;

                    const uint32_t dg_beg = data_graphs.graph_offsets[data_graph_id];
                    const uint32_t dg_end = data_graphs.graph_offsets[data_graph_id + 1];


                    const uint32_t cand0_count = candidates.getCandidatesCount(qnode_off + 0, dg_beg, dg_end);


                    for (uint32_t c0 = lid; c0 < cand0_count; c0 += lsize) {

                        types::node_t mapping[MAX_QUERY_NODES];
                        utils::detail::Bitset<uint64_t> visited{dg_beg};
                        struct Stack { uint depth; uint32_t cand_idx; };
                        Stack stack[MAX_QUERY_NODES];


                        const auto v0 = candidates.getCandidateAt(qnode_off + 0, c0, dg_beg, dg_end);
                        mapping[0] = v0;
                        visited.clear();
                        visited.set(v0);

                        uint top = 0;
                        stack[top++] = {1u, 0u};

                        while (top > 0) {
                            auto &frame = stack[top - 1];
                            const uint qdepth = frame.depth;


                            if (qdepth == end) {

                                const size_t idx = global_match_counter.fetch_add(1);
                                const size_t base = idx * (end + 1);

                                for (uint32_t i = 0; i < end; ++i) {
                                    partial_matches[base + i] = mapping[i];
                                }

                                partial_matches[base + end] = static_cast<int>(data_graph_id);


                                top--;
                                continue;
                            }


                            const uint32_t qnode = qdepth;
                            const uint32_t cand_cnt = candidates.getCandidatesCount(qnode_off + qnode, dg_beg, dg_end);
                            if (frame.cand_idx >= cand_cnt) {
                                top--;

                                if (qdepth > 0) {
                                    visited.unset(mapping[qdepth]);
                                }
                                continue;
                            }


                            const auto dv = candidates.getCandidateAt(qnode_off + qnode, frame.cand_idx, dg_beg, dg_end);
                            frame.cand_idx++;


                            if (visited.get(dv)) continue;


                            if (qdepth == 0 || isValidMapping(dv, qdepth, mapping, query_graphs, qid, data_graphs, data_graph_id)) {
                                mapping[qdepth] = dv;
                                visited.set(dv);
                                stack[top++] = {qdepth + 1, 0u};
                            }
                        } // while DFS


                        visited.clear();
                    } // for c0
                } // for data_graph_id
            });
    });

    e.add(e1);
    return e;
}

utils::BatchedEvent joinWithPartialMatches(sycl::queue& queue,
                                           sigmo::DeviceBatchedCSRGraph& query_graphs,
                                           sigmo::DeviceBatchedCSRGraph& data_graphs,
                                           sigmo::candidates::Candidates& candidates,
                                           sigmo::isomorphism::mapping::GMCR& gmcr,
                                           const size_t end,
                                           const int* partial_matches,
                                           const size_t num_partial_matches,
                                           size_t* total_full_matches) {
    utils::BatchedEvent e;
    if (num_partial_matches == 0) {
        return e; }

    const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;
    

    size_t global_size = ((num_partial_matches + preferred_workgroup_size - 1) / preferred_workgroup_size) * preferred_workgroup_size;
    sycl::nd_range<1> nd_range{num_partial_matches * preferred_workgroup_size, preferred_workgroup_size};

    constexpr size_t MAX_QUERY_NODES = 30;

    auto e1 = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<device::kernels::JoinWithPartialMatchesKernel>( 
            nd_range,
            [=,query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice(), gmcr = gmcr.getGMCRDevice()](sycl::nd_item<1> item) {
                const auto wg = item.get_group();
                const size_t wgid = wg.get_group_linear_id();   
                const size_t wglid = wg.get_local_linear_id();  
                const size_t wgsize = wg.get_local_range()[0];

                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> total_matches_ref{total_full_matches[0]};

                size_t private_num_matches = 0;


                for (size_t partial_match_idx = wgid; partial_match_idx < num_partial_matches; partial_match_idx += wg.get_group_linear_range()) {
                    

                    types::node_t mapping[MAX_QUERY_NODES];
                    const size_t partial_match_size = end + 1;
                    const size_t base_offset = partial_match_idx * partial_match_size;
                    
                    for (size_t i = 0; i < end; ++i) {
                        mapping[i] = partial_matches[base_offset + i];
                    }
                    const uint32_t data_graph_id = partial_matches[base_offset + end];
                    
                    const uint32_t start_data_graph = data_graphs.graph_offsets[data_graph_id];
                    const uint32_t end_data_graph = data_graphs.graph_offsets[data_graph_id + 1];

                    const uint32_t start_query = gmcr.data_graph_offsets[data_graph_id];
                    const uint32_t end_query = gmcr.data_graph_offsets[data_graph_id + 1];


                    for (uint32_t query_graph_it = wglid; query_graph_it < (end_query - start_query); query_graph_it += wgsize) {
                        const uint32_t query_graph_id = gmcr.query_graph_indices[start_query + query_graph_it];
                        
                        const uint32_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
                        const uint16_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);

                        if (end >= num_query_nodes) {
                            if (end == num_query_nodes) {
                                private_num_matches++;
                            }
                            continue;
                        }


                        Stack stack[MAX_QUERY_NODES];
                        utils::detail::Bitset<uint64_t> visited{start_data_graph};

                        for (size_t i = 0; i < end; ++i) {
                            visited.set(mapping[i]);
                        }

                        uint top = 0;

                        stack[top++] = { (uint16_t)end, 0 };


                        while (top > 0) {
                            auto frame = stack[top - 1];
                            auto query_node = frame.depth;


                            if (frame.depth == num_query_nodes) {
                                private_num_matches++;
                                top--;
                                continue;
                            }


                            if (frame.candidateIdx >= candidates.getCandidatesCount(query_node + offset_query_nodes, start_data_graph, end_data_graph)) {
                                top--;
                                if (top > 0 && frame.depth > end) {
                                     visited.unset(mapping[query_node]);
                                }
                                continue;
                            }
                            
                            auto candidate = candidates.getCandidateAt(query_node + offset_query_nodes, frame.candidateIdx, start_data_graph, end_data_graph);
                            stack[top - 1].candidateIdx++;

                            if (visited.get(candidate)) {
                                continue;
                            }
                            

                            if (isValidMapping(candidate, frame.depth, mapping, query_graphs, query_graph_id, data_graphs, data_graph_id)) {
                                mapping[frame.depth] = candidate;
                                visited.set(candidate);
                                stack[top++] = { (uint16_t)(frame.depth + 1), 0 };
                            }
                        }
                    }
                }


                private_num_matches = sycl::reduce_over_group(wg, private_num_matches, sycl::plus<>());
                

                if (wg.leader()) {
                    total_matches_ref += private_num_matches;
                }
            });
    });

    e.add(e1);
    return e;
}

template<int SG_SIZE = 32> 
utils::BatchedEvent joinWithPartialMatches(sycl::queue& queue,
                                           sigmo::DeviceBatchedCSRGraph& query_graphs,
                                           sigmo::DeviceBatchedCSRGraph& data_graphs,
                                           sigmo::candidates::Candidates& candidates,
                                           sigmo::isomorphism::mapping::GMCR& gmcr,
                                           const size_t end,
                                           const int* partial_matches,          // [num_partial_matches * (end+1)]
                                           const size_t num_partial_matches,
                                           size_t* total_full_matches,
                                           bool find_first,                      // true=find_first, false=find_all
                                           uint32_t* pair_done                   // [num_query_graphs * num_data_graphs], 0/1
                                           )
{
    utils::BatchedEvent e;
    if (num_partial_matches == 0) return e;

    const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;
    constexpr size_t MAX_QUERY_NODES = 30;
    

    const size_t num_subgroups_per_wg = preferred_workgroup_size / SG_SIZE;

    const size_t num_wgs = (num_partial_matches + num_subgroups_per_wg - 1) / num_subgroups_per_wg;
    
    sycl::nd_range<1> nd_range{num_wgs * preferred_workgroup_size, preferred_workgroup_size};

    auto ev = queue.submit([&](sycl::handler& cgh) {

        cgh.parallel_for<class JoinWithPartialMatchesKernel_SubGroupOptimized_v2>(
            nd_range,
            [=, query_graphs = query_graphs,
                data_graphs = data_graphs,
                candidates  = candidates.getCandidatesDevice(),
                gmcr        = gmcr.getGMCRDevice()](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] 
            {
                const auto wg = item.get_group();
                const auto sg = item.get_sub_group();
                
                const size_t wgid = wg.get_group_linear_id();
                const size_t sglid = sg.get_local_linear_id(); 
                

                const size_t num_sgs_in_wg = wg.get_local_range()[0] / SG_SIZE;
                const size_t sg_in_wg_id = item.get_local_linear_id() / SG_SIZE;
                
  
                const size_t task_id = wgid * num_sgs_in_wg + sg_in_wg_id;

                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    total_matches_ref{total_full_matches[0]};

 
                if (task_id >= num_partial_matches) return;

                const uint32_t num_data_graphs = data_graphs.num_graphs;
                

                types::node_t mapping[MAX_QUERY_NODES];
                const size_t rec_size = end + 1;
                const size_t base_offset = task_id * rec_size;
                

                const uint32_t data_graph_id = static_cast<uint32_t>(partial_matches[base_offset + end]);

                for (size_t i = 0; i < end; ++i) {
                    mapping[i] = partial_matches[base_offset + i];
                }

                const uint32_t dg_beg = data_graphs.graph_offsets[data_graph_id];
                const uint32_t dg_end = data_graphs.graph_offsets[data_graph_id + 1];
                const uint32_t qbeg = gmcr.data_graph_offsets[data_graph_id];
                const uint32_t qend = gmcr.data_graph_offsets[data_graph_id + 1];


                uint32_t query_adj_masks[MAX_QUERY_NODES];
                uint32_t candidate_counts[MAX_QUERY_NODES];
                
                size_t private_count = 0;

                for (uint32_t it = 0; it < (qend - qbeg); ++it) {
                    const uint32_t query_graph_id = gmcr.query_graph_indices[qbeg + it];
        
                    const uint32_t flag_idx = query_graph_id * num_data_graphs + data_graph_id;
                    auto flag_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(pair_done[flag_idx]);

       
                    if (find_first && flag_ref.load() != 0u) continue;

                    const uint32_t qoff = query_graphs.getPreviousNodes(query_graph_id);
                    const uint16_t qn = query_graphs.getGraphNodes(query_graph_id);

  
                    if (end >= qn) {
        
                        if (sglid == 0 && end == qn) {
                            if (find_first) {
                                uint32_t old = flag_ref.exchange(1u);
                                if (old == 0u) private_count++;
                            } else {
                                private_count++;
                            }
                        }
                        continue;
                    }


                    for (uint16_t i = 0; i < qn; ++i) {
                        candidate_counts[i] = candidates.getCandidatesCount(qoff + i, dg_beg, dg_end);
                        
)
                        uint32_t mask = 0;
                        for (uint16_t j = 0; j < i; ++j) {
                            if (query_graphs.isNeighbor(qoff + i, qoff + j)) mask |= (1u << j);
                        }
                        query_adj_masks[i] = mask;
                    }


                    
                    const uint32_t current_candidates_count = candidate_counts[end];
                    size_t thread_matches = 0;

                    for (uint32_t c_idx = sglid; c_idx < current_candidates_count; c_idx += SG_SIZE) {
                        
          
                        auto branch_root_cand = candidates.getCandidateAt(qoff + end, c_idx, dg_beg, dg_end);

                        bool conflict = false;
                        for(size_t i=0; i<end; ++i) {
                            if (mapping[i] == branch_root_cand) { conflict = true; break; }
                        }
                        if (conflict) continue;


                        if (!isValidMapping_fast_bitwise(branch_root_cand, end, mapping, query_adj_masks, data_graphs, data_graph_id)) {
                            continue;
                        }

                        Stack stack[MAX_QUERY_NODES];
                        utils::detail::Bitset<uint64_t> visited{dg_beg};
                        

                        for (size_t i = 0; i < end; ++i) visited.set(mapping[i]);
                        visited.set(branch_root_cand);
                        

                        types::node_t local_mapping[MAX_QUERY_NODES];
                        for(size_t i=0; i<end; ++i) local_mapping[i] = mapping[i];
                        local_mapping[end] = branch_root_cand;

                        uint top = 0;
                        stack[top++] = {static_cast<uint>(end + 1), 0};


                        while (top > 0) {
                            auto frame = stack[top - 1];
                            auto q_node = frame.depth;


                            if (frame.depth == qn) {
                                thread_matches++;
                                top--;
                                if (find_first) break; 
                                else continue;
                            }

                            if (frame.candidateIdx >= candidate_counts[q_node]) {
                                top--;
                                visited.unset(local_mapping[q_node]);
                                continue;
                            }

           
                            auto cand = candidates.getCandidateAt(q_node + qoff, frame.candidateIdx, dg_beg, dg_end);
                            stack[top - 1].candidateIdx++;

                            if (visited.get(cand)) continue;

             
                            if (isValidMapping_fast_bitwise(cand, frame.depth, local_mapping, query_adj_masks, data_graphs, data_graph_id)) {
                                local_mapping[frame.depth] = cand;
                                visited.set(cand);
                                stack[top++] = {frame.depth + 1, 0};
                            }
                        } // End While DFS
                        
                        if (find_first && thread_matches > 0) break;

                    } // End Parallel Expansion For-Loop

                    // ==========================

                    // ==========================
                    if (find_first) {

                        bool any_found = sycl::any_of_group(sg, thread_matches > 0);
                        if (any_found) {

                            if (sglid == 0) {
                                uint32_t old = flag_ref.exchange(1u);
                                if (old == 0u) private_count++;
                            }
                        }
                    } else {

                        size_t total_sg_matches = sycl::reduce_over_group(sg, thread_matches, sycl::plus<>());
                        if (sglid == 0) private_count += total_sg_matches;
                    }

                } // End Query Graph Loop


                if (sglid == 0 && private_count > 0) {
                    total_matches_ref += private_count;
                }
            });
    });

    e.add(ev);
    return e;
}
template<size_t BUFFER_SIZE_INTS = 64> 
struct LocalOutputBuffer {
    int buffer[BUFFER_SIZE_INTS];
    int current_idx = 0;
    

    bool is_full(size_t match_len) const {
        return (current_idx + match_len + 1) > BUFFER_SIZE_INTS;
    }


    void append(const types::node_t* mapping, size_t end, int data_graph_id) {
        for (size_t i = 0; i < end; ++i) {
            buffer[current_idx++] = mapping[i];
        }
        buffer[current_idx++] = data_graph_id;
    }


    void flush(sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>& global_counter,
               int* global_output, 
               size_t end) {
        if (current_idx == 0) return;


        size_t match_size = end + 1;
        size_t num_matches_in_buffer = current_idx / match_size;


        size_t global_start_idx = global_counter.fetch_add(num_matches_in_buffer);
        

        size_t global_offset = global_start_idx * match_size;
        
        for (int i = 0; i < current_idx; ++i) {
            global_output[global_offset + i] = buffer[i];
        }


        current_idx = 0;
    }
};

template<int SG_SIZE = 32>
utils::BatchedEvent joinPartialCandidates2(sycl::queue& queue,
                                          sigmo::DeviceBatchedCSRGraph& query_graphs,
                                          sigmo::DeviceBatchedCSRGraph& data_graphs,
                                          sigmo::candidates::Candidates& candidates,
                                          const size_t end,
                                          int* partial_matches,
                                          size_t* num_partial_matches) {
    utils::BatchedEvent e;
    const size_t total_data_graphs = data_graphs.num_graphs;
    const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;
    
    constexpr size_t MAX_QUERY_NODES = 30;
    if (end > MAX_QUERY_NODES) {
        throw std::runtime_error("Partial match depth 'end' exceeds MAX_QUERY_NODES");
    }


    constexpr size_t TASKS_PER_WORKGROUP = 256; 
    const size_t num_workgroups = (total_data_graphs + TASKS_PER_WORKGROUP - 1) / TASKS_PER_WORKGROUP;
    const size_t global_size = num_workgroups * preferred_workgroup_size;

    sycl::nd_range<1> nd_range{global_size, preferred_workgroup_size};

    auto e1 = queue.submit([&](sycl::handler& cgh) {

        sycl::local_accessor<uint32_t, 1> shared_task_idx_accessor(sycl::range<1>(1), cgh);

        cgh.parallel_for<class JoinPartialCandidates2HighlyOptimized>( 
            nd_range,
            [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()]
            (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                
                const auto wg = item.get_group();
                const auto sg = item.get_sub_group();
                const size_t sglid = sg.get_local_linear_id();
                const size_t sg_size = sg.get_local_range()[0];

        
                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    global_match_counter{num_partial_matches[0]};

    
                LocalOutputBuffer<64> output_aggregator;

    
                uint32_t query_adj_masks[MAX_QUERY_NODES];
     
                uint32_t candidate_counts[MAX_QUERY_NODES];

                const size_t wgid = wg.get_group_linear_id();
                const size_t task_chunk_start = wgid * TASKS_PER_WORKGROUP;
                if (task_chunk_start >= total_data_graphs) return;

                const size_t num_tasks_in_chunk = sycl::min(TASKS_PER_WORKGROUP, total_data_graphs - task_chunk_start);

                if (wg.get_local_linear_id() == 0) { shared_task_idx_accessor[0] = 0; }
                sycl::group_barrier(wg);

                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
                    atomic_task_idx(shared_task_idx_accessor[0]);

                while (true) {
                    uint32_t my_task_idx_in_chunk;

                    if (sg.leader()) { my_task_idx_in_chunk = atomic_task_idx.fetch_add(1); }
                    my_task_idx_in_chunk = sycl::select_from_group(sg, my_task_idx_in_chunk, 0);

                    if (my_task_idx_in_chunk >= num_tasks_in_chunk) break;

                    const uint32_t data_graph_id = task_chunk_start + my_task_idx_in_chunk;


                    const uint32_t qid = 0; 
                    const uint32_t qnode_off = query_graphs.getPreviousNodes(qid);
                    const uint16_t qnodes = query_graphs.getGraphNodes(qid);
                    
                    if (end > qnodes) continue;

                    const uint32_t dg_beg = data_graphs.graph_offsets[data_graph_id];
                    const uint32_t dg_end = data_graphs.graph_offsets[data_graph_id + 1];


                    for(uint16_t i = 0; i < qnodes; ++i) {
                        candidate_counts[i] = candidates.getCandidatesCount(i + qnode_off, dg_beg, dg_end);
                        
                        uint32_t mask = 0;
                        for(uint16_t j = 0; j < i; ++j) {
                             if (query_graphs.isNeighbor(qnode_off + i, qnode_off + j)) {
                                 mask |= (1u << j);
                             }
                        }
                        query_adj_masks[i] = mask;
                    }

         
                    const uint32_t root_count = candidate_counts[0];

                    for (uint32_t c0 = sglid; c0 < root_count; c0 += sg_size) {
                         types::node_t mapping[MAX_QUERY_NODES];
                         utils::detail::Bitset<uint64_t> visited{dg_beg};
                         struct Stack { uint depth; uint32_t cand_idx; };
                         Stack stack[MAX_QUERY_NODES];

                         auto v0 = candidates.getCandidateAt(qnode_off + 0, c0, dg_beg, dg_end);
                         mapping[0] = v0;
                         visited.set(v0);
                         
                         uint top = 0;
                         stack[top++] = {1u, 0u};

                         while(top > 0) {
                             auto &frame = stack[top - 1];
                             const uint qdepth = frame.depth;

         
                             if (qdepth == end) {
          
                                 if (output_aggregator.is_full(end)) {
                                     output_aggregator.flush(global_match_counter, partial_matches, end);
                                 }
                                 
             
                                 output_aggregator.append(mapping, end, static_cast<int>(data_graph_id));

                                 top--;
                                 continue;
                             }

                             if (frame.cand_idx >= candidate_counts[qdepth]) {
                                 top--;
                                 visited.unset(mapping[qdepth]);
                                 continue;
                             }

                             auto dv = candidates.getCandidateAt(qnode_off + qdepth, frame.cand_idx, dg_beg, dg_end);
                             frame.cand_idx++; 
                             
                             if (visited.get(dv)) continue;

                             // L4: Bitwise Fast Check
                             if (isValidMapping_fast_bitwise(dv, qdepth, mapping, query_adj_masks, data_graphs, data_graph_id)) {
                                 mapping[qdepth] = dv;
                                 visited.set(dv);
                                 stack[top++] = {qdepth + 1, 0u};
                             }
                         }
                    }
                } // End Task Loop


                output_aggregator.flush(global_match_counter, partial_matches, end);
            });
    });

    e.add(e1);
    return e;
}

} // namespace join
} // namespace isomorphism
} // namespace sigmo
