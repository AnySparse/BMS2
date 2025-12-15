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
// 辅助函数：将负载数据导出到 CSV 文件
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
                                     sigmo::signature::Signature<>& label_signatures,      // <--- 修改：重命名
                                     sigmo::signature::PathSignature& path_signatures,   // <--- 新增
                                     sigmo::signature::CycleSignature& cycle_signatures, // <--- 新增
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
         // 标签签名
         query_label_sigs = label_signatures.getDeviceQuerySignatures(), // <--- 修改
         data_label_sigs = label_signatures.getDeviceDataSignatures(),   // <--- 修改
         max_labels = label_signatures.getMaxLabels(),                   // <--- 修改
         // 路径签名
         query_path_sigs = path_signatures.getDeviceQuerySignatures(), // <--- 新增
         data_path_sigs = path_signatures.getDeviceDataSignatures(),   // <--- 新增
         // 环签名
         query_cycle_sigs = cycle_signatures.getDeviceQuerySignatures(), // <--- 新增
         data_cycle_sigs = cycle_signatures.getDeviceDataSignatures()    // <--- 新增
        ](sycl::nd_item<1> item) {
          auto data_node_id = item.get_global_id(0);
          if (data_node_id >= total_data_nodes) { return; }
          
          auto query_labels = query_graph.node_labels;
          auto data_labels = data_graph.node_labels;

          auto data_row_offsets = data_graph.row_offsets;
          auto data_degree = data_row_offsets[data_node_id + 1] - data_row_offsets[data_node_id];
          auto query_row_offsets = query_graph.row_offsets;

          // ========== 新增代码 [开始] ==========
          // 预取数据节点的所有签名
          auto data_sig_label = data_label_sigs[data_node_id];
          auto data_sig_path = data_path_sigs[data_node_id];
          auto data_sig_cycle = data_cycle_sigs[data_node_id];
          // ========== 新增代码 [结束] ==========

          for (size_t query_node_id = 0; query_node_id < total_query_nodes; ++query_node_id) {
            // 1. 标签检查
            if (query_labels[query_node_id] != data_labels[data_node_id] && query_labels[query_node_id] != types::WILDCARD_NODE) { continue; }

            // 2. 度检查
            auto query_degree = query_row_offsets[query_node_id + 1] - query_row_offsets[query_node_id];
            if (query_degree > data_degree) { continue; }

            // ========== 新增代码 [开始] ==========
            // 4. 路径签名检查 (来自 refineCandidatesByPath)
            auto query_sig_path = query_path_sigs[query_node_id];
            if (!data_sig_path.contains(query_sig_path)) { continue; }

            // 5. 环签名检查 (来自 refineCandidatesByCycle)
            auto query_sig_cycle = query_cycle_sigs[query_node_id];
            if (!data_sig_cycle.contains(query_sig_cycle)) { continue; }
            // ========== 新增代码 [结束] ==========


            // 如果所有检查都通过，则插入候选
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

  const size_t local_range_size = 256; // 调整为 256 以获得更好的占用率
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
             
             // 1. 读取 Candidates Block (Global Memory Read)
             types::candidates_t candidate_block = candidates_dev.candidates[current_idx];
             
             // 快速跳过
             if (candidate_block == 0) continue; 

             size_t query_node_id = current_idx / candidates_dev.single_node_size;
             size_t block_idx = current_idx % candidates_dev.single_node_size;

             // 2. 缓存 Query Signature 到私有寄存器 (Crucial Optimization)
             // 避免在循环中重复访问 Global Memory
             auto query_signature = query_signatures[query_node_id];

             types::candidates_t new_block = candidate_block;
             types::candidates_t bits_to_check = candidate_block;

             // 3. CTZ Loop
             while (bits_to_check != 0) {
               uint32_t bit = sycl::ctz(bits_to_check);
               
               size_t data_node_id = block_idx * candidates_dev.num_bits + bit;

               // 边界检查优化：利用 likely/unlikely 提示编译器 (如果支持)
               if (data_node_id < total_data_nodes) {
                   // 获取 Data Signature (Global Random Read - Bottleneck)
                   // 无法完全优化，但减少寄存器压力有帮助
                   auto data_signature = data_signatures[data_node_id];

                   bool keep = true;
                   // 展开循环比较
                   #pragma unroll
                   for (types::label_t l = 0; l < sigmo::signature::Signature<>::SignatureDevice::getMaxLabels(); l++) {
                        // 使用位运算代替逻辑运算以减少分支
                        if (query_signature.getLabelCount(l) > data_signature.getLabelCount(l)) {
                            keep = false;
                            break;
                        }
                   }

                   if (!keep) {
                     new_block &= ~(static_cast<types::candidates_t>(1) << bit);
                   }
               } else {
                   // 越界位直接清除（通常不会发生，除非 padding）
                   new_block &= ~(static_cast<types::candidates_t>(1) << bit);
               }

               // 清除最低位
               bits_to_check &= (bits_to_check - 1);
             }

             // 4. 写回结果
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
// ... (join 命名空间无变化) ...
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

/**
 * L4 优化: 内联 isValidMapping，使用私有寄存器缓存查询图。
 *
 * 这个函数被内联到 joinCandidates3 内核中，因为它依赖于
 * work-item 的私有寄存器数组 query_adj_matrix。
 */
constexpr size_t MAX_QUERY_NODES = 30; 
// 辅助函数：快速验证映射 (利用私有寄存器缓存的 Query 邻接矩阵)
// 请将此函数放在 isomorphism.hpp 中 joinCandidates3 之前
SYCL_EXTERNAL bool isValidMapping_fast(types::node_t candidate,
                                     uint depth,
                                     const types::node_t* mapping, 
                                     const bool query_adj_matrix[][MAX_QUERY_NODES], 
                                     const sigmo::DeviceBatchedCSRGraph& data_graphs,
                                     uint data_graph_id) {
  for (int i = 0; i < depth; i++) {
    // L4 优化: 从寄存器直接读取，零延迟
    bool isQueryNeighbor = query_adj_matrix[i][depth]; 

    // 唯一的全局内存访问
    if (isQueryNeighbor != data_graphs.isNeighbor(mapping[i], candidate)) {
      return false;
    }
  }
  return true;
}
// 优化后的辅助函数：使用位运算进行快速邻接检查
// 将此函数放在 joinCandidates3 之前
SYCL_EXTERNAL bool isValidMapping_fast_bitwise(
    types::node_t candidate,
    uint depth,
    const types::node_t* mapping, 
    const uint32_t* query_adj_masks, // 修改：传入位掩码数组
    const sigmo::DeviceBatchedCSRGraph& data_graphs,
    uint data_graph_id) 
{
  // 获取当前 depth 节点对应的邻接位掩码
  uint32_t current_adj_mask = query_adj_masks[depth];

  for (int i = 0; i < depth; i++) {
    // 优化：通过位运算检查 i 是否是 depth 的邻居 (1 bit vs 1 byte)
    bool isQueryNeighbor = (current_adj_mask >> i) & 1;

    // 全局内存访问：检查数据图
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

            // 构建位掩码邻接信息 & 候选数缓存
            for(uint16_t i = 0; i < num_query_nodes; ++i) {
                uint32_t count = candidates.getCandidatesCount(
                    i + offset_query_nodes, start_data_graph, end_data_graph);
                
                candidate_counts[i] = count;

                // ========== 优化代码 [开始] ==========
                // 如果任意一个查询节点没有候选，则整个子图匹配不可能成功
                if (count == 0) {
                    possible = false;
                    break; // 提前退出预处理循环
                }
                // ========== 优化代码 [结束] ==========

                uint32_t mask = 0;
                for(uint16_t j = 0; j < i; ++j) {
                    if (query_graphs.isNeighbor(offset_query_nodes + i,
                                                offset_query_nodes + j)) {
                        mask |= (1u << j);
                    }
                }
                query_adj_masks[i] = mask;
            }

            // ========== 优化代码 [开始] ==========
            // 如果标记为不可能，直接跳过 DFS 阶段，处理下一个任务
            if (!possible) continue;
            // ========== 优化代码 [结束] ==========

            const uint32_t root_candidates_count = candidate_counts[0];
            size_t current_task_matches = 0;
            
            // Sub-Group 内并行扩展多个根候选
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


// ... (后文代码) ...
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

                // 全局原子计数器
                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    global_match_counter{num_partial_matches[0]};

                // grid-stride：每个work-group处理多个data_graph
                for (uint32_t data_graph_id = gid; data_graph_id < total_data_graphs; data_graph_id += gcount) {
                    const uint32_t qid = 0; // 只用第0个查询图作为公共前缀
                    const uint32_t qnode_off = query_graphs.getPreviousNodes(qid);
                    const uint16_t qnodes    = query_graphs.getGraphNodes(qid);
                    if (end > qnodes) continue;

                    const uint32_t dg_beg = data_graphs.graph_offsets[data_graph_id];
                    const uint32_t dg_end = data_graphs.graph_offsets[data_graph_id + 1];

                    // 第0个查询节点的候选数量
                    const uint32_t cand0_count = candidates.getCandidatesCount(qnode_off + 0, dg_beg, dg_end);

                    // 线程级并行：每个线程处理不同的首层候选（带步长）
                    for (uint32_t c0 = lid; c0 < cand0_count; c0 += lsize) {
                        // 线程本地数据结构
                        types::node_t mapping[MAX_QUERY_NODES];
                        utils::detail::Bitset<uint64_t> visited{dg_beg};
                        struct Stack { uint depth; uint32_t cand_idx; };
                        Stack stack[MAX_QUERY_NODES];

                        // 取该线程的首候选并进行有效性检查
                        const auto v0 = candidates.getCandidateAt(qnode_off + 0, c0, dg_beg, dg_end);
                        mapping[0] = v0;
                        visited.clear();
                        visited.set(v0);

                        // 若需要，可在此加入首节点候选的快速剪枝（度、标签、环信息等），示例：
                        // if (!fastFeasible(0, v0, query_graphs, qid, data_graphs, data_graph_id)) { visited.clear(); continue; }

                        // 初始化DFS，从depth=1开始
                        uint top = 0;
                        stack[top++] = {1u, 0u};

                        while (top > 0) {
                            auto &frame = stack[top - 1];
                            const uint qdepth = frame.depth;

                            // 达到部分匹配深度
                            if (qdepth == end) {
                                // 直接全局原子写出。若匹配极多，可改成组内批量保留以进一步降原子竞争。
                                const size_t idx = global_match_counter.fetch_add(1);
                                const size_t base = idx * (end + 1);
                                // 写入前end个映射
                                for (uint32_t i = 0; i < end; ++i) {
                                    partial_matches[base + i] = mapping[i];
                                }
                                // 末位写入data_graph_id
                                partial_matches[base + end] = static_cast<int>(data_graph_id);

                                // 回溯
                                top--;
                                continue;
                            }

                            // 当前查询节点的候选空间
                            const uint32_t qnode = qdepth; // 按编号顺序扩展；如需启发式可换为重编号
                            const uint32_t cand_cnt = candidates.getCandidatesCount(qnode_off + qnode, dg_beg, dg_end);

                            // 候选遍历结束，回溯
                            if (frame.cand_idx >= cand_cnt) {
                                top--;
                                // 释放上一步的占用
                                if (qdepth > 0) {
                                    visited.unset(mapping[qdepth]);
                                }
                                continue;
                            }

                            // 取下一个候选
                            const auto dv = candidates.getCandidateAt(qnode_off + qnode, frame.cand_idx, dg_beg, dg_end);
                            frame.cand_idx++;

                            // 去重
                            if (visited.get(dv)) continue;

                            // 结构一致性校验
                            if (qdepth == 0 || isValidMapping(dv, qdepth, mapping, query_graphs, qid, data_graphs, data_graph_id)) {
                                mapping[qdepth] = dv;
                                visited.set(dv);
                                stack[top++] = {qdepth + 1, 0u};
                            }
                        } // while DFS

                        // 清理
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
        return e; // 如果没有部分匹配，直接返回
    }

    const size_t preferred_workgroup_size = device::deviceOptions.join_work_group_size;
    
    // 全局大小根据部分匹配的数量来确定
    size_t global_size = ((num_partial_matches + preferred_workgroup_size - 1) / preferred_workgroup_size) * preferred_workgroup_size;
    sycl::nd_range<1> nd_range{num_partial_matches * preferred_workgroup_size, preferred_workgroup_size};

    constexpr size_t MAX_QUERY_NODES = 30;

    auto e1 = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<device::kernels::JoinWithPartialMatchesKernel>( // 使用新的Kernel名称
            nd_range,
            [=,query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice(), gmcr = gmcr.getGMCRDevice()](sycl::nd_item<1> item) {
                const auto wg = item.get_group();
                const size_t wgid = wg.get_group_linear_id();   // 工作组ID，直接对应一个部分匹配
                const size_t wglid = wg.get_local_linear_id();  // 工作组内的线程ID
                const size_t wgsize = wg.get_local_range()[0];

                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> total_matches_ref{total_full_matches[0]};

                size_t private_num_matches = 0;

                // Grid-stride loop: 每个工作组处理一个部分匹配
                for (size_t partial_match_idx = wgid; partial_match_idx < num_partial_matches; partial_match_idx += wg.get_group_linear_range()) {
                    
                    // 1. 从全局内存加载部分匹配数据
                    types::node_t mapping[MAX_QUERY_NODES];
                    const size_t partial_match_size = end + 1;
                    const size_t base_offset = partial_match_idx * partial_match_size;
                    
                    for (size_t i = 0; i < end; ++i) {
                        mapping[i] = partial_matches[base_offset + i];
                    }
                    const uint32_t data_graph_id = partial_matches[base_offset + end];
                    
                    const uint32_t start_data_graph = data_graphs.graph_offsets[data_graph_id];
                    const uint32_t end_data_graph = data_graphs.graph_offsets[data_graph_id + 1];

                    // 2. 利用GMCR筛选需要处理的查询图
                    const uint32_t start_query = gmcr.data_graph_offsets[data_graph_id];
                    const uint32_t end_query = gmcr.data_graph_offsets[data_graph_id + 1];

                    // 3. 工作组内的线程并行处理不同的查询图
                    for (uint32_t query_graph_it = wglid; query_graph_it < (end_query - start_query); query_graph_it += wgsize) {
                        const uint32_t query_graph_id = gmcr.query_graph_indices[start_query + query_graph_it];
                        
                        const uint32_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
                        const uint16_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);

                        // 如果部分匹配的深度已经等于或超过当前查询图的节点数，说明它本身就是一个完整匹配
                        if (end >= num_query_nodes) {
                            if (end == num_query_nodes) {
                                private_num_matches++;
                            }
                            continue;
                        }

                        // 4. 初始化DFS状态以继续搜索
                        Stack stack[MAX_QUERY_NODES];
                        utils::detail::Bitset<uint64_t> visited{start_data_graph};
                        // 将部分匹配中的节点标记为已访问
                        for (size_t i = 0; i < end; ++i) {
                            visited.set(mapping[i]);
                        }

                        uint top = 0;
                        // 将栈的初始状态设置为从第 `end` 个节点开始搜索
                        stack[top++] = { (uint16_t)end, 0 };

                        // 5. 继续DFS循环
                        while (top > 0) {
                            auto frame = stack[top - 1];
                            auto query_node = frame.depth;

                            // 找到一个完整的匹配
                            if (frame.depth == num_query_nodes) {
                                private_num_matches++;
                                top--;
                                continue;
                            }

                            // 检查当前深度的候选集是否已用尽
                            if (frame.candidateIdx >= candidates.getCandidatesCount(query_node + offset_query_nodes, start_data_graph, end_data_graph)) {
                                top--;
                                if (top > 0 && frame.depth > end) { // 只有在搜索新节点时才需要释放
                                     visited.unset(mapping[query_node]);
                                }
                                continue;
                            }
                            
                            auto candidate = candidates.getCandidateAt(query_node + offset_query_nodes, frame.candidateIdx, start_data_graph, end_data_graph);
                            stack[top - 1].candidateIdx++;

                            if (visited.get(candidate)) {
                                continue;
                            }
                            
                            // 检查新节点的映射是否有效
                            if (isValidMapping(candidate, frame.depth, mapping, query_graphs, query_graph_id, data_graphs, data_graph_id)) {
                                mapping[frame.depth] = candidate;
                                visited.set(candidate);
                                stack[top++] = { (uint16_t)(frame.depth + 1), 0 };
                            }
                        }
                    }
                }

                // 6. 统计结果
                // 首先，在工作组内进行归约（Reduction），将所有线程的计数值相加
                private_num_matches = sycl::reduce_over_group(wg, private_num_matches, sycl::plus<>());
                
                // 然后，由leader线程将工作组的总数原子地加到全局计数器上
                if (wg.leader()) {
                    total_matches_ref += private_num_matches;
                }
            });
    });

    e.add(e1);
    return e;
}
// 新增重载：带 find_first / find_all 与去重标记数组
// 确保在调用此函数前，isValidMapping_fast_bitwise 已经在 isomorphism.hpp 中定义
// (它应该在前面的代码块中，通常位于 joinCandidates3 附近)

template<int SG_SIZE = 32> // 默认子组大小32
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
    
    // 计算并行规模：基于 Sub-Group 的数量来规划
    // 策略：一个 Sub-Group (32线程) 协作处理一个 Partial Match 任务
    const size_t num_subgroups_per_wg = preferred_workgroup_size / SG_SIZE;
    // 向上取整计算需要的 WG 数量
    const size_t num_wgs = (num_partial_matches + num_subgroups_per_wg - 1) / num_subgroups_per_wg;
    
    sycl::nd_range<1> nd_range{num_wgs * preferred_workgroup_size, preferred_workgroup_size};

    auto ev = queue.submit([&](sycl::handler& cgh) {
        // 使用 SubGroupOptimized Kernel 名称
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
                const size_t sglid = sg.get_local_linear_id(); // 子组内 Lane ID (0-31)
                
                // 计算当前 Sub-Group 在 WG 内的索引
                const size_t num_sgs_in_wg = wg.get_local_range()[0] / SG_SIZE;
                const size_t sg_in_wg_id = item.get_local_linear_id() / SG_SIZE;
                
                // 计算全局任务 ID (Partial Match Index)
                const size_t task_id = wgid * num_sgs_in_wg + sg_in_wg_id;

                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    total_matches_ref{total_full_matches[0]};

                // 边界检查：如果超出任务数，直接返回
                if (task_id >= num_partial_matches) return;

                const uint32_t num_data_graphs = data_graphs.num_graphs;
                
                // ==========================
                // 1. 还原部分匹配 (Sub-Group 共享)
                // ==========================
                // Mapping 数据很小，让每个线程都持有一份拷贝到寄存器
                types::node_t mapping[MAX_QUERY_NODES];
                const size_t rec_size = end + 1;
                const size_t base_offset = task_id * rec_size;
                
                // 读取数据图 ID
                const uint32_t data_graph_id = static_cast<uint32_t>(partial_matches[base_offset + end]);
                
                // 协同读取 Mapping：每个线程读取一部分，然后广播？
                // 由于 end 很小 (通常<30)，直接每个线程读取所有可能更快 (避免同步开销)
                // 或者简单的循环读取
                for (size_t i = 0; i < end; ++i) {
                    mapping[i] = partial_matches[base_offset + i];
                }

                const uint32_t dg_beg = data_graphs.graph_offsets[data_graph_id];
                const uint32_t dg_end = data_graphs.graph_offsets[data_graph_id + 1];
                const uint32_t qbeg = gmcr.data_graph_offsets[data_graph_id];
                const uint32_t qend = gmcr.data_graph_offsets[data_graph_id + 1];

                // 私有寄存器缓存：用于加速
                uint32_t query_adj_masks[MAX_QUERY_NODES];
                uint32_t candidate_counts[MAX_QUERY_NODES];
                
                size_t private_count = 0;

                // ==========================
                // 2. 遍历查询图
                // ==========================
                for (uint32_t it = 0; it < (qend - qbeg); ++it) {
                    const uint32_t query_graph_id = gmcr.query_graph_indices[qbeg + it];
                    
                    // 标记检查：针对 (Query, Data) 对的去重
                    const uint32_t flag_idx = query_graph_id * num_data_graphs + data_graph_id;
                    auto flag_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(pair_done[flag_idx]);

                    // 如果是 find_first 且已完成，所有线程跳过
                    if (find_first && flag_ref.load() != 0u) continue;

                    const uint32_t qoff = query_graphs.getPreviousNodes(query_graph_id);
                    const uint16_t qn = query_graphs.getGraphNodes(query_graph_id);

                    // 如果部分匹配已经是完整匹配
                    if (end >= qn) {
                         // 只有 Lane 0 执行一次计数
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

                    // ==========================
                    // 3. 预计算 & 缓存
                    // ==========================
                    // 填充寄存器缓存：避免在深层循环中访问全局内存
                    for (uint16_t i = 0; i < qn; ++i) {
                        candidate_counts[i] = candidates.getCandidatesCount(qoff + i, dg_beg, dg_end);
                        
                        // 构建位掩码邻接矩阵 (仅编码 j < i 的前驱边)
                        uint32_t mask = 0;
                        for (uint16_t j = 0; j < i; ++j) {
                            if (query_graphs.isNeighbor(qoff + i, qoff + j)) mask |= (1u << j);
                        }
                        query_adj_masks[i] = mask;
                    }

                    // ==========================
                    // 4. Sub-Group 并行扩展 (核心优化)
                    // ==========================
                    // 我们位于深度 `end`。该节点的候选集大小为 `candidate_counts[end]`。
                    // 这里的关键是：Sub-Group 的 32 个线程 并行探索这一层的不同候选。
                    
                    const uint32_t current_candidates_count = candidate_counts[end];
                    size_t thread_matches = 0;

                    // 步长循环：线程 k 处理候选 k, k+32, k+64...
                    for (uint32_t c_idx = sglid; c_idx < current_candidates_count; c_idx += SG_SIZE) {
                        
                        // 获取分配给当前线程的候选 (作为第 end 层的节点)
                        auto branch_root_cand = candidates.getCandidateAt(qoff + end, c_idx, dg_beg, dg_end);

                        // 快速剪枝：检查是否与已有的 partial match 重复
                        bool conflict = false;
                        for(size_t i=0; i<end; ++i) {
                            if (mapping[i] == branch_root_cand) { conflict = true; break; }
                        }
                        if (conflict) continue;

                        // 快速剪枝：位运算拓扑检查 (针对 end 层)
                        if (!isValidMapping_fast_bitwise(branch_root_cand, end, mapping, query_adj_masks, data_graphs, data_graph_id)) {
                            continue;
                        }

                        // 初始化私有栈，准备从 end + 1 开始 DFS
                        Stack stack[MAX_QUERY_NODES];
                        utils::detail::Bitset<uint64_t> visited{dg_beg};
                        
                        // 构建 visited 集
                        for (size_t i = 0; i < end; ++i) visited.set(mapping[i]);
                        visited.set(branch_root_cand);
                        
                        // 构建当前线程的局部映射
                        types::node_t local_mapping[MAX_QUERY_NODES];
                        for(size_t i=0; i<end; ++i) local_mapping[i] = mapping[i];
                        local_mapping[end] = branch_root_cand;

                        uint top = 0;
                        stack[top++] = {static_cast<uint>(end + 1), 0};

                        // 线程私有 DFS (Depth-First Search)
                        while (top > 0) {
                            auto frame = stack[top - 1];
                            auto q_node = frame.depth;

                            // 找到完整匹配
                            if (frame.depth == qn) {
                                thread_matches++;
                                top--;
                                if (find_first) break; // 线程内优化：find_first 模式下找到一个即可停止当前分支
                                else continue;
                            }

                            // 候选耗尽
                            if (frame.candidateIdx >= candidate_counts[q_node]) {
                                top--;
                                visited.unset(local_mapping[q_node]);
                                continue;
                            }

                            // 获取下一个候选
                            auto cand = candidates.getCandidateAt(q_node + qoff, frame.candidateIdx, dg_beg, dg_end);
                            stack[top - 1].candidateIdx++;

                            if (visited.get(cand)) continue;

                            // 使用寄存器缓存的位掩码进行快速检查
                            if (isValidMapping_fast_bitwise(cand, frame.depth, local_mapping, query_adj_masks, data_graphs, data_graph_id)) {
                                local_mapping[frame.depth] = cand;
                                visited.set(cand);
                                stack[top++] = {frame.depth + 1, 0};
                            }
                        } // End While DFS
                        
                        if (find_first && thread_matches > 0) break;

                    } // End Parallel Expansion For-Loop

                    // ==========================
                    // 5. 结果归约 (Sub-Group Reduction)
                    // ==========================
                    if (find_first) {
                        // 只要 Sub-Group 中任意线程找到了
                        bool any_found = sycl::any_of_group(sg, thread_matches > 0);
                        if (any_found) {
                            // 只有 Lane 0 尝试设置全局 Flag 并计数
                            if (sglid == 0) {
                                uint32_t old = flag_ref.exchange(1u);
                                if (old == 0u) private_count++;
                            }
                        }
                    } else {
                        // Find All: 累加所有线程的结果
                        size_t total_sg_matches = sycl::reduce_over_group(sg, thread_matches, sycl::plus<>());
                        if (sglid == 0) private_count += total_sg_matches;
                    }

                } // End Query Graph Loop

                // 最终写回全局内存 (每个任务只有 Lane 0 写回)
                if (sglid == 0 && private_count > 0) {
                    total_matches_ref += private_count;
                }
            });
    });

    e.add(ev);
    return e;
}
template<size_t BUFFER_SIZE_INTS = 64> // 默认缓存64个整数 (约256字节)
struct LocalOutputBuffer {
    int buffer[BUFFER_SIZE_INTS];
    int current_idx = 0;
    
    // 检查是否已满
    bool is_full(size_t match_len) const {
        return (current_idx + match_len + 1) > BUFFER_SIZE_INTS;
    }

    // 添加到本地缓存
    void append(const types::node_t* mapping, size_t end, int data_graph_id) {
        for (size_t i = 0; i < end; ++i) {
            buffer[current_idx++] = mapping[i];
        }
        buffer[current_idx++] = data_graph_id;
    }

    // 刷新到全局内存
    void flush(sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>& global_counter,
               int* global_output, 
               size_t end) {
        if (current_idx == 0) return;

        // 计算本次有多少个匹配
        size_t match_size = end + 1;
        size_t num_matches_in_buffer = current_idx / match_size;

        // 1. 执行一次原子加法，申请所有空间
        size_t global_start_idx = global_counter.fetch_add(num_matches_in_buffer);
        
        // 2. 批量写入全局内存 (Coalesced write 尽可能)
        size_t global_offset = global_start_idx * match_size;
        
        for (int i = 0; i < current_idx; ++i) {
            global_output[global_offset + i] = buffer[i];
        }

        // 重置缓冲区
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

    // L1: 任务分块
    constexpr size_t TASKS_PER_WORKGROUP = 256; 
    const size_t num_workgroups = (total_data_graphs + TASKS_PER_WORKGROUP - 1) / TASKS_PER_WORKGROUP;
    const size_t global_size = num_workgroups * preferred_workgroup_size;

    sycl::nd_range<1> nd_range{global_size, preferred_workgroup_size};

    auto e1 = queue.submit([&](sycl::handler& cgh) {
        // L2: 动态任务索引
        sycl::local_accessor<uint32_t, 1> shared_task_idx_accessor(sycl::range<1>(1), cgh);

        cgh.parallel_for<class JoinPartialCandidates2HighlyOptimized>( 
            nd_range,
            [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()]
            (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                
                const auto wg = item.get_group();
                const auto sg = item.get_sub_group();
                const size_t sglid = sg.get_local_linear_id();
                const size_t sg_size = sg.get_local_range()[0];

                // 全局原子计数器
                sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    global_match_counter{num_partial_matches[0]};

                // L6: 定义本地输出缓冲区 (减少原子争用)
                // 大小设置为 64 ints，足以容纳约 10-15 个 partial matches (视 end 大小而定)
                LocalOutputBuffer<64> output_aggregator;

                // L4: 位掩码优化
                uint32_t query_adj_masks[MAX_QUERY_NODES];
                // L3: 计数缓存
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
                    // L2: Sub-Group 级任务窃取
                    if (sg.leader()) { my_task_idx_in_chunk = atomic_task_idx.fetch_add(1); }
                    my_task_idx_in_chunk = sycl::select_from_group(sg, my_task_idx_in_chunk, 0);

                    if (my_task_idx_in_chunk >= num_tasks_in_chunk) break;

                    const uint32_t data_graph_id = task_chunk_start + my_task_idx_in_chunk;

                    // Query 0 上下文
                    const uint32_t qid = 0; 
                    const uint32_t qnode_off = query_graphs.getPreviousNodes(qid);
                    const uint16_t qnodes = query_graphs.getGraphNodes(qid);
                    
                    if (end > qnodes) continue;

                    const uint32_t dg_beg = data_graphs.graph_offsets[data_graph_id];
                    const uint32_t dg_end = data_graphs.graph_offsets[data_graph_id + 1];

                    // 预处理 Query 信息
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

                    // L5: Sub-Group 并行扩展
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

                             // -------------------------------------------------
                             // L6: 缓冲区写入逻辑
                             // -------------------------------------------------
                             if (qdepth == end) {
                                 // 检查缓冲区是否已满，若满则先 flush
                                 if (output_aggregator.is_full(end)) {
                                     output_aggregator.flush(global_match_counter, partial_matches, end);
                                 }
                                 
                                 // 添加到本地缓存
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

                // 任务结束后，刷新剩余缓冲区
                output_aggregator.flush(global_match_counter, partial_matches, end);
            });
    });

    e.add(e1);
    return e;
}

} // namespace join
} // namespace isomorphism
} // namespace sigmo
