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
namespace mapping {

class GMCR {
private:
  struct GMCRDevice {
    uint32_t* data_graph_offsets;
    uint32_t* query_graph_indices;
    size_t total_query_indices;
  } gmcr;
  sycl::queue& queue;

public:
  GMCR(sycl::queue& queue) : queue(queue) {}
  ~GMCR() {
    sycl::free(gmcr.data_graph_offsets, queue);
    sycl::free(gmcr.query_graph_indices, queue);
  }

  // Offloaded version of generateGMCR using SYCL kernels.
  // Optimized: Uses a validity mask to avoid re-computing candidate checks in the second pass.
  utils::BatchedEvent
  generateGMCR(sigmo::DeviceBatchedCSRGraph& query_graphs, sigmo::DeviceBatchedCSRGraph& data_graphs, sigmo::candidates::Candidates& candidates) {
    // Get dimensions
    const size_t total_query_graphs = query_graphs.num_graphs;
    const size_t total_data_graphs = data_graphs.num_graphs;

    // Allocate device memory for data_graph_offsets (size = total_data_graphs+1)
    uint32_t* d_data_graph_offsets = device::memory::malloc<uint32_t>(total_data_graphs + 1, queue);
    // Initialize to zero
    queue.fill(d_data_graph_offsets, 0, total_data_graphs + 1).wait();

    // --- Optimization: Allocate temporary validity mask ---
    // Stores the result of the expensive check (Query x Data)
    // Size: total_query_graphs * total_data_graphs
    size_t total_pairs = total_query_graphs * total_data_graphs;
    uint8_t* d_validity_mask = device::memory::malloc<uint8_t>(total_pairs, queue);

    // --- Kernel 1: Check validity and Count ---
    // For each pair (query_graph, data_graph), check if valid.
    // Store result in d_validity_mask AND atomically increment counter.
    auto k1 = queue.parallel_for(
        sycl::range<2>(total_query_graphs, total_data_graphs),
        [=, query_graphs = query_graphs, data_graphs = data_graphs, candidates = candidates.getCandidatesDevice()](sycl::item<2> item) {
          size_t query_graph_id = item.get_id(0);
          size_t data_graph_id = item.get_id(1);
          
          size_t num_query_nodes = query_graphs.getGraphNodes(query_graph_id);
          bool is_valid = false;

          // Only process if query graph makes sense (>1 node usually, logic kept from original)
          if (num_query_nodes > 1) { 
            size_t start_data = data_graphs.graph_offsets[data_graph_id];
            size_t end_data = data_graphs.graph_offsets[data_graph_id + 1];
            size_t offset_query_nodes = query_graphs.getPreviousNodes(query_graph_id);
            
            bool add = true;
            // Expensive loop: checks global memory candidates
            for (size_t i = 0; i < num_query_nodes && add; i++) {
              size_t global_query_node = offset_query_nodes + i;
              add = add && (candidates.getCandidatesCount(global_query_node, start_data, end_data) > 0);
            }
            is_valid = add;
          }

          // Store validity for the second pass (Critical Optimization)
          size_t linear_idx = query_graph_id * total_data_graphs + data_graph_id;
          d_validity_mask[linear_idx] = is_valid ? 1 : 0;

          if (is_valid) {
            // Increment the counter for this data graph.
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_offset(
                d_data_graph_offsets[data_graph_id + 1]);
            atomic_offset.fetch_add(1);
          }
        });
    k1.wait();

    // --- Kernel 2: Compute prefix sum of data_graph_offsets ---
    auto k2 = queue.single_task([=]() {
      for (size_t i = 1; i <= total_data_graphs; i++) { d_data_graph_offsets[i] += d_data_graph_offsets[i - 1]; }
    });
    k2.wait();

    // The total number of query indices is now the last element in d_data_graph_offsets.
    uint32_t total_query_indices;
    queue.copy(&d_data_graph_offsets[total_data_graphs], &total_query_indices, 1).wait();
    gmcr.total_query_indices = total_query_indices;

    // Allocate device memory for query_graph_indices.
    uint32_t* d_query_graph_indices = device::memory::malloc<uint32_t>(total_query_indices, queue);

    // Create a copy of the prefix sum array to serve as atomic “current offsets”
    uint32_t* current_offsets = device::memory::malloc<uint32_t>(total_data_graphs + 1, queue);
    queue.copy(d_data_graph_offsets, current_offsets, total_data_graphs + 1).wait();

    // --- Kernel 3: Fill query_graph_indices (Optimized) ---
    // Instead of re-computing validity, read d_validity_mask.
    auto k3 = queue.parallel_for(
        sycl::range<2>(total_query_graphs, total_data_graphs),
        [=](sycl::item<2> item) {
          size_t query_graph_id = item.get_id(0);
          size_t data_graph_id = item.get_id(1);

          size_t linear_idx = query_graph_id * total_data_graphs + data_graph_id;
          
          // Fast check: read from cached validity mask
          if (d_validity_mask[linear_idx] == 1) {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_off(
                current_offsets[data_graph_id]);
            uint32_t index = atomic_off.fetch_add(1);
            d_query_graph_indices[index] = static_cast<uint32_t>(query_graph_id);
          }
        });
    k3.wait();

    // Cleanup temporary memory
    sycl::free(d_validity_mask, queue);
    sycl::free(current_offsets, queue);

    // Build the result structure.
    gmcr.data_graph_offsets = d_data_graph_offsets;
    gmcr.query_graph_indices = d_query_graph_indices;

    utils::BatchedEvent ret;
    ret.add(k1);
    ret.add(k2);
    ret.add(k3);
    return ret;
  }

  GMCRDevice getGMCRDevice() { return gmcr; }
};


} // namespace mapping
} // namespace isomorphism
} // namespace sigmo