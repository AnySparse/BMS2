/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "candidates.hpp"
#include "device.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "graph.hpp" // <--- 新增
#include <cstdint>
#include <sycl/sycl.hpp>

namespace sigmo {
namespace signature {

enum class Algorithm { ViewBased, PowerGraph };
enum class SignatureScope { Data, Query };

template<Algorithm A = Algorithm::PowerGraph, size_t Bits = 4>
class Signature {
// ... (此类无变化) ...
public:
  struct SignatureDevice {
    uint64_t signature;

    SignatureDevice() : signature(0) {}

    SignatureDevice(uint64_t signature) : signature(signature) {}

    SYCL_EXTERNAL static uint16_t getMaxLabels() { return sizeof(signature) * 8 / Bits; }

    SYCL_EXTERNAL void setLabelCount(uint8_t label, uint8_t count) {
      if (label < (64 / Bits) && count < (1 << Bits)) {
        signature &= ~((static_cast<uint64_t>((1 << Bits) - 1)) << (label * Bits)); // Clear the bits for the label
        signature |= (static_cast<uint64_t>(count) << (label * Bits));              // Set the new count
      }
    }

    SYCL_EXTERNAL uint8_t getLabelCount(uint8_t label) const {
      if (label < (64 / Bits)) { return (signature >> (label * Bits)) & ((1 << Bits) - 1); }
      return 0;
    }

    SYCL_EXTERNAL void incrementLabelCount(uint8_t label, uint8_t add = 1) {
      if (label < (64 / Bits)) {
        uint8_t count = getLabelCount(label);
        if (count < ((1 << Bits) - 1)) { // Ensure count does not exceed max value
          setLabelCount(label, count + static_cast<uint8_t>(add));
        }
      }
    }

    SYCL_EXTERNAL void clear() { signature = 0; }
  };

  template<typename T>
  utils::BatchedEvent generateQuerySignatures(T& graphs) {
    return generateSignatures(graphs, SignatureScope::Query);
  }

  template<typename T>
  utils::BatchedEvent refineQuerySignatures(T& graphs, size_t view_size = 1) {
    return refineSignatures(graphs, view_size, SignatureScope::Query);
  }

  template<typename T>
  utils::BatchedEvent generateDataSignatures(T& graphs) {
    return generateSignatures(graphs, SignatureScope::Data);
  }

  template<typename T>
  utils::BatchedEvent refineDataSignatures(T& graphs, size_t view_size = 1) {
    return refineSignatures(graphs, view_size, SignatureScope::Data);
  }

  template<typename T>
  utils::BatchedEvent generateSignatures(T& graphs, SignatureScope s) {
    if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedAMGraph>) {
      return generateAMSignatures(graphs, s);
    } else if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedCSRGraph>) {
      return generateCSRSignatures(graphs, s);
    } else {
      throw std::runtime_error("Unsupported graph type");
    }
  }

  template<typename T>
  utils::BatchedEvent refineSignatures(T& graphs, size_t view_size, SignatureScope s) {
    if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedAMGraph>) {
      return refineAMSignatures(graphs, view_size, s);
    } else if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedCSRGraph>) {
      return refineCSRSignatures(graphs, view_size, s);
    } else {
      throw std::runtime_error("Unsupported graph type");
    }
  }


  // TODO consider to use the shared memory to store the graph to avoid uncoallesced memory access
  utils::BatchedEvent generateAMSignatures(DeviceBatchedAMGraph& graphs, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    SignatureDevice* signatures = s == SignatureScope::Data ? data_signatures : query_signatures;
    auto e = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<sigmo::device::kernels::GenerateQuerySignaturesKernel>(
          sycl::range<1>{graphs.total_nodes}, [=, graphs = graphs](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            // Get the neighbors of the current node
            types::node_t neighbors[types::MAX_NEIGHBORS];
            graphs.getNeighbors(node_id, neighbors);
            for (types::node_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
              auto neighbor = neighbors[i];
              signatures[node_id].incrementLabelCount(graphs.node_labels[neighbor]);
            }
          });
    });
    event.add(e);

    return event;
  }

  template<Algorithm _A = A>
  utils::BatchedEvent refineAMSignatures(DeviceBatchedAMGraph& graphs, size_t view_size, SignatureScope s);

  utils::BatchedEvent generateCSRSignatures(DeviceBatchedCSRGraph& graphs, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    SignatureDevice* signatures = s == SignatureScope::Data ? data_signatures : query_signatures;

    auto e = queue.submit([&](sycl::handler& cgh) {
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* node_labels = graphs.node_labels;


      cgh.parallel_for<sigmo::device::kernels::GenerateDataSignaturesKernel>(global_range, [=](sycl::item<1> item) {
        auto node_id = item.get_id(0);

        uint32_t start_neighbor = row_offsets[node_id];
        uint32_t end_neighbor = row_offsets[node_id + 1];
        sigmo::types::label_t node_label = node_labels[node_id];

        for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
          auto neighbor = column_indices[i];
          signatures[node_id].incrementLabelCount(node_labels[neighbor]);
        }
      });
    });
    event.add(e);

    return event;
  }

  template<Algorithm _A = A>
  utils::BatchedEvent refineCSRSignatures(DeviceBatchedCSRGraph& graphs, size_t view_size, SignatureScope s);

  Signature(sycl::queue& queue, size_t data_nodes, size_t query_nodes) : queue(queue), data_nodes(data_nodes), query_nodes(query_nodes) {
    data_signatures = device::memory::malloc<SignatureDevice>(data_nodes, queue);
    query_signatures = device::memory::malloc<SignatureDevice>(query_nodes, queue);
    if constexpr (A == Algorithm::ViewBased) {
      tmp_buff = device::memory::malloc<SignatureDevice>(std::max(data_nodes, query_nodes), queue);
    } else if constexpr (A == Algorithm::PowerGraph) {
      data_reachables = device::memory::malloc<utils::detail::Bitset<uint64_t>>(data_nodes, queue);
      query_reachables = device::memory::malloc<utils::detail::Bitset<uint64_t>>(query_nodes, queue);
    }
    queue.fill(data_signatures, 0, data_nodes).wait();
    queue.fill(query_signatures, 0, query_nodes).wait();
  }

  ~Signature() {
    sycl::free(data_signatures, queue);
    sycl::free(query_signatures, queue);
    if constexpr (A == Algorithm::ViewBased) {
      sycl::free(tmp_buff, queue);
    } else if constexpr (A == Algorithm::PowerGraph) {
      sycl::free(data_reachables, queue);
      sycl::free(query_reachables, queue);
    }
  }

  size_t getDataSignatureAllocationSize() const {
    size_t alloc = data_nodes * sizeof(SignatureDevice);
    if constexpr (A == Algorithm::PowerGraph) {
      alloc += data_nodes * sizeof(utils::detail::Bitset<uint64_t>);
    } else if constexpr (A == Algorithm::ViewBased) {
      alloc += data_nodes * sizeof(SignatureDevice);
    }
    return alloc;
  }
  size_t getQuerySignatureAllocationSize() const {
    size_t alloc = query_nodes * sizeof(SignatureDevice);
    if constexpr (A == Algorithm::PowerGraph) {
      alloc += query_nodes * sizeof(utils::detail::Bitset<uint64_t>);
    } else if constexpr (A == Algorithm::ViewBased) {
      alloc += query_nodes * sizeof(SignatureDevice);
    }
    return alloc;
  }
  SignatureDevice* getDeviceDataSignatures() const { return data_signatures; }
  SignatureDevice* getDeviceQuerySignatures() const { return query_signatures; }
  size_t getMaxLabels() const { return SignatureDevice::getMaxLabels(); }

private:
  sycl::queue& queue;
  size_t data_nodes;
  size_t query_nodes;
  SignatureDevice* data_signatures;
  SignatureDevice* query_signatures;
  SignatureDevice* tmp_buff;
  utils::detail::Bitset<uint64_t>* data_reachables;
  utils::detail::Bitset<uint64_t>* query_reachables;

  template<>
  utils::BatchedEvent refineAMSignatures<Algorithm::ViewBased>(DeviceBatchedAMGraph& graphs, size_t view_size, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);

    auto signatures = s == SignatureScope::Data ? data_signatures : query_signatures;
    auto copy_event = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=, tmp_buff = this->tmp_buff](sycl::item<1> item) { tmp_buff[item] = signatures[item]; });
    });
    event.add(copy_event);
    copy_event.wait();

    auto refinement_event = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(copy_event);
      const uint16_t max_labels_count = Signature::SignatureDevice::getMaxLabels();

      cgh.parallel_for<sigmo::device::kernels::RefineQuerySignaturesKernel>(
          sycl::range<1>{graphs.total_nodes}, [=, graphs = graphs, tmp_buff = this->tmp_buff](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            // Get the neighbors of the current node
            types::node_t neighbors[types::MAX_NEIGHBORS];
            types::label_t node_label = graphs.node_labels[node_id];
            graphs.getNeighbors(node_id, neighbors);
            for (types::node_t i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
              auto neighbor = neighbors[i];
              for (types::label_t l = 0; l < max_labels_count; l++) {
                auto count = tmp_buff[neighbor].getLabelCount(l);
                if (l == node_label) { count -= view_size; }
                if (count > 0) signatures[node_id].incrementLabelCount(l, count);
              }
            }
          });
    });
    event.add(refinement_event);
    return event;
  }

  template<>
  utils::BatchedEvent refineAMSignatures<Algorithm::PowerGraph>(DeviceBatchedAMGraph& graphs, size_t view_size, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    const uint16_t max_labels_count = Signature::SignatureDevice::getMaxLabels();
    auto signatures = s == SignatureScope::Data ? data_signatures : query_signatures;

    auto refinement_event = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<sigmo::device::kernels::RefineQuerySignaturesKernel>(
          sycl::range<1>{graphs.total_nodes}, [=, graphs = graphs, tmp_buff = this->tmp_buff](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            auto graph_id = graphs.getGraphId(node_id);
            auto prev_nodes = graphs.getPreviousNodes(graph_id);

            types::node_t neighbors[types::MAX_NEIGHBORS];
            utils::detail::Bitset<uint32_t> frontier, reachable;

            frontier.set(node_id - prev_nodes);
            reachable.set(node_id - prev_nodes);
            for (uint curr_iter = 0; curr_iter < view_size && !frontier.empty(); curr_iter++) {
              utils::detail::Bitset<uint32_t> next_frontier;

              for (uint idx = 0; idx < frontier.size(); idx++) {
                auto u = frontier.getSetBit(idx);
                graphs.getNeighbors(u + prev_nodes, neighbors, graph_id, prev_nodes);
                for (uint i = 0; neighbors[i] != types::NULL_NODE && i < types::MAX_NEIGHBORS; ++i) {
                  auto v = neighbors[i] - prev_nodes;
                  if (!reachable.get(v)) {
                    reachable.set(v);
                    next_frontier.set(v);
                  }
                }
              }
              frontier = next_frontier;
            }
            reachable.unset(node_id - prev_nodes);
            signatures[node_id].clear();
            for (uint idx = 0; idx < reachable.size(); idx++) {
              auto u = reachable.getSetBit(idx) + prev_nodes;
              types::label_t u_label = graphs.node_labels[u];
              signatures[node_id].incrementLabelCount(u_label);
            }
          });
    });
    event.add(refinement_event);
    return event;
  }

  template<>
  utils::BatchedEvent refineCSRSignatures<Algorithm::ViewBased>(DeviceBatchedCSRGraph& graphs, size_t view_size, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    auto signatures = s == SignatureScope::Data ? data_signatures : query_signatures;
    auto tmp_buff = this->tmp_buff;

    auto copy_event = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(graphs.total_nodes), [=](sycl::item<1> item) { tmp_buff[item] = signatures[item]; });
    });
    event.add(copy_event);
    auto refine_event = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(copy_event);
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* node_labels = graphs.node_labels;

      cgh.parallel_for<sigmo::device::kernels::RefineDataSignaturesKernel>(global_range, [=](sycl::item<1> item) {
        auto node_id = item.get_id(0);

        uint32_t start_neighbor = row_offsets[node_id];
        uint32_t end_neighbor = row_offsets[node_id + 1];
        sigmo::types::label_t node_label = node_labels[node_id];

        for (uint32_t i = start_neighbor; i < end_neighbor; ++i) {
          auto neighbor = column_indices[i];
          for (types::label_t l = 0; l < Signature::SignatureDevice::getMaxLabels(); l++) {
            auto count = tmp_buff[neighbor].getLabelCount(l);
            if (l == node_label) { count -= view_size; }
            if (count > 0) signatures[node_id].incrementLabelCount(l, count);
          }
        }
      });
    });
    event.add(refine_event);
    return event;
  }

  template<>
  utils::BatchedEvent refineCSRSignatures<Algorithm::PowerGraph>(DeviceBatchedCSRGraph& graphs, size_t view_size, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    auto signatures = s == SignatureScope::Data ? data_signatures : query_signatures;
    auto old_reachables = s == SignatureScope::Data ? data_reachables : query_reachables;

    auto refine_event = queue.submit([&](sycl::handler& cgh) {
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* node_labels = graphs.node_labels;

      cgh.parallel_for<sigmo::device::kernels::RefineDataSignaturesKernel>(
          global_range, [=, max_labels_count = Signature::SignatureDevice::getMaxLabels()](sycl::item<1> item) {
            auto node_id = item.get_id(0);
            auto graph_id = graphs.getGraphID(node_id);
            auto prev_nodes = graphs.getPreviousNodes(graph_id);
            utils::detail::Bitset<uint64_t> frontier, reachable;
            utils::detail::Bitset<uint64_t> old_reachable = old_reachables[node_id];


            frontier.set(node_id - prev_nodes);
            reachable.set(node_id - prev_nodes);
            for (uint curr_iter = 0; curr_iter < view_size && !frontier.empty(); curr_iter++) {
              utils::detail::Bitset<uint64_t> next_frontier;
              for (uint idx = 0; idx < frontier.size(); idx++) {
                auto u = frontier.getSetBit(idx) + prev_nodes;
                auto start_neighbor = row_offsets[u];
                auto end_neighbor = row_offsets[u + 1];
                for (auto i = start_neighbor; i < end_neighbor; ++i) {
                  auto neighbor = column_indices[i] - prev_nodes;
                  if (!reachable.get(neighbor)) {
                    reachable.set(neighbor);
                    next_frontier.set(neighbor);
                  }
                }
              }
              frontier = next_frontier;
            }
            reachable.unset(node_id - prev_nodes);
            reachable.difference(old_reachable);
            signatures[node_id].clear();
            for (uint idx = 0; idx < reachable.size(); idx++) {
              auto u = reachable.getSetBit(idx) + prev_nodes;
              types::label_t u_label = graphs.node_labels[u];
              // if (u_label == types::WILDCARD_NODE) { continue; }
              signatures[node_id].incrementLabelCount(u_label);
            }
            reachable.merge(old_reachable);
            old_reachables[node_id] = reachable;
          });
    });
    event.add(refine_event);
    return event;
  }
};

// ... (PathSignature class 无变化) ...
SYCL_EXTERNAL inline uint32_t hash32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

SYCL_EXTERNAL inline uint32_t combine_hash(uint32_t h1, uint32_t h2) {
  return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
}

class PathSignature {
public:
  constexpr static size_t N = 4; // N = number of uint64_t, so N*64 = 256 bits

  struct PathSignatureDevice {
    uint64_t bits[N];
    constexpr static size_t num_bits = N * 64;

    SYCL_EXTERNAL void clear() {
      for (size_t i = 0; i < N; ++i) bits[i] = 0;
    }

    SYCL_EXTERNAL void insert(uint32_t hash_val) {
      uint32_t h1 = hash_val;
      uint32_t h2 = hash32(h1);
      uint32_t h3 = hash32(h2);

      uint32_t idx1 = h1 % num_bits;
      uint32_t idx2 = h2 % num_bits;
      uint32_t idx3 = h3 % num_bits;

      bits[idx1 / 64] |= (1ULL << (idx1 % 64));
      bits[idx2 / 64] |= (1ULL << (idx2 % 64));
      bits[idx3 / 64] |= (1ULL << (idx3 % 64));
    }

    SYCL_EXTERNAL bool contains(const PathSignatureDevice& other) const {
      for (size_t i = 0; i < N; ++i) {
        if ((bits[i] & other.bits[i]) != other.bits[i]) { return false; }
      }
      return true;
    }
  };

  PathSignature(sycl::queue& queue, size_t data_nodes, size_t query_nodes)
      : queue(queue), data_nodes(data_nodes), query_nodes(query_nodes) {
    data_path_signatures = device::memory::malloc<PathSignatureDevice>(data_nodes, queue);
    query_path_signatures = device::memory::malloc<PathSignatureDevice>(query_nodes, queue);

    queue.fill(data_path_signatures, (uint64_t)0, data_nodes * N);
    queue.fill(query_path_signatures, (uint64_t)0, query_nodes * N).wait();
  }

  ~PathSignature() {
    sycl::free(data_path_signatures, queue);
    sycl::free(query_path_signatures, queue);
  }

  template<typename T>
  utils::BatchedEvent generateQueryPathSignatures(T& graphs) {
    if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedCSRGraph>) {
      return generateCSRPathSignatures(graphs, SignatureScope::Query);
    } else {
      throw std::runtime_error("Unsupported graph type for PathSignature");
    }
  }

  template<typename T>
  utils::BatchedEvent generateDataPathSignatures(T& graphs) {
    if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedCSRGraph>) {
      return generateCSRPathSignatures(graphs, SignatureScope::Data);
    } else {
      throw std::runtime_error("Unsupported graph type for PathSignature");
    }
  }

  utils::BatchedEvent generateCSRPathSignatures(DeviceBatchedCSRGraph& graphs, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    PathSignatureDevice* signatures = (s == SignatureScope::Data) ? data_path_signatures : query_path_signatures;

    auto e = queue.submit([&](sycl::handler& cgh) {
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* node_labels = graphs.node_labels;

      cgh.parallel_for<sigmo::device::kernels::GenerateCSRPathSignaturesKernel>(global_range, [=](sycl::item<1> item) {
        auto node_u = item.get_id(0);
        auto label_u = node_labels[node_u];

        uint32_t start_v = row_offsets[node_u];
        uint32_t end_v = row_offsets[node_u + 1];

        for (uint32_t i = start_v; i < end_v; ++i) {
          auto node_v = column_indices[i];
          auto label_v = node_labels[node_v];

          uint32_t start_w = row_offsets[node_v];
          uint32_t end_w = row_offsets[node_v + 1];
          for (uint32_t j = start_w; j < end_w; ++j) {
            auto node_w = column_indices[j];
            if (node_w == node_u) continue; 
            auto label_w = node_labels[node_w];

            uint32_t path_hash_l2 = combine_hash(label_u, combine_hash(label_v, label_w));
            signatures[node_u].insert(hash32(path_hash_l2));

            uint32_t start_x = row_offsets[node_w];
            uint32_t end_x = row_offsets[node_w + 1];
            for (uint32_t k = start_x; k < end_x; ++k) {
              auto node_x = column_indices[k];
              if (node_x == node_v || node_x == node_u) continue; 
              auto label_x = node_labels[node_x];

              uint32_t path_hash_l3 = combine_hash(path_hash_l2, label_x);
              signatures[node_u].insert(hash32(path_hash_l3));
            }
          }
        }
      });
    });
    event.add(e);
    return event;
  }

  size_t getDataSignatureAllocationSize() const { return data_nodes * sizeof(PathSignatureDevice); }
  size_t getQuerySignatureAllocationSize() const { return query_nodes * sizeof(PathSignatureDevice); }
  PathSignatureDevice* getDeviceDataSignatures() const { return data_path_signatures; }
  PathSignatureDevice* getDeviceQuerySignatures() const { return query_path_signatures; }

private:
  sycl::queue& queue;
  size_t data_nodes;
  size_t query_nodes;
  PathSignatureDevice* data_path_signatures;
  PathSignatureDevice* query_path_signatures;
};

// ========== 修改代码 [开始] ==========
class CycleSignature {
public:
  struct CycleSignatureDevice {
    uint16_t count_5_rings_same; // 标签全部相同的5元环
    uint16_t count_5_rings_diff; // 标签不全相同的5元环
    uint16_t count_6_rings_same; // 标签全部相同的6元环
    uint16_t count_6_rings_diff; // 标签不全相同的6元环

    SYCL_EXTERNAL void clear() {
      count_5_rings_same = 0;
      count_5_rings_diff = 0;
      count_6_rings_same = 0;
      count_6_rings_diff = 0;
    }

    // Check if this (data) contains other (query)
    SYCL_EXTERNAL bool contains(const CycleSignatureDevice& other) const {
      return (count_5_rings_same >= other.count_5_rings_same) && (count_5_rings_diff >= other.count_5_rings_diff)
             && (count_6_rings_same >= other.count_6_rings_same) && (count_6_rings_diff >= other.count_6_rings_diff);
    }
  };

  CycleSignature(sycl::queue& queue, size_t data_nodes, size_t query_nodes)
      : queue(queue), data_nodes(data_nodes), query_nodes(query_nodes) {

    data_cycle_signatures = device::memory::malloc<CycleSignatureDevice>(data_nodes, queue);
    query_cycle_signatures = device::memory::malloc<CycleSignatureDevice>(query_nodes, queue);

    // Zero-initialize
    queue.fill(data_cycle_signatures, CycleSignatureDevice{0, 0, 0, 0}, data_nodes);
    queue.fill(query_cycle_signatures, CycleSignatureDevice{0, 0, 0, 0}, query_nodes).wait();
  }
// ... (析构函数, generateQueryCycleSignatures, generateDataCycleSignatures 无变化) ...
  ~CycleSignature() {
    sycl::free(data_cycle_signatures, queue);
    sycl::free(query_cycle_signatures, queue);
  }

  template<typename T>
  utils::BatchedEvent generateQueryCycleSignatures(T& graphs) {
    if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedCSRGraph>) {
      return generateCSRCycleSignatures(graphs, SignatureScope::Query);
    } else {
      throw std::runtime_error("Unsupported graph type for CycleSignature");
    }
  }

  template<typename T>
  utils::BatchedEvent generateDataCycleSignatures(T& graphs) {
    if constexpr (std::is_same_v<std::decay_t<T>, DeviceBatchedCSRGraph>) {
      return generateCSRCycleSignatures(graphs, SignatureScope::Data);
    } else {
      throw std::runtime_error("Unsupported graph type for CycleSignature");
    }
  }

  utils::BatchedEvent generateCSRCycleSignatures(DeviceBatchedCSRGraph& graphs, SignatureScope s) {
    utils::BatchedEvent event;
    sycl::range<1> global_range(graphs.total_nodes);
    CycleSignatureDevice* signatures = (s == SignatureScope::Data) ? data_cycle_signatures : query_cycle_signatures;

    auto e = queue.submit([&](sycl::handler& cgh) {
      auto* row_offsets = graphs.row_offsets;
      auto* column_indices = graphs.column_indices;
      auto* graph_offsets = graphs.graph_offsets; // Need this
      auto* node_labels = graphs.node_labels;     // Need this

      cgh.parallel_for<sigmo::device::kernels::GenerateCSRCycleSignaturesKernel>(global_range, [=](sycl::item<1> item) {
        auto start_node_global = item.get_id(0);
        auto label_start_node = node_labels[start_node_global];

        // Find graph boundaries
        auto graph_id = graphs.getGraphID(start_node_global);
        auto graph_start_node = graphs.getPreviousNodes(graph_id);
        auto graph_num_nodes = graphs.getGraphNodes(graph_id);

        // ** CRITICAL ASSUMPTION **
        // This algorithm uses a uint64_t as a visited bitmask.
        // It will only work if graphs have <= 64 nodes.
        if (graph_num_nodes > 64) { return; }

        auto start_node_local = start_node_global - graph_start_node;

        uint16_t count_5_same = 0;
        uint16_t count_5_diff = 0;
        uint16_t count_6_same = 0;
        uint16_t count_6_diff = 0;

        // Stack for iterative DFS
        // {current_node_local, depth, all_labels_same_so_far}
        sycl::vec<uint32_t, 3> path_stack[128]; // Local stack, size 128
        uint64_t mask_stack[128];
        int stack_top = 0;

        // Push neighbors of start_node
        uint32_t start_v = row_offsets[start_node_global];
        uint32_t end_v = row_offsets[start_node_global + 1];

        for (uint32_t i = start_v; i < end_v; ++i) {
          auto neighbor_global = column_indices[i];
          auto neighbor_local = neighbor_global - graph_start_node;
          auto neighbor_label = node_labels[neighbor_global];

          if (stack_top < 128) {
            uint64_t visited_mask = (1ULL << start_node_local) | (1ULL << neighbor_local);
            uint32_t all_same = (neighbor_label == label_start_node) ? 1 : 0;
            path_stack[stack_top] = {neighbor_local, 1, all_same}; // depth 1
            mask_stack[stack_top] = visited_mask;
            stack_top++;
          }
        }

        while (stack_top > 0) {
          stack_top--;
          sycl::vec<uint32_t, 3> frame = path_stack[stack_top];
          uint64_t current_mask = mask_stack[stack_top];
          uint32_t current_node_local = frame[0];
          uint32_t current_depth = frame[1];
          uint32_t all_same_so_far = frame[2];
          auto current_node_global = current_node_local + graph_start_node;

          // Check for cycles
          // Path ends at depth 4 (5-cycle) or 5 (6-cycle)
          if (current_depth == 4) { // Path length 4
            uint32_t start_w = row_offsets[current_node_global];
            uint32_t end_w = row_offsets[current_node_global + 1];
            for (uint32_t i = start_w; i < end_w; ++i) {
              if (column_indices[i] == start_node_global) {
                if (all_same_so_far) {
                  count_5_same++;
                } else {
                  count_5_diff++;
                }
              }
            }
          } else if (current_depth == 5) { // Path length 5
            uint32_t start_w = row_offsets[current_node_global];
            uint32_t end_w = row_offsets[current_node_global + 1];
            for (uint32_t i = start_w; i < end_w; ++i) {
              if (column_indices[i] == start_node_global) {
                if (all_same_so_far) {
                  count_6_same++;
                } else {
                  count_6_diff++;
                }
              }
            }
          }

          // Continue traversal if not at max depth
          if (current_depth < 5) {
            uint32_t start_w = row_offsets[current_node_global];
            uint32_t end_w = row_offsets[current_node_global + 1];
            for (uint32_t i = start_w; i < end_w; ++i) {
              auto neighbor_global = column_indices[i];
              // Don't visit start_node (unless at end)
              if (neighbor_global == start_node_global) continue;

              auto neighbor_local = neighbor_global - graph_start_node;

              // If not visited
              if (!(current_mask & (1ULL << neighbor_local))) {
                if (stack_top < 128) {
                  auto neighbor_label = node_labels[neighbor_global];
                  uint32_t new_all_same = all_same_so_far && (neighbor_label == label_start_node);
                  uint64_t new_mask = current_mask | (1ULL << neighbor_local);
                  path_stack[stack_top] = {neighbor_local, current_depth + 1, new_all_same};
                  mask_stack[stack_top] = new_mask;
                  stack_top++;
                }
              }
            }
          }
        } // while stack not empty

        // Each cycle is counted twice (e.g., u->v..->u and u->..v->u)
        signatures[start_node_global].count_5_rings_same = count_5_same / 2;
        signatures[start_node_global].count_5_rings_diff = count_5_diff / 2;
        signatures[start_node_global].count_6_rings_same = count_6_same / 2;
        signatures[start_node_global].count_6_rings_diff = count_6_diff / 2;
      });
    });
    event.add(e);
    return event;
  }
// ... (getSize... functions 无变化) ...
  size_t getDataSignatureAllocationSize() const { return data_nodes * sizeof(CycleSignatureDevice); }
  size_t getQuerySignatureAllocationSize() const { return query_nodes * sizeof(CycleSignatureDevice); }
  CycleSignatureDevice* getDeviceDataSignatures() const { return data_cycle_signatures; }
  CycleSignatureDevice* getDeviceQuerySignatures() const { return query_cycle_signatures; }

private:
  sycl::queue& queue;
  size_t data_nodes;
  size_t query_nodes;
  CycleSignatureDevice* data_cycle_signatures;
  CycleSignatureDevice* query_cycle_signatures;
};
// ========== 修改代码 [结束] ==========

} // namespace signature
} // namespace sigmo
