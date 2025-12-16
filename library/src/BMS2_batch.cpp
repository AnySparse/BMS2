/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./utils.hpp"
#include <numeric>
#include <sigmo.hpp>
#include <sycl/sycl.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>


const size_t DATA_GRAPH_BATCH_SIZE = 100000; 

int main(int argc, char** argv) {
  Args args{argc, argv, sigmo::device::deviceOptions};


  sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
  size_t gpu_mem = queue.get_device().get_info<sycl::info::device::global_mem_size>();
  std::string gpu_name = queue.get_device().get_info<sycl::info::device::name>();


  std::vector<std::chrono::duration<double>> data_sig_times, query_sig_times, filter_times;
  std::vector<std::chrono::duration<double>> data_path_sig_times, query_path_sig_times, path_filter_times;
  std::vector<std::chrono::duration<double>> data_cycle_sig_times, query_cycle_sig_times, cycle_filter_times;
  

  std::chrono::duration<double> total_join_time_gpu{0};


  std::chrono::duration<double> total_host_setup_time{0};
  std::chrono::duration<double> total_host_sig_gen_time{0};
  std::chrono::duration<double> total_host_filter_time{0};
  std::chrono::duration<double> total_host_mapping_time{0};
  std::chrono::duration<double> total_host_join_time{0};
  std::chrono::duration<double> total_host_wall_time{0}; 

  auto global_start_clock = std::chrono::high_resolution_clock::now();


  size_t global_total_matches = 0;
  size_t total_data_graphs_processed = 0;
  
  size_t first_query_nodes = 0;

  sigmo::DeviceBatchedCSRGraph device_query_graph;
  std::vector<sigmo::CSRGraph> host_query_graphs;
  std::ifstream data_file_stream;

  if (args.query_data) {
    std::cout << "[Init] Loading Query graphs..." << std::endl;
    host_query_graphs = sigmo::io::loadCSRGraphsFromFile(args.query_file);


    if (args.query_filter.active) {
      for (int i = 0; i < (int)host_query_graphs.size(); ++i) {
        if (host_query_graphs[i].getNumNodes() > args.query_filter.max_nodes || 
            host_query_graphs[i].getNumNodes() < args.query_filter.min_nodes) {
          host_query_graphs.erase(host_query_graphs.begin() + i);
          i--;
        }
      }
    }
   
    if (!host_query_graphs.empty()) {
        first_query_nodes = host_query_graphs[0].getNumNodes();
    }


    size_t num_q_graphs = host_query_graphs.size();
    for (size_t i = 1; i < args.multiply_factor_query; ++i) {
      host_query_graphs.insert(host_query_graphs.end(), host_query_graphs.begin(), host_query_graphs.begin() + num_q_graphs);
    }

    if (host_query_graphs.size() > args.max_query_graphs) { 
        host_query_graphs.erase(host_query_graphs.begin() + args.max_query_graphs, host_query_graphs.end()); 
    }

    
    data_file_stream.open(args.data_file);
    if (!data_file_stream.is_open()) throw std::runtime_error("Could not open data file");

    device_query_graph = sigmo::createDeviceCSRGraph(queue, host_query_graphs);
  } else {
    throw std::runtime_error("Specify input data");
  }

  size_t num_query_graphs = device_query_graph.num_graphs;
  size_t query_nodes = device_query_graph.total_nodes;
  size_t query_graphs_bytes = sigmo::getDeviceGraphAllocSize(device_query_graph);

  std::cout << "------------- Configs -------------" << std::endl;
  std::cout << "Device: " << gpu_name << std::endl;
  std::cout << "Batch Size: " << DATA_GRAPH_BATCH_SIZE << std::endl;
  std::cout << "Refinement Steps: " << args.refinement_steps << std::endl;
  if (args.use_cs) {
      std::cout << "Use CS (Partial Join based on Query Size): Yes" << std::endl;
  }

  size_t batch_idx = 0;
  bool stop_processing = false;

  while (!stop_processing) {
    if (total_data_graphs_processed >= args.max_data_graphs) break;

    std::vector<std::string> batch_lines;
    std::string line;
    size_t lines_read = 0;
    while (lines_read < DATA_GRAPH_BATCH_SIZE && std::getline(data_file_stream, line)) {
        if (!line.empty()) { batch_lines.push_back(line); lines_read++; }
    }
    if (batch_lines.empty()) break;

    batch_idx++;
    std::cout << "\n>>> Processing Batch " << batch_idx << " (" << batch_lines.size() << " raw lines)..." << std::endl;


    std::vector<sigmo::CSRGraph> current_batch_host_graphs;
    current_batch_host_graphs.reserve(batch_lines.size());
    for (const auto& l : batch_lines) {
        try { current_batch_host_graphs.push_back(sigmo::IntermediateGraph{l}.toCSRGraph()); } 
        catch (...) { continue; }
    }

    // Data Multiply Factor
    if (args.multiply_factor_data > 1) {
        size_t orig = current_batch_host_graphs.size();
        for (size_t i = 1; i < args.multiply_factor_data; ++i)
            current_batch_host_graphs.insert(current_batch_host_graphs.end(), current_batch_host_graphs.begin(), current_batch_host_graphs.begin() + orig);
    }

    // Max Graphs Check
    if (total_data_graphs_processed + current_batch_host_graphs.size() > args.max_data_graphs) {
        size_t keep = args.max_data_graphs - total_data_graphs_processed;
        if (current_batch_host_graphs.size() > keep) 
            current_batch_host_graphs.erase(current_batch_host_graphs.begin() + keep, current_batch_host_graphs.end());
        stop_processing = true;
    }
    total_data_graphs_processed += current_batch_host_graphs.size();


    TimeEvents host_time_events;
    host_time_events.add("setup_data_start");

    sigmo::DeviceBatchedCSRGraph device_data_graph = sigmo::createDeviceCSRGraph(queue, current_batch_host_graphs);
    size_t data_nodes = device_data_graph.total_nodes;
    size_t num_data_graphs = device_data_graph.num_graphs;


    sigmo::candidates::Candidates candidates{queue, query_nodes, data_nodes};
    sigmo::signature::Signature<> signatures{queue, data_nodes, query_nodes};
    sigmo::signature::PathSignature path_signatures{queue, data_nodes, query_nodes};
    sigmo::signature::CycleSignature cycle_signatures{queue, data_nodes, query_nodes};

    host_time_events.add("setup_data_end");

    host_time_events.add("sig_gen_start");
    std::chrono::duration<double> time;

    // 1. Label Signatures
    auto e1 = signatures.generateDataSignatures(device_data_graph);
    queue.wait_and_throw();
    data_sig_times.push_back(e1.getProfilingInfo());


    auto e2 = signatures.generateQuerySignatures(device_query_graph);
    queue.wait_and_throw();
    query_sig_times.push_back(e2.getProfilingInfo());

    // 2. Path Signatures
    auto e_path_1 = path_signatures.generateDataPathSignatures(device_data_graph);
    queue.wait_and_throw();
    data_path_sig_times.push_back(e_path_1.getProfilingInfo());

    auto e_path_2 = path_signatures.generateQueryPathSignatures(device_query_graph);
    queue.wait_and_throw();
    query_path_sig_times.push_back(e_path_2.getProfilingInfo());

    // 3. Cycle Signatures
    auto e_cycle_1 = cycle_signatures.generateDataCycleSignatures(device_data_graph);
    queue.wait_and_throw();
    data_cycle_sig_times.push_back(e_cycle_1.getProfilingInfo());

    auto e_cycle_2 = cycle_signatures.generateQueryCycleSignatures(device_query_graph);
    queue.wait_and_throw();
    query_cycle_sig_times.push_back(e_cycle_2.getProfilingInfo());

    host_time_events.add("sig_gen_end");
    host_time_events.add("filter_proc_start");

    // 4. Initial Filter
    auto e3 = sigmo::isomorphism::filter::filterCandidates(
        queue, device_query_graph, device_data_graph, signatures, path_signatures, cycle_signatures, candidates);
    queue.wait_and_throw();
    filter_times.push_back(e3.getProfilingInfo());

    // 5. Refinement Loop
    for (size_t ref_step = 1; ref_step <= args.refinement_steps; ++ref_step) {
        auto er1 = signatures.refineDataSignatures(device_data_graph, ref_step);
        queue.wait_and_throw();
        data_sig_times.push_back(er1.getProfilingInfo());

        auto er2 = signatures.refineQuerySignatures(device_query_graph, ref_step);
        queue.wait_and_throw();
        query_sig_times.push_back(er2.getProfilingInfo());

        auto er3 = sigmo::isomorphism::filter::refineCandidates(queue, device_query_graph, device_data_graph, signatures, candidates);
        queue.wait_and_throw();
        filter_times.push_back(er3.getProfilingInfo());
    }
    host_time_events.add("filter_proc_end");

    size_t* batch_full_matches = sycl::malloc_shared<size_t>(1, queue);
    batch_full_matches[0] = 0;
    std::chrono::duration<double> batch_join_gpu_time{0};

    if (!args.skip_join) {
        host_time_events.add("mapping_start");
        sigmo::isomorphism::mapping::GMCR gmcr{queue};
        gmcr.generateGMCR(device_query_graph, device_data_graph, candidates);
        host_time_events.add("mapping_end");

        host_time_events.add("join_start");
        int partial_match_depth = 0; 

        if (args.use_cs) {
            if (first_query_nodes > 0) {
                partial_match_depth = static_cast<int>(first_query_nodes);
      
                if (batch_idx == 1) {
                    std::cout << "[Config] --use-cs enabled. partial_match_depth set to " << partial_match_depth << std::endl;
                }
            } else {
                if (batch_idx == 1) {
                    std::cerr << "[Warning] --use-cs enabled but no query graphs found. Keeping partial_match_depth at 0." << std::endl;
                }
            }
        }

        if (partial_match_depth == 0) {
            // Optimized Direct Join
            auto join_e = sigmo::isomorphism::join::joinCandidates(
                queue, device_query_graph, device_data_graph, candidates, gmcr, batch_full_matches, !args.find_all);
            join_e.wait();
            batch_join_gpu_time = join_e.getProfilingInfo();
        } else {
 
            const size_t MAX_PARTIAL_MATCHES = 100000000;
            const size_t partial_match_size = partial_match_depth + 1;
            int* partial_matches = sycl::malloc_shared<int>(MAX_PARTIAL_MATCHES * partial_match_size, queue);
            size_t* num_partial_matches = sycl::malloc_shared<size_t>(1, queue);
            num_partial_matches[0] = 0;

            // Step 1: Partial
            auto start_cpu_step1 = std::chrono::high_resolution_clock::now();
            auto join_e2 = sigmo::isomorphism::join::joinPartialCandidates2(
                queue, device_query_graph, device_data_graph, candidates, partial_match_depth, partial_matches, num_partial_matches);
            join_e2.wait();
            auto end_cpu_step1 = std::chrono::high_resolution_clock::now();
            
            size_t partial_matches_found = num_partial_matches[0];
            uint32_t* pair_done = sycl::malloc_shared<uint32_t>(num_query_graphs * num_data_graphs, queue);
            queue.memset(pair_done, 0, sizeof(uint32_t) * num_query_graphs * num_data_graphs).wait();

            // Step 2: Full Join
            auto start_cpu_step2 = std::chrono::high_resolution_clock::now();
            auto full_join_e = sigmo::isomorphism::join::joinWithPartialMatches(
                queue, device_query_graph, device_data_graph, candidates, gmcr,
                partial_match_depth, partial_matches, partial_matches_found,
                batch_full_matches, !args.find_all, pair_done);
            full_join_e.wait();
            auto end_cpu_step2 = std::chrono::high_resolution_clock::now();

       
            batch_join_gpu_time = (end_cpu_step1 - start_cpu_step1) + (end_cpu_step2 - start_cpu_step2);

            sycl::free(pair_done, queue);
            sycl::free(partial_matches, queue);
            sycl::free(num_partial_matches, queue);
        }
        host_time_events.add("join_end");
    }


    size_t batch_matches = batch_full_matches[0];
    global_total_matches += batch_matches;
    total_join_time_gpu += batch_join_gpu_time;


    total_host_setup_time += host_time_events.getRangeTime("setup_data_start", "setup_data_end");
    total_host_sig_gen_time += host_time_events.getRangeTime("sig_gen_start", "sig_gen_end");
    total_host_filter_time += host_time_events.getRangeTime("filter_proc_start", "filter_proc_end");
    if (!args.skip_join) {
        total_host_mapping_time += host_time_events.getRangeTime("mapping_start", "mapping_end");
        total_host_join_time += host_time_events.getRangeTime("join_start", "join_end");
    }


    std::cout << "    -> Matches found: " << batch_matches << std::endl;


    sycl::free(batch_full_matches, queue);
    sigmo::destroyDeviceCSRGraph(device_data_graph, queue);
    current_batch_host_graphs.clear();


    if (!args.find_all && global_total_matches > 0) {
        std::cout << "[Info] Match found (find_all=false). Stopping." << std::endl;
        break;
    }
  }

  if (data_file_stream.is_open()) data_file_stream.close();


  auto global_end_clock = std::chrono::high_resolution_clock::now();
  total_host_wall_time = global_end_clock - global_start_clock;
  std::cout << "\n============= Final Stats (Batched) =============" << std::endl;
  std::cout << "Total Data Graphs: " << total_data_graphs_processed << std::endl;


  std::cout << "------------- Overall GPU Stats -------------" << std::endl;
  std::chrono::duration<double> total_sig_query_time = std::accumulate(query_sig_times.begin(), query_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_sig_data_time = std::accumulate(data_sig_times.begin(), data_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_filter_time_accum = std::accumulate(filter_times.begin(), filter_times.end(), std::chrono::duration<double>(0));

  std::chrono::duration<double> total_path_sig_query_time = std::accumulate(query_path_sig_times.begin(), query_path_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_path_sig_data_time = std::accumulate(data_path_sig_times.begin(), data_path_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_path_filter_time = std::accumulate(path_filter_times.begin(), path_filter_times.end(), std::chrono::duration<double>(0));

  std::chrono::duration<double> total_cycle_sig_query_time = std::accumulate(query_cycle_sig_times.begin(), query_cycle_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_cycle_sig_data_time = std::accumulate(data_cycle_sig_times.begin(), data_cycle_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_cycle_filter_time = std::accumulate(cycle_filter_times.begin(), cycle_filter_times.end(), std::chrono::duration<double>(0));

  std::chrono::duration<double> total_time_gpu = total_sig_data_time + total_filter_time_accum + total_sig_query_time
                                           + total_path_sig_data_time + total_path_sig_query_time + total_path_filter_time
                                           + total_cycle_sig_data_time + total_cycle_sig_query_time + total_cycle_filter_time
                                           + total_join_time_gpu;

  std::cout << "Data (label) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_data_time).count() << " ms" << std::endl;
  std::cout << "Query (label) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_query_time).count() << " ms" << std::endl;
  std::cout << "Data (path) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_path_sig_data_time).count() << " ms" << std::endl;
  std::cout << "Query (path) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_path_sig_query_time).count() << " ms" << std::endl;
  std::cout << "Data (cycle) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_cycle_sig_data_time).count() << " ms" << std::endl;
  std::cout << "Query (cycle) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_cycle_sig_query_time).count() << " ms" << std::endl;
  std::cout << "Filter (label/fused) time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_filter_time_accum).count() << " ms" << std::endl;

  if (args.skip_join) {
    std::cout << "Join time: skipped" << std::endl;
  } else {
    std::cout << "Join time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_join_time_gpu).count() << " ms" << std::endl;
  }
  std::cout << "Total time (GPU Sum): " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time_gpu).count() << " ms" << std::endl;


  std::cout << "------------- Overall Host Stats -------------" << std::endl;
  std::cout << "Setup Data time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(total_host_setup_time).count()
            << " ms (not included in total)" << std::endl;
  
  std::cout << "Signature Gen time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(total_host_sig_gen_time).count() << " ms"
            << std::endl;

  std::cout << "Filter & Refine time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(total_host_filter_time).count() << " ms"
            << std::endl;

  if (args.skip_join) {
    std::cout << "Mapping time: skipped" << std::endl;
    std::cout << "Join time: skipped" << std::endl;
  } else {
    std::cout << "Mapping time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(total_host_mapping_time).count() << " ms"
              << std::endl;
    std::cout << "Join time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(total_host_join_time).count() << " ms"
              << std::endl;
  }
  std::cout << "Total time (Wall Clock): " << std::chrono::duration_cast<std::chrono::milliseconds>(total_host_wall_time).count()
            << " ms" << std::endl;

  std::cout << "------------- Results -------------" << std::endl;
  if (!args.skip_join) { std::cout << "# Matches: " << formatNumber(global_total_matches) << std::endl; }


  sigmo::destroyDeviceCSRGraph(device_query_graph, queue);
  std::cout << "[!] End" << std::endl;

  return 0;
}