/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>
#include <sycl/sycl.hpp>
#include "./utils.hpp"
#include <numeric>
#include <sigmo.hpp>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip> 
#include <fstream> 
#include <algorithm> 


struct BatchLog {
    size_t batch_id;
    double gpu_filter_ms;
    double cpu_filter_ms;
    double gpu_join_ms;
    double cpu_join_ms;
    size_t batch_matches;
};

int main(int argc, char** argv) {
  // Initialize MPI.
  MPI_Init(&argc, &argv);
  
  // Get MPI rank and size 
  int mpi_rank, mpi_size, local_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_free(&local_comm);

  // Initialize program arguments and device options.
  Args args{argc, argv, sigmo::device::deviceOptions};

  const size_t DATA_BATCH_SIZE = 5000000; 

  sigmo::DeviceBatchedCSRGraph device_query_graph;
  size_t num_query_graphs;
  size_t num_data_graphs; 

  // Query all GPU devices available on the node.
  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (devices.empty()) {
    throw std::runtime_error("No GPU devices found on this node");
  }
  // Select device based on the local MPI rank.
  sycl::device selected_device = devices[local_rank % devices.size()];
  // Create a SYCL queue with the selected device and profiling enabled.
  sycl::queue queue{selected_device, sycl::property::queue::enable_profiling{}};

  size_t gpu_mem = queue.get_device().get_info<sycl::info::device::global_mem_size>();
  std::string gpu_name = queue.get_device().get_info<sycl::info::device::name>();


  std::vector<sigmo::CSRGraph> data_graphs;

  size_t total_valid_graphs_global = 0; 

  size_t first_query_nodes = 0;

  if (args.query_data) {
    // Load query graphs serially (all ranks get all queries)
    auto query_graphs = sigmo::io::loadCSRGraphsFromFile(args.query_file);

    auto isValidLine = [](const std::string &line) -> bool {
        return (std::count(line.begin(), line.end(), 'n') == 1 &&
                std::count(line.begin(), line.end(), 'e') == 1 &&
                std::count(line.begin(), line.end(), 'l') == 1);
    };

    auto estimateGraphCost = [](const std::string &line) -> double {
        std::istringstream iss(line);
        std::string token;
        size_t num_nodes = 0;
        size_t num_edges = 0;

        if (iss >> token) {
            auto pos = token.find_first_of("0123456789");
            if (pos != std::string::npos) {
                num_nodes = static_cast<size_t>(std::stoul(token.substr(pos)));
            }
        }
        if (iss >> token) { /* l=xx ignored */ }
        if (iss >> token) {
            auto pos = token.find_first_of("0123456789");
            if (pos != std::string::npos) {
                num_edges = static_cast<size_t>(std::stoul(token.substr(pos)));
            }
        }

        double cost = static_cast<double>(num_nodes) * static_cast<double>(num_nodes)
                      + static_cast<double>(num_edges);
        return (cost > 0.0) ? cost : 1.0; 
    };

    std::vector<std::string> data_lines; 
    std::vector<int> owners; 

    if (mpi_rank == 0) {
        std::ifstream file(args.data_file);
        if (!file.is_open()) {
            std::cerr << "Error: Rank 0 failed to open data file: " << args.data_file << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::string line;
        std::vector<double> costs;
        costs.reserve(100000);

        while (std::getline(file, line)) {
            if (!isValidLine(line)) { continue; }
            if (args.max_data_graphs > 0 && static_cast<long long>(total_valid_graphs_global) >= args.max_data_graphs) {
                break; 
            }
            double c = estimateGraphCost(line);
            costs.push_back(c);
            ++total_valid_graphs_global;
        }
        file.close();

        if (total_valid_graphs_global == 0) {
            std::cerr << "Error: No valid data graphs found in file: " << args.data_file << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        owners.resize(total_valid_graphs_global);


        struct Task { size_t idx; double cost; };
        std::vector<Task> tasks;
        tasks.reserve(total_valid_graphs_global);
        for (size_t i = 0; i < total_valid_graphs_global; ++i) {
            tasks.push_back(Task{i, costs[i]});
        }

        std::sort(tasks.begin(), tasks.end(), [](const Task &a, const Task &b) { return a.cost > b.cost; });

        std::vector<double> load(mpi_size, 0.0);
        for (const auto &t : tasks) {
            int best_rank = 0;
            double best_load = load[0];
            for (int r = 1; r < mpi_size; ++r) {
                if (load[r] < best_load) {
                    best_load = load[r];
                    best_rank = r;
                }
            }
            owners[t.idx] = best_rank;
            load[best_rank] += t.cost;
        }

        std::cout << "------------- Load Balancing (Estimated Cost) -------------" << std::endl;
        for (int r = 0; r < mpi_size; ++r) {
            std::cout << "  Rank " << r << " estimated load = " << load[r] << std::endl;
        }
        std::cout << "  Total valid data graphs = " << total_valid_graphs_global << std::endl;
    }


    MPI_Bcast(&total_valid_graphs_global, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (total_valid_graphs_global == 0) {
        MPI_Finalize();
        return 0;
    }

    if (mpi_rank != 0) {
        owners.resize(total_valid_graphs_global);
    }
    MPI_Bcast(owners.data(), static_cast<int>(total_valid_graphs_global), MPI_INT, 0, MPI_COMM_WORLD);


    data_lines.reserve(total_valid_graphs_global / mpi_size + 1);
    {
        std::ifstream file_reader(args.data_file);
        if (!file_reader.is_open()) {
            std::cerr << "Error: Rank " << mpi_rank << " failed to open data file: " << args.data_file << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        std::string line;
        size_t valid_idx = 0; 
        while (std::getline(file_reader, line)) {
            if (!isValidLine(line)) { continue; }
            if (valid_idx >= total_valid_graphs_global) { break; }
            if (owners[valid_idx] == mpi_rank) {
                data_lines.push_back(line);
            }
            ++valid_idx;
        }
        file_reader.close();
    }
    
    data_graphs = sigmo::io::loadCSRGraphsFromLines(data_lines);



    // Query Filtering
    if (args.query_filter.active) {
      for (int i = 0; i < (int)query_graphs.size(); ++i) {
        if (query_graphs[i].getNumNodes() > args.query_filter.max_nodes || query_graphs[i].getNumNodes() < args.query_filter.min_nodes) {
          query_graphs.erase(query_graphs.begin() + i);
          i--;
        }
      }
    }


    if (!query_graphs.empty()) {
        first_query_nodes = query_graphs[0].getNumNodes();
    }

    num_query_graphs = query_graphs.size();
    for (size_t i = 1; i < args.multiply_factor_query; ++i) {
      query_graphs.insert(query_graphs.end(), query_graphs.begin(), query_graphs.begin() + num_query_graphs);
    }
    num_data_graphs = data_graphs.size();
    for (size_t i = 1; i < args.multiply_factor_data; ++i) {
      data_graphs.insert(data_graphs.end(), data_graphs.begin(), data_graphs.begin() + num_data_graphs);
    }
    if (query_graphs.size() > args.max_query_graphs) { query_graphs.erase(query_graphs.begin() + args.max_query_graphs, query_graphs.end()); }
    if (data_graphs.size() > args.max_data_graphs) { data_graphs.erase(data_graphs.begin() + args.max_data_graphs, data_graphs.end()); }
    
    device_query_graph = sigmo::createDeviceCSRGraph(queue, query_graphs);
  } else {
    throw std::runtime_error("Specify input data");
  }

  num_query_graphs = device_query_graph.num_graphs;
  num_data_graphs = data_graphs.size();
  size_t query_nodes = device_query_graph.total_nodes;

  size_t query_graphs_bytes = sigmo::getDeviceGraphAllocSize(device_query_graph);
  TimeEvents host_time_events;
  
  // Get total data graphs across all ranks (Verify)
  size_t total_data_graphs_check = 0;
  MPI_Reduce(&num_data_graphs, &total_data_graphs_check, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    std::cout << "------------- Input Data -------------" << std::endl;
    std::cout << "Read data and query graphs" << std::endl;
    std::cout << "# Query Graphs " << num_query_graphs << std::endl;
    std::cout << "# Data Graphs (Total across all ranks) " << total_valid_graphs_global << std::endl;
    std::cout << "(MPI_Reduce check sum: " << total_data_graphs_check << ")" << std::endl;
    std::cout << "Data Distribution: Cost-Balanced (LPT on nodes^2+edges)" << std::endl;

    std::cout << "------------- Configs -------------" << std::endl;
    std::cout << "MPI Ranks: " << mpi_size << std::endl;
    std::cout << "Filter domain: " << args.candidates_domain << std::endl;
    std::cout << "Filter Work Group Size: " << sigmo::device::deviceOptions.filter_work_group_size << std::endl;
    std::cout << "Join Work Group Size: " << sigmo::device::deviceOptions.join_work_group_size << std::endl;
    std::cout << "Find all: " << (args.find_all ? "Yes" : "No") << std::endl;
    std::cout << "Data Batch Size (per rank): " << DATA_BATCH_SIZE << std::endl;
    
    if (args.use_cs) {
        std::cout << "Use CS (Partial Join based on Query Size): Yes" << std::endl;
    }
  }

  host_time_events.add("setup_data_start");
  if (mpi_rank == 0) {
    std::cout << "------------- Setup Data (Rank 0) -------------" << std::endl;
    std::cout << "Allocated " << getBytesSize(query_graphs_bytes) << " for query data (fixed)" << std::endl;
  }

  host_time_events.add("setup_data_end");

  size_t* total_full_matches = sycl::malloc_shared<size_t>(1, queue);
  total_full_matches[0] = 0;

  std::vector<BatchLog> batch_logs;

  double total_rank_gpu_filter_time = 0;
  double total_rank_cpu_filter_time = 0;
  double total_rank_gpu_join_time = 0;
  double total_rank_cpu_join_time = 0;
  
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    host_time_events.add("mpi_start");
  }

  // ========== BATCH PROCESSING LOOP [START] ==========
  size_t num_data_batches = (data_graphs.size() + DATA_BATCH_SIZE - 1) / DATA_BATCH_SIZE;

  for (size_t batch_idx = 0; batch_idx < num_data_batches; ++batch_idx) {
   
    std::chrono::duration<double> batch_gpu_filter_time{0};
    std::chrono::duration<double> batch_gpu_join_time{0};

    size_t matches_in_this_batch = 0;
    size_t matches_before_batch = total_full_matches[0]; 


    size_t start_idx = batch_idx * DATA_BATCH_SIZE;
    size_t end_idx = std::min((batch_idx + 1) * DATA_BATCH_SIZE, data_graphs.size());
    std::vector<sigmo::CSRGraph> current_data_batch(data_graphs.begin() + start_idx, data_graphs.begin() + end_idx);

    if (current_data_batch.empty()) continue;

    sigmo::DeviceBatchedCSRGraph device_data_graph = sigmo::createDeviceCSRGraph(queue, current_data_batch);
    size_t data_nodes_batch = device_data_graph.total_nodes;
    

    sigmo::candidates::Candidates candidates{queue, query_nodes, data_nodes_batch};
    sigmo::signature::Signature<> signatures{queue, data_nodes_batch, query_nodes};
    sigmo::signature::PathSignature path_signatures{queue, data_nodes_batch, query_nodes};
    sigmo::signature::CycleSignature cycle_signatures{queue, data_nodes_batch, query_nodes};

    auto host_batch_filter_start = std::chrono::steady_clock::now();

    // 4. Runtime Filter Phase (for this batch)
    std::chrono::duration<double> time;

    auto e1 = signatures.generateDataSignatures(device_data_graph);
    auto e_path_1 = path_signatures.generateDataPathSignatures(device_data_graph);
    auto e_cycle_1 = cycle_signatures.generateDataCycleSignatures(device_data_graph);
    e1.wait();
    batch_gpu_filter_time += e1.getProfilingInfo();
    e_path_1.wait();
    batch_gpu_filter_time += e_path_1.getProfilingInfo();
    e_cycle_1.wait();
    batch_gpu_filter_time += e_cycle_1.getProfilingInfo();
  auto e2 = signatures.generateQuerySignatures(device_query_graph);
    auto e_path_2 = path_signatures.generateQueryPathSignatures(device_query_graph);
    auto e_cycle_2 = cycle_signatures.generateQueryCycleSignatures(device_query_graph);
    e2.wait();
    batch_gpu_filter_time += e2.getProfilingInfo();
    e_path_2.wait();
    batch_gpu_filter_time += e_path_2.getProfilingInfo();
    e_cycle_2.wait();
    batch_gpu_filter_time += e_cycle_2.getProfilingInfo();


    auto e3 = sigmo::isomorphism::filter::filterCandidates(
        queue, device_query_graph, device_data_graph, signatures, path_signatures, cycle_signatures, candidates);
    e3.wait();
    batch_gpu_filter_time += e3.getProfilingInfo();

    for (size_t ref_step = 1; ref_step <= args.refinement_steps; ++ref_step) {
      auto e1_ref = signatures.refineDataSignatures(device_data_graph, ref_step);
      e1_ref.wait();
      batch_gpu_filter_time += e1_ref.getProfilingInfo();

      auto e2_ref = signatures.refineQuerySignatures(device_query_graph, ref_step);
      e2_ref.wait();
      batch_gpu_filter_time += e2_ref.getProfilingInfo();

      auto e3_ref = sigmo::isomorphism::filter::refineCandidates(queue, device_query_graph, device_data_graph, signatures, candidates);
      e3_ref.wait();
      batch_gpu_filter_time += e3_ref.getProfilingInfo();
    }
    

    auto host_batch_filter_end = std::chrono::steady_clock::now();

    // 5. Join Phase
    auto host_batch_mapping_start = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> host_batch_join_end;
    
    if (!args.skip_join) {
      sigmo::isomorphism::mapping::GMCR gmcr{queue};
      gmcr.generateGMCR(device_query_graph, device_data_graph, candidates);
      
      matches_before_batch = total_full_matches[0];

      int partial_match_depth = 0; 

      if (args.use_cs) {
          if (first_query_nodes > 0) {
              partial_match_depth = static_cast<int>(first_query_nodes);

              if (batch_idx == 0 && mpi_rank == 0) {
                  std::cout << "[Config] --use-cs enabled. partial_match_depth set to " << partial_match_depth << std::endl;
              }
          } else {
              if (batch_idx == 0 && mpi_rank == 0) {
                  std::cerr << "[Warning] --use-cs enabled but no query graphs found. Keeping partial_match_depth at 0." << std::endl;
              }
          }
      }

      std::chrono::duration<double> join_time_part1{0};
      std::chrono::duration<double> join_time_part2{0};

      if (partial_match_depth == 0) {

        auto join_e = sigmo::isomorphism::join::joinCandidates(
            queue, device_query_graph, device_data_graph, candidates, gmcr, total_full_matches, !args.find_all);
        join_e.wait();
        join_time_part1 = join_e.getProfilingInfo();
      } else {

        const size_t MAX_PARTIAL_MATCHES = 100000000;
        const size_t partial_match_size = partial_match_depth + 1; 

        int* partial_matches = sycl::malloc_shared<int>(MAX_PARTIAL_MATCHES * partial_match_size, queue);
        size_t* num_partial_matches = sycl::malloc_shared<size_t>(1, queue);
        num_partial_matches[0] = 0;

        auto join_e2 = sigmo::isomorphism::join::joinPartialCandidates2(
            queue, device_query_graph, device_data_graph, candidates, partial_match_depth,
            partial_matches, num_partial_matches);
        join_e2.wait();
        join_time_part1 = join_e2.getProfilingInfo(); 
        size_t partial_matches_found = num_partial_matches[0];

        const uint32_t num_q = device_query_graph.num_graphs;
        const uint32_t num_d = device_data_graph.num_graphs;

        uint32_t* pair_done = sycl::malloc_shared<uint32_t>(num_q * num_d, queue);
        queue.memset(pair_done, 0, sizeof(uint32_t) * num_q * num_d).wait();

        bool find_first = !args.find_all;

        auto full_join_e = sigmo::isomorphism::join::joinWithPartialMatches(
            queue, device_query_graph, device_data_graph, candidates, gmcr,
            partial_match_depth, partial_matches, partial_matches_found,
            total_full_matches, find_first, pair_done);
        full_join_e.wait();
        join_time_part2 = full_join_e.getProfilingInfo();

        sycl::free(pair_done, queue);
        sycl::free(partial_matches, queue);
        sycl::free(num_partial_matches, queue);
      }

      host_batch_join_end = std::chrono::steady_clock::now();
      
      batch_gpu_join_time = join_time_part1 + join_time_part2; 
      matches_in_this_batch = total_full_matches[0] - matches_before_batch; 

    } else {
      host_batch_join_end = std::chrono::steady_clock::now(); 
      batch_gpu_join_time = std::chrono::duration<double>(0);
      matches_in_this_batch = 0;
    }

    BatchLog log;
    log.batch_id = batch_idx + 1;
    log.gpu_filter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_gpu_filter_time).count();
    log.cpu_filter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(host_batch_filter_end - host_batch_filter_start).count();
    log.gpu_join_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_gpu_join_time).count();
    log.cpu_join_ms = std::chrono::duration_cast<std::chrono::milliseconds>(host_batch_join_end - host_batch_mapping_start).count();
    log.batch_matches = matches_in_this_batch;
    batch_logs.push_back(log);

    total_rank_gpu_filter_time += log.gpu_filter_ms;
    total_rank_cpu_filter_time += log.cpu_filter_ms;
    total_rank_gpu_join_time += log.gpu_join_ms;
    total_rank_cpu_join_time += log.cpu_join_ms;

    sigmo::destroyDeviceCSRGraph(device_data_graph, queue);
    queue.wait_and_throw(); 
  }
  // ========== BATCH PROCESSING LOOP [END] ==========


  host_time_events.add("processing_end");
  
  double rank_time = std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("setup_data_start", "processing_end")).count();
  
  std::vector<double> all_rank_times;
  std::vector<double> all_filter_times; 
  std::vector<double> all_join_times;   
  std::vector<size_t> all_data_graphs_counts;

  if (mpi_rank == 0) {
    all_rank_times.resize(mpi_size);
    all_filter_times.resize(mpi_size);
    all_join_times.resize(mpi_size);
    all_data_graphs_counts.resize(mpi_size);
  }

  // ========== ORDERED PER-RANK PRINTING ==========
  for (int i = 0; i < mpi_size; ++i) {
      if (mpi_rank == i) {
          std::cout << "\n========================================================================================\n";
          std::cout << "DETAILED REPORT FOR RANK " << i << " (Node: " << gpu_name << ")" << "\n";
          std::cout << "========================================================================================\n";
          
          std::cout << "--- Per-Batch Details ---\n";
          std::cout << std::left
                    << "| " << std::setw(7) << "Batch"
                    << "| " << std::setw(17) << "GPU Filter (ms)"
                    << "| " << std::setw(17) << "CPU Filter (ms)"
                    << "| " << std::setw(15) << "GPU Join (ms)"
                    << "| " << std::setw(15) << "CPU Join (ms)"
                    << "| " << std::setw(12) << "Matches" << "|\n";
          std::cout << "|---------|-------------------|-------------------|-----------------|-----------------|--------------|\n";
          
          for (const auto& log : batch_logs) {
              std::cout << std::left
                        << "| " << std::setw(7) << log.batch_id
                        << "| " << std::setw(17) << log.gpu_filter_ms
                        << "| " << std::setw(17) << log.cpu_filter_ms
                        << "| " << std::setw(15) << log.gpu_join_ms
                        << "| " << std::setw(15) << log.cpu_join_ms
                        << "| " << std::setw(12) << log.batch_matches << "|\n";
          }

          std::cout << "\n--- Summary for Rank " << i << " ---\n";
          std::cout << std::left
                    << "| " << std::setw(23) << "Total GPU Filter (ms)"
                    << "| " << std::setw(23) << "Total CPU Filter (ms)"
                    << "| " << std::setw(21) << "Total GPU Join (ms)"
                    << "| " << std::setw(21) << "Total CPU Join (ms)"
                    << "|\n";
          std::cout << "|-------------------------|-------------------------|-----------------------|-----------------------|\n";
          std::cout << std::left
                    << "| " << std::setw(23) << total_rank_gpu_filter_time
                    << "| " << std::setw(23) << total_rank_cpu_filter_time
                    << "| " << std::setw(21) << total_rank_gpu_join_time
                    << "| " << std::setw(21) << total_rank_cpu_join_time << "|\n";

          std::cout << "\n"
                    << "| " << std::setw(25) << "Total Rank Wall-Time (ms)"
                    << "| " << std::setw(25) << "Processed Data Graphs"
                    << "| " << std::setw(25) << "Total Rank Matches" << "|\n";
          std::cout << "|---------------------------|---------------------------|---------------------------|\n";
          std::cout << std::left
                    << "| " << std::setw(25) << rank_time 
                    << "| " << std::setw(25) << num_data_graphs 
                    << "| " << std::setw(25) << total_full_matches[0]
                    << "|\n";
      }
      MPI_Barrier(MPI_COMM_WORLD);
  }
  // ========== END OF ORDERED PRINTING ==========


  MPI_Gather(&rank_time, 1, MPI_DOUBLE, all_rank_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&total_rank_gpu_filter_time, 1, MPI_DOUBLE, all_filter_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&total_rank_gpu_join_time, 1, MPI_DOUBLE, all_join_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&num_data_graphs, 1, MPI_UNSIGNED_LONG, all_data_graphs_counts.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    host_time_events.add("mpi_end");
  }

  size_t final_total_matches = 0;
  MPI_Reduce(total_full_matches, &final_total_matches, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  // --- End MPI Aggregation ---


  if (mpi_rank == 0) {
    std::cout << "\n========================================================================================\n";
    std::cout << "GLOBAL SUMMARY (Aggregated from all " << mpi_size << " ranks)\n";
    std::cout << "========================================================================================\n";

    std::cout << "MPI time (Wall-Clock for rank 0 setup-to-end): "
      << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("mpi_start", "mpi_end")).count() << " ms"
      << std::endl;

    double sum_rank_time = 0;
    double max_rank_time = 0;
    for (int i = 0; i < mpi_size; ++i) {
        sum_rank_time += all_rank_times[i];
        if (all_rank_times[i] > max_rank_time) {
            max_rank_time = all_rank_times[i];
        }
    }
    std::cout << "Max rank time (Wall-Clock): " << max_rank_time << " ms (This is the 'Total Program Time')" << std::endl;
    std::cout << "Avg rank time (Wall-Clock): " << (sum_rank_time / mpi_size) << " ms" << std::endl;

    std::cout << "\n------------- Rank-wise Performance and Load -------------\n"; 
    std::cout << std::left 
              << "| " << std::setw(6) << "Rank"
              << "| " << std::setw(20) << "Host Wall-Time" 
              << "| " << std::setw(18) << "Data Graphs" 
              << "| " << std::setw(20) << "Total GPU Filter" 
              << "| " << std::setw(18) << "Total GPU Join" << "|\n"; 
    std::cout << "|--------|----------------------|--------------------|----------------------|--------------------|\n"; 
    
    for (int i = 0; i < mpi_size; ++i) {
      std::cout << std::left 
                << "| " << std::setw(6) << i
                << "| " << std::setw(20) << all_rank_times[i] 
                << "| " << std::setw(18) << all_data_graphs_counts[i] 
                << "| " << std::setw(20) << all_filter_times[i] 
                << "| " << std::setw(18) << all_join_times[i] << "|\n"; 
    }
    std::cout << "|--------|----------------------|--------------------|----------------------|--------------------|\n"; 
    
    std::cout << "\n------------- Global Match Results -------------\n";
    if (!args.skip_join) { 
      std::cout << "# Total Matches (Summed from all ranks): " << formatNumber(final_total_matches) << std::endl;
      std::cout << "# Average Matches (per rank): " << formatNumber(final_total_matches / mpi_size) << std::endl;
    }
  }

  sycl::free(total_full_matches, queue);
  sigmo::destroyDeviceCSRGraph(device_query_graph, queue);

  // Finalize MPI.
  MPI_Finalize();
  
  return 0;
}