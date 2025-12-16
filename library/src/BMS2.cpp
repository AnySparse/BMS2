#include "./utils.hpp"
#include <numeric>
#include <sigmo.hpp>
#include <sycl/sycl.hpp>
#include <chrono> 

int main(int argc, char** argv) {
  Args args{argc, argv, sigmo::device::deviceOptions};

  sigmo::DeviceBatchedCSRGraph device_data_graph;
  sigmo::DeviceBatchedCSRGraph device_query_graph;
  size_t num_query_graphs;
  size_t num_data_graphs;
  sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
  size_t gpu_mem = queue.get_device().get_info<sycl::info::device::global_mem_size>();
  std::string gpu_name = queue.get_device().get_info<sycl::info::device::name>();

  TimeEvents host_time_events;

  size_t first_query_nodes = 0;

  if (args.query_data) {
    auto query_graphs = sigmo::io::loadCSRGraphsFromFile(args.query_file);
    auto data_graphs = sigmo::io::loadCSRGraphsFromFile(args.data_file);
    

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
    device_data_graph = sigmo::createDeviceCSRGraph(queue, data_graphs);
  } else {
    throw std::runtime_error("Specify input data");
  }

  num_query_graphs = device_query_graph.num_graphs;
  num_data_graphs = device_data_graph.num_graphs;
  size_t query_nodes = device_query_graph.total_nodes;
  size_t data_nodes = device_data_graph.total_nodes;

  size_t data_graph_bytes = sigmo::getDeviceGraphAllocSize(device_data_graph);
  size_t query_graphs_bytes = sigmo::getDeviceGraphAllocSize(device_query_graph);

  std::vector<std::chrono::duration<double>> data_sig_times, query_sig_times, filter_times;

  std::cout << "------------- Input Data -------------" << std::endl;
  std::cout << "Read data graph and query graph" << std::endl;
  std::cout << "# Query Nodes " << query_nodes << std::endl;
  std::cout << "# Query Graphs " << num_query_graphs << std::endl;
  std::cout << "# Data Nodes " << data_nodes << std::endl;
  std::cout << "# Data Graphs " << num_data_graphs << std::endl;

  std::cout << "------------- Configs -------------" << std::endl;
  std::cout << "Filter domain: " << args.candidates_domain << std::endl;
  std::cout << "Filter Work Group Size: " << sigmo::device::deviceOptions.filter_work_group_size << std::endl;
  std::cout << "Join Work Group Size: " << sigmo::device::deviceOptions.join_work_group_size << std::endl;
  std::cout << "Find all: " << (args.find_all ? "Yes" : "No") << std::endl;
  if (args.use_cs) {
      std::cout << "Use CS (Partial Join based on Query Size): Yes" << std::endl;
  }

  host_time_events.add("setup_data_start");
  std::cout << "------------- Setup Data -------------" << std::endl;
  std::cout << "Allocated " << getBytesSize(data_graph_bytes) << " for graph data" << std::endl;
  std::cout << "Allocated " << getBytesSize(query_graphs_bytes) << " for query data" << std::endl;

  sigmo::candidates::Candidates candidates{queue, query_nodes, data_nodes};
  size_t candidates_bytes = candidates.getAllocationSize();
  std::cout << "Allocated " << getBytesSize(candidates_bytes) << " for candidates" << std::endl;


  sigmo::signature::Signature<> signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};
  size_t data_signatures_bytes = signatures.getDataSignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(data_signatures_bytes) << " for data signatures" << std::endl;
  size_t query_signatures_bytes = signatures.getQuerySignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(query_signatures_bytes) << " for query signatures" << std::endl;
  size_t tmp_buff_bytes = std::max(data_signatures_bytes, query_signatures_bytes);
  std::cout << "Allocated " << getBytesSize(tmp_buff_bytes) << " for temporary buffer" << std::endl;

  sigmo::signature::PathSignature path_signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};
  size_t data_path_signatures_bytes = path_signatures.getDataSignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(data_path_signatures_bytes) << " for data path signatures" << std::endl;
  size_t query_path_signatures_bytes = path_signatures.getQuerySignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(query_path_signatures_bytes) << " for query path signatures" << std::endl;

  sigmo::signature::CycleSignature cycle_signatures{queue, device_data_graph.total_nodes, device_query_graph.total_nodes};
  size_t data_cycle_signatures_bytes = cycle_signatures.getDataSignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(data_cycle_signatures_bytes) << " for data cycle signatures" << std::endl;
  size_t query_cycle_signatures_bytes = cycle_signatures.getQuerySignatureAllocationSize();
  std::cout << "Allocated " << getBytesSize(query_cycle_signatures_bytes) << " for query cycle signatures" << std::endl;

  host_time_events.add("setup_data_end");

  std::cout << "Total allocated memory: "
            << getBytesSize(
                   data_signatures_bytes + query_signatures_bytes + candidates_bytes + tmp_buff_bytes + data_graph_bytes + query_graphs_bytes
                   + data_path_signatures_bytes + query_path_signatures_bytes
                   + data_cycle_signatures_bytes + query_cycle_signatures_bytes,
                   false)
            << " out of " << getBytesSize(gpu_mem) << " available on " << gpu_name << std::endl;

  std::cout << "------------- Runtime Filter Phase -------------" << std::endl;
   
  host_time_events.add("sig_gen_start");
   
  std::cout << "[*] Initialization Step:" << std::endl;
  std::chrono::duration<double> time;
  std::vector<std::chrono::duration<double>> data_path_sig_times, query_path_sig_times, path_filter_times;
  std::vector<std::chrono::duration<double>> data_cycle_sig_times, query_cycle_sig_times, cycle_filter_times;

  auto e1 = signatures.generateDataSignatures(device_data_graph);
  queue.wait_and_throw();
  time = e1.getProfilingInfo();
  data_sig_times.push_back(time);
  std::cout << "- Data (label) signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  auto e_path_1 = path_signatures.generateDataPathSignatures(device_data_graph);
  queue.wait_and_throw();
  time = e_path_1.getProfilingInfo();
  data_path_sig_times.push_back(time);
  std::cout << "- Data (path) signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  auto e_cycle_1 = cycle_signatures.generateDataCycleSignatures(device_data_graph);
  queue.wait_and_throw();
  time = e_cycle_1.getProfilingInfo();
  data_cycle_sig_times.push_back(time);
  std::cout << "- Data (cycle) signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  auto e2 = signatures.generateQuerySignatures(device_query_graph);
  queue.wait_and_throw();
  time = e2.getProfilingInfo();
  query_sig_times.push_back(time);
  std::cout << "- Query (label) signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  auto e_path_2 = path_signatures.generateQueryPathSignatures(device_query_graph);
  queue.wait_and_throw();
  time = e_path_2.getProfilingInfo();
  query_path_sig_times.push_back(time);
  std::cout << "- Query (path) signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  auto e_cycle_2 = cycle_signatures.generateQueryCycleSignatures(device_query_graph);
  queue.wait_and_throw();
  time = e_cycle_2.getProfilingInfo();
  query_cycle_sig_times.push_back(time);
  std::cout << "- Query (cycle) signatures generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  host_time_events.add("sig_gen_end");
  host_time_events.add("filter_proc_start");

  auto e3 = sigmo::isomorphism::filter::filterCandidates(
      queue, device_query_graph, device_data_graph, signatures, path_signatures, cycle_signatures, candidates);
  queue.wait_and_throw();
  time = e3.getProfilingInfo();
  filter_times.push_back(time);
  std::cout << "- Candidates filtered (fused label + path + cycle) in " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

  for (size_t ref_step = 1; ref_step <= args.refinement_steps; ++ref_step) {
    std::cout << "[*] Refinement step " << ref_step << ":" << std::endl;

    auto e1 = signatures.refineDataSignatures(device_data_graph, ref_step);
    queue.wait_and_throw();
    time = e1.getProfilingInfo();
    data_sig_times.push_back(time);
    std::cout << "- Data signatures refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

    auto e2 = signatures.refineQuerySignatures(device_query_graph, ref_step);
    queue.wait_and_throw();
    time = e2.getProfilingInfo();
    query_sig_times.push_back(time);
    std::cout << "- Query signatures refined in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;

    auto e3 = sigmo::isomorphism::filter::refineCandidates(queue, device_query_graph, device_data_graph, signatures, candidates);
    queue.wait_and_throw();
    time = e3.getProfilingInfo();
    filter_times.push_back(time);
    std::cout << "- Candidates refined (label) in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << std::endl;
  }
   
  host_time_events.add("filter_proc_end");


  std::chrono::duration<double> join_time{0};
  std::chrono::duration<double> join_time2{0};
  size_t* total_full_matches = sycl::malloc_shared<size_t>(1, queue);
  total_full_matches[0] = 0;

  if (!args.skip_join) {
    std::cout << "[*] Generating DQCR" << std::endl;
    host_time_events.add("mapping_start");
    sigmo::isomorphism::mapping::GMCR gmcr{queue};
    gmcr.generateGMCR(device_query_graph, device_data_graph, candidates);
    host_time_events.add("mapping_end");

    std::cout << "[*] Starting Join" << std::endl;
    host_time_events.add("join_start");

    int partial_match_depth = 0; 

    if (args.use_cs) {
        if (first_query_nodes > 0) {
            partial_match_depth = static_cast<int>(first_query_nodes);
            std::cout << "[Config] --use-cs enabled. partial_match_depth set to " << partial_match_depth << std::endl;
        } else {
            std::cerr << "[Warning] --use-cs enabled but no query graphs found. Keeping partial_match_depth at 0." << std::endl;
        }
    }

    if (partial_match_depth == 0) {

      auto join_e = sigmo::isomorphism::join::joinCandidates(
          queue, device_query_graph, device_data_graph, candidates, gmcr, total_full_matches, !args.find_all);
      join_e.wait();
      join_time2 = join_e.getProfilingInfo();
    } else {
 
      std::cout << "------------- Partial Join Phase -------------" << std::endl;
      std::cout << "[*] Starting Partial Join for first query graph with depth: " << partial_match_depth << std::endl;

      const size_t MAX_PARTIAL_MATCHES = 100000000;
      const size_t partial_match_size = partial_match_depth + 1;

      int* partial_matches = sycl::malloc_shared<int>(MAX_PARTIAL_MATCHES * partial_match_size, queue);
      size_t* num_partial_matches = sycl::malloc_shared<size_t>(1, queue);
      num_partial_matches[0] = 0;


      auto start_cpu_step1 = std::chrono::high_resolution_clock::now();
      
      auto join_e2 = sigmo::isomorphism::join::joinPartialCandidates2(
          queue, device_query_graph, device_data_graph, candidates, partial_match_depth,
          partial_matches, num_partial_matches);
      
      join_e2.wait();
      
      auto end_cpu_step1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_part1 = end_cpu_step1 - start_cpu_step1;

      std::cout << "- Step 1 (Partial Candidates) time: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(time_part1).count() 
                << " ms" << std::endl;

      size_t partial_matches_found = num_partial_matches[0];
      std::cout << "  > Found " << formatNumber(partial_matches_found) << " partial matches." << std::endl;

      const uint32_t num_q = device_query_graph.num_graphs;
      const uint32_t num_d = device_data_graph.num_graphs;

      uint32_t* pair_done = sycl::malloc_shared<uint32_t>(num_q * num_d, queue);
      queue.memset(pair_done, 0, sizeof(uint32_t) * num_q * num_d).wait();

      bool find_first = !args.find_all;


      auto start_cpu_step2 = std::chrono::high_resolution_clock::now();

      auto full_join_e = sigmo::isomorphism::join::joinWithPartialMatches(
          queue, device_query_graph, device_data_graph, candidates, gmcr,
          partial_match_depth, partial_matches, partial_matches_found,
          total_full_matches, find_first, pair_done);
      
      full_join_e.wait();
      
      auto end_cpu_step2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_part2 = end_cpu_step2 - start_cpu_step2;

      std::cout << "- Step 2 (Expand to Full) time: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(time_part2).count() 
                << " ms" << std::endl;

      join_time2 = time_part1 + time_part2;

      sycl::free(pair_done, queue);
      sycl::free(partial_matches, queue);
      sycl::free(num_partial_matches, queue);
    }

    host_time_events.add("join_end");
  }
  std::cout << "[!] End" << std::endl;


  std::cout << "------------- Overall GPU Stats -------------" << std::endl;
  std::chrono::duration<double> total_sig_query_time
      = std::accumulate(query_sig_times.begin(), query_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_sig_data_time
      = std::accumulate(data_sig_times.begin(), data_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_filter_time
      = std::accumulate(filter_times.begin(), filter_times.end(), std::chrono::duration<double>(0));

  std::chrono::duration<double> total_path_sig_query_time
      = std::accumulate(query_path_sig_times.begin(), query_path_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_path_sig_data_time
      = std::accumulate(data_path_sig_times.begin(), data_path_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_path_filter_time
      = std::accumulate(path_filter_times.begin(), path_filter_times.end(), std::chrono::duration<double>(0));

  std::chrono::duration<double> total_cycle_sig_query_time
      = std::accumulate(query_cycle_sig_times.begin(), query_cycle_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_cycle_sig_data_time
      = std::accumulate(data_cycle_sig_times.begin(), data_cycle_sig_times.end(), std::chrono::duration<double>(0));
  std::chrono::duration<double> total_cycle_filter_time
      = std::accumulate(cycle_filter_times.begin(), cycle_filter_times.end(), std::chrono::duration<double>(0));

  std::chrono::duration<double> total_time = total_sig_data_time + total_filter_time + total_sig_query_time
                                           + total_path_sig_data_time + total_path_sig_query_time + total_path_filter_time
                                           + total_cycle_sig_data_time + total_cycle_sig_query_time + total_cycle_filter_time
                                           + join_time2;

  std::cout << "Data (label) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_data_time).count() << " ms" << std::endl;
  std::cout << "Query (label) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_sig_query_time).count() << " ms" << std::endl;
  std::cout << "Data (path) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_path_sig_data_time).count() << " ms" << std::endl;
  std::cout << "Query (path) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_path_sig_query_time).count() << " ms" << std::endl;
  std::cout << "Data (cycle) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_cycle_sig_data_time).count() << " ms" << std::endl;
  std::cout << "Query (cycle) signature time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_cycle_sig_query_time).count() << " ms" << std::endl;
  std::cout << "Filter (label/fused) time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_filter_time).count() << " ms" << std::endl;

  if (args.skip_join) {
    std::cout << "Join time: skipped" << std::endl;
  } else {
    std::cout << "Join time: " << std::chrono::duration_cast<std::chrono::milliseconds>(join_time2).count() << " ms" << std::endl;
  }
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " ms" << std::endl;

  std::cout << "------------- Overall Host Stats -------------" << std::endl;
  std::cout << "Setup Data time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("setup_data_start", "setup_data_end")).count()
            << " ms (not included in total)" << std::endl;
   
  std::cout << "Signature Gen time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("sig_gen_start", "sig_gen_end")).count() << " ms"
            << std::endl;

  std::cout << "Filter & Refine time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("filter_proc_start", "filter_proc_end")).count() << " ms"
            << std::endl;

  if (args.skip_join) {
    std::cout << "Mapping time: skipped" << std::endl;
    std::cout << "Join time: skipped" << std::endl;
  } else {
    std::cout << "Mapping time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("mapping_start", "mapping_end")).count() << " ms"
              << std::endl;
    std::cout << "Join time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getRangeTime("join_start", "join_end")).count() << " ms"
              << std::endl;
  }
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(host_time_events.getTimeFrom("setup_data_end")).count()
            << " ms" << std::endl;


  std::cout << "------------- Results -------------" << std::endl;
  if (!args.skip_print_candidates) {
    CandidatesInspector inspector;
    auto host_candidates = candidates.getHostCandidates();
    for (size_t i = 0; i < query_nodes; ++i) {
      auto count = host_candidates.getCandidatesCount(i);
      inspector.add(count);
      if (args.print_candidates) std::cerr << "Node " << i << ": " << count << std::endl;
    }
    inspector.finalize();
    std::cout << "# Total candidates: " << formatNumber(inspector.total) << std::endl;
    std::cout << "# Average candidates: " << formatNumber(inspector.avg) << std::endl;
    std::cout << "# Median candidates: " << formatNumber(inspector.median) << std::endl;
    std::cout << "# Zero candidates: " << formatNumber(inspector.zero_count) << std::endl;
  }
  if (!args.skip_join) { std::cout << "# Matches: " << formatNumber(total_full_matches[0]) << std::endl; }

  sycl::free(total_full_matches, queue);
  sigmo::destroyDeviceCSRGraph(device_data_graph, queue);
  sigmo::destroyDeviceCSRGraph(device_query_graph, queue);
}
