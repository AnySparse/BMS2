
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sigmo {


enum class CandidatesDomain { Query, Data };
namespace device {

static struct DeviceOptions {
  size_t join_work_group_size = 128;
  size_t filter_work_group_size = 512;

} deviceOptions;

inline size_t getDeviceMemorySize(sycl::queue& queue) {
  return queue.get_info<sycl::info::queue::device>().get_info<sycl::info::device::global_mem_size>();
}

inline size_t getPreferredWorkGroupSize(sycl::queue& queue) { return queue.get_device().get_info<sycl::info::device::max_work_group_size>(); }

namespace kernels {

class PrefixSumKernel;
class GenerateQuerySignaturesKernel;
class RefineQuerySignaturesKernel;
class GenerateDataSignaturesKernel;
class RefineDataSignaturesKernel;

class GenerateCSRPathSignaturesKernel;
class GenerateCSRCycleSignaturesKernel;
template<CandidatesDomain D>
class RefineCandidatesByPathKernel;
template<CandidatesDomain D>
class RefineCandidatesByCycleKernel;

template<CandidatesDomain D>
class FilterCandidatesKernel;
class FilterCandidatesKernel_v2;
template<CandidatesDomain D>
class RefineCandidatesKernel;
class RefineCandidatesKernel_v2;
class RefineCandidatesKernel_v3;
class JoinCandidatesKernel;
class JoinWildcardCandidatesKernel;
class JoinCandidates2Kernel;
class JoinCandidates3Kernel;
class JoinCandidates4Kernel;
class JoinCandidates5Kernel;
class JoinPartialCandidatesKernel;
class JoinPartialCandidates2Kernel;
class JoinWithPartialMatchesKernel;
class JoinWithPartialMatchesKernel_v2;
class JoinWithPartialMatchesKernel_v3;
} // namespace kernels

namespace memory {

enum class MemoryScope {
  Device,
  Host,
  Shared,
};

#if SIGMO_ALLOCATION == 0
constexpr MemoryScope default_location = MemoryScope::Host;
#elif SIGMO_ALLOCATION == 1
constexpr MemoryScope default_location = MemoryScope::Device;
#elif SIGMO_ALLOCATION == 2
constexpr MemoryScope default_location = MemoryScope::Shared;
#else
#error "Invalid GRAPH_LOCATION value. Must be 0 (host), 1 (device), or 2 (shared)."
#endif

template<typename T>
inline T* malloc(size_t count, sycl::queue& queue, MemoryScope scope = default_location) {
  if (scope == MemoryScope::Device) {
    return sycl::malloc_device<T>(count, queue);
  } else if (scope == MemoryScope::Host) {
    return sycl::malloc_host<T>(count, queue);
  } else if (scope == MemoryScope::Shared) {
    return sycl::malloc_shared<T>(count, queue);
  }
  throw std::runtime_error("Invalid memory scope");
}

} // namespace memory

} // namespace device
} // namespace sigmo
