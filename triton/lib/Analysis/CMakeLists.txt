if(TTGPU)
  set(TRITON_ANALYSIS_UTILITY_CPP "${CMAKE_SOURCE_DIR}/csrc/lib/ttgpu/Analysis/Utility.cpp")
else()
  set(TRITON_ANALYSIS_UTILITY_CPP Utility.cpp)
endif()

add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  # Utility.cpp
  ${TRITON_ANALYSIS_UTILITY_CPP}

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
)
