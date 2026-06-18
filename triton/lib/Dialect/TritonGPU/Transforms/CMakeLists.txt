if(TTGPU)
  set(TRITONGPU_TRANSFORMS 
    "${CMAKE_SOURCE_DIR}/csrc/lib/ttgpu/Dialect/TritonGPU/Transforms/Coalesce.cpp"
    "${CMAKE_SOURCE_DIR}/csrc/lib/ttgpu/Dialect/TritonGPU/Transforms/ReduceDataDuplication.cpp"
    "${CMAKE_SOURCE_DIR}/csrc/lib/ttgpu/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp"
    "${CMAKE_SOURCE_DIR}/csrc/lib/ttgpu/Dialect/TritonGPU/Transforms/Utility.cpp"
  )
else()
  set(TRITONGPU_TRANSFORMS
    Coalesce.cpp
    ReduceDataDuplication.cpp
    RemoveLayoutConversions.cpp
    Utility.cpp
  )
endif()

add_triton_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  # Coalesce.cpp
  F32DotTC.cpp
  CombineTensorSelectAndIf.cpp
  # ReduceDataDuplication.cpp
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/OuterLoopPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Pipeliner/TMAStoresPipeline.cpp
  Pipeliner/PipeliningUtility.cpp
  Prefetch.cpp
  # RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  # Utility.cpp
  ${TRITONGPU_TRANSFORMS}

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
