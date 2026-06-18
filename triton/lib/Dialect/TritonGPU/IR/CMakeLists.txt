if(TTGPU)
  set(TRITONGPU_IR_DIALECT "${CMAKE_SOURCE_DIR}/csrc/lib/ttgpu/Dialect/TritonGPU/IR/Dialect.cpp")
else()
  set(TRITONGPU_IR_DIALECT Dialect.cpp)
endif()

add_triton_library(TritonGPUIR
  # Dialect.cpp
  ${TRITONGPU_IR_DIALECT}
  LinearLayoutConversions.cpp
  Types.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
