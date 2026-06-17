if(TTGPU)
  set(TRITON_TT2TTG_PASS "${CMAKE_SOURCE_DIR}/csrc/lib/ttgpu/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp")
else()
  set(TRITON_TT2TTG_PASS TritonToTritonGPUPass.cpp)
endif()

add_triton_library(TritonToTritonGPU
    TritonGPUConversion.cpp
    # TritonToTritonGPUPass.cpp
    ${TRITON_TT2TTG_PASS}

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    $<$<BOOL:${TTGPU}>:MLIRMathTransforms>
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
