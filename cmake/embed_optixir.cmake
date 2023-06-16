## Adapted from OWL: https://github.com/owl-project/owl/blob/master/owl/cmake/embed_optixir.cmake 
## Copyright 2021 Jefferson Amstutz
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.27)

# NOTE(jda) - CMake 3.17 defines CMAKE_CURRENT_FUNCTION_LIST_DIR, but alas can't
#             use it yet.
set(EMBED_OPTIXIR_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")

# embed_optixir
#
# Compile CUDA sources to OPTIXIR and use bin2c from the CUDA SDK to create data arrays
# containing the resulting OPTIXIR output.
#
# Keyword arguments:
# CONST             Pass --const to bin2c to generate constant data arrays.
# RELOCATABLE       Pass -rdc=true to nvcc to generate relocatable OPTIXIR.
#
# Single value arguments:
# OUTPUT_TARGET     Name of the target that contains the generated C file.
#                   Required.
# OPTIXIR_TARGET    Name of the target that compiles CUDA to OPTIXIR.
#                   Default: ${OUTPUT_TAGET}_optixir
# FOLDER            IDE folder property for generated targets, if any.
# HEADER            Generate a header file with the given name to contain
#                   declarations for the generated data arrays.
#
# Multiple value arguments:
# OPTIXIR_INCLUDE_DIRECTORIES   List of directories to search when compiling to OPTIXIR.
# OPTIXIR_LINK_LIBRARIES        List of libraries to link against when compiling OPTIXIR.
# SOURCES                       List of CUDA source files to compile to OPTIXIR.
# EMBEDDED_SYMBOL_NAMES         List of names for embedded data arrays, one per source file.
#
function(embed_optixir)
  set(noArgs CONST RELOCATABLE)
  set(oneArgs OUTPUT_TARGET OPTIXIR_TARGET FOLDER HEADER)
  set(multiArgs OPTIXIR_INCLUDE_DIRECTORIES OPTIXIR_LINK_LIBRARIES SOURCES EMBEDDED_SYMBOL_NAMES)
  cmake_parse_arguments(EMBED_OPTIXIR "${noArgs}" "${oneArgs}" "${multiArgs}" ${ARGN})

  if(NOT EMBED_OPTIXIR_OUTPUT_TARGET)
    message(FATAL_ERROR "Missing required OUTPUT_TARGET argument")
  endif()

  if(EMBED_OPTIXIR_EMBEDDED_SYMBOL_NAMES)
    list(LENGTH EMBED_OPTIXIR_EMBEDDED_SYMBOL_NAMES NUM_NAMES)
    list(LENGTH EMBED_OPTIXIR_SOURCES NUM_SOURCES)
    if (NOT ${NUM_SOURCES} EQUAL ${NUM_NAMES})
      message(FATAL_ERROR
        "embed_optixir(): the number of names passed as EMBEDDED_SYMBOL_NAMES must \
        match the number of files in SOURCES."
      )
    endif()
  else()
    unset(EMBED_OPTIXIR_EMBEDDED_SYMBOL_NAMES)
    foreach(source ${EMBED_OPTIXIR_SOURCES})
      get_filename_component(name ${source} NAME_WE)
      list(APPEND EMBED_OPTIXIR_EMBEDDED_SYMBOL_NAMES ${name}_optixir)
    endforeach()
  endif()

  ## Find bin2c and CMake script to feed it ##

  # We need to wrap bin2c with a script for multiple reasons:
  #   1. bin2c only converts a single file at a time
  #   2. bin2c has only standard out support, so we have to manually redirect to
  #      a cmake buffer
  #   3. We want to pack everything into a single output file, so we need to use
  #      the --name option

  get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  find_program(BIN_TO_C NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
  if(NOT BIN_TO_C)
    message(FATAL_ERROR
      "bin2c not found:\n"
      "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
      "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
      )
  endif()

  set(EMBED_OPTIXIR_RUN ${EMBED_OPTIXIR_DIR}/run_bin2c.cmake)

  ## Create OPTIXIR object target ##

  if (NOT EMBED_OPTIXIR_OPTIXIR_TARGET)
    set(OPTIXIR_TARGET ${EMBED_OPTIXIR_OUTPUT_TARGET}_optixir)
  else()
    set(OPTIXIR_TARGET ${EMBED_OPTIXIR_OPTIXIR_TARGET})
  endif()

  add_library(${OPTIXIR_TARGET} OBJECT)
  target_sources(${OPTIXIR_TARGET} PRIVATE ${EMBED_OPTIXIR_SOURCES})
  target_include_directories(${OPTIXIR_TARGET} PRIVATE ${EMBED_OPTIXIR_OPTIXIR_INCLUDE_DIRECTORIES})
  target_link_libraries(${OPTIXIR_TARGET} PRIVATE ${EMBED_OPTIXIR_OPTIXIR_LINK_LIBRARIES})
  set_property(TARGET ${OPTIXIR_TARGET} PROPERTY CUDA_OPTIX_COMPILATION ON)
  target_compile_options(${OPTIXIR_TARGET} PRIVATE "-D_DEBUG=1 -lineinfo -G -O0")
  if(EMBED_OPTIXIR_RELOCATABLE)
    target_compile_options(${OPTIXIR_TARGET} PRIVATE "-rdc=true")
  endif()
  if(EMBED_OPTIXIR_FOLDER)
    set_property(TARGET ${OPTIXIR_TARGET} PROPERTY FOLDER ${EMBED_OPTIXIR_FOLDER})
  endif()

  ## Create command to run the bin2c via the CMake script ##

  set(EMBED_OPTIXIR_C_FILE ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_OPTIXIR_OUTPUT_TARGET}/${EMBED_OPTIXIR_OUTPUT_TARGET}.c)
  if(EMBED_OPTIXIR_HEADER)
    set(EMBED_OPTIXIR_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_OPTIXIR_OUTPUT_TARGET}/${EMBED_OPTIXIR_HEADER})
  endif()
  get_filename_component(OUTPUT_FILE_NAME ${EMBED_OPTIXIR_C_FILE} NAME)
  if(EMBED_OPTIXIR_HEADER)
    add_custom_command(
      OUTPUT ${EMBED_OPTIXIR_C_FILE} ${EMBED_OPTIXIR_HEADER}
      COMMAND ${CMAKE_COMMAND}
        "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
        "-DOBJECTS=$<TARGET_OBJECTS:${OPTIXIR_TARGET}>"
        "-DSYMBOL_NAMES=${EMBED_OPTIXIR_EMBEDDED_SYMBOL_NAMES}"
        "-DOUTPUT=${EMBED_OPTIXIR_C_FILE}"
        "-DCONST=${EMBED_OPTIXIR_CONST}"
        "-DHEADER=${EMBED_OPTIXIR_HEADER}"
        -P ${EMBED_OPTIXIR_RUN}
      VERBATIM
      DEPENDS $<TARGET_OBJECTS:${OPTIXIR_TARGET}> ${OPTIXIR_TARGET}
      COMMENT "Generating embedded OPTIXIR file: ${OUTPUT_FILE_NAME}"
    )
  else()
    add_custom_command(
      OUTPUT ${EMBED_OPTIXIR_C_FILE}
      COMMAND ${CMAKE_COMMAND}
        "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
        "-DOBJECTS=$<TARGET_OBJECTS:${OPTIXIR_TARGET}>"
        "-DSYMBOL_NAMES=${EMBED_OPTIXIR_EMBEDDED_SYMBOL_NAMES}"
        "-DOUTPUT=${EMBED_OPTIXIR_C_FILE}"
        "-DCONST=${EMBED_OPTIXIR_CONST}"
        "-DHEADER=${EMBED_OPTIXIR_HEADER}"
        -P ${EMBED_OPTIXIR_RUN}
      VERBATIM
      DEPENDS $<TARGET_OBJECTS:${OPTIXIR_TARGET}> ${OPTIXIR_TARGET}
      COMMENT "Generating embedded OPTIXIR file: ${OUTPUT_FILE_NAME}"
    )
  endif()

  add_library(${EMBED_OPTIXIR_OUTPUT_TARGET} OBJECT)
  target_sources(${EMBED_OPTIXIR_OUTPUT_TARGET} PRIVATE ${EMBED_OPTIXIR_C_FILE})
  if(EMBED_OPTIXIR_HEADER)
    target_sources(${EMBED_OPTIXIR_OUTPUT_TARGET} PRIVATE ${EMBED_OPTIXIR_HEADER})
    target_include_directories(${EMBED_OPTIXIR_OUTPUT_TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_OPTIXIR_OUTPUT_TARGET})
  endif()
  if(EMBED_OPTIXIR_FOLDER)
    set_property(TARGET ${EMBED_OPTIXIR_OUTPUT_TARGET} PROPERTY FOLDER ${EMBED_OPTIXIR_FOLDER})
  endif()
endfunction()
