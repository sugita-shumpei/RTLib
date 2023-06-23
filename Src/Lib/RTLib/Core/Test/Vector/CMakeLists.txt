add_executable(RTLib-Core-Test-Vector 
	Test_Vector.cpp
)

target_link_libraries(RTLib-Core-Test-Vector gtest_main RTLib-Core)
add_test(NAME RTLib-Core-Test-Vector COMMAND RTLib-Core-Test-Vector)
enable_testing()

add_custom_command(TARGET RTLib-Core-Test-Vector POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:RTLib-Core-Test-Vector> $<TARGET_RUNTIME_DLLS:RTLib-Core-Test-Vector>
  COMMAND_EXPAND_LISTS
)