# Installed as lib/cmake/swxbx/swxbxConfig.cmake inside a release archive.
get_filename_component(_swxbx_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT TARGET swxbx::swxbx)
  add_library(swxbx::swxbx SHARED IMPORTED)
  set_target_properties(swxbx::swxbx PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_swxbx_prefix}/include")
  if(WIN32)
    set_target_properties(swxbx::swxbx PROPERTIES
      IMPORTED_LOCATION "${_swxbx_prefix}/lib/swxbx.dll"
      IMPORTED_IMPLIB "${_swxbx_prefix}/lib/swxbx.lib")
  else()
    set_target_properties(swxbx::swxbx PROPERTIES
      IMPORTED_LOCATION "${_swxbx_prefix}/lib/libswxbx.so")
  endif()
endif()

unset(_swxbx_prefix)
