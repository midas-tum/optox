# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 01.07.2018

project(optoX)

##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/python)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/python)

##----------------------------------------------------------------------------
## compile using python setup
##----------------------------------------------------------------------------
find_program(PYTHON "python")

message(STATUS "python: ${PYTHON}")

##-----------------------------------------------------------------------------
## define macro to finish lib
##-----------------------------------------------------------------------------
macro (add_deps_py opname)
set(DEPS_PY
    ${DEPS_PY}
    optopy/${opname}.py
    py_${opname}_operator.cpp
)
endmacro (add_deps_py)

# nabla & nabla2
add_deps_py(nabla)
set(DEPS_PY
    ${DEPS_PY}
    optopy/nabla2.py
)

# other
add_deps_py(warp)
add_deps_py(pad)
add_deps_py(maxpooling)
add_deps_py(averagepooling)
add_deps_py(scale)

if(WITH_GPUNUFFT)
    add_deps_py(gpunufft)
    set(DEPS_PY
    ${DEPS_PY}
    py_gpunufft_singlecoil_operator.cpp
)
ENDIF(WITH_GPUNUFFT)

if (PYTHON)
    set(SETUP_PY_IN setup.py.in)
    set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS_PY
        optopy/__init__.py
        py_utils.h
        ${DEPS_PY}
    )

    set(OUTPUT_PY "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT_PY}
                       COMMAND ${PYTHON} ${SETUP_PY} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_PY}
                       DEPENDS ${DEPS_PY})

    add_custom_target(setup_py ALL DEPENDS ${OUTPUT_PY} ${DEPS_PY} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_dependencies(setup_py optox)

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
endif()
