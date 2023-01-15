RESOURCES_LIBRARY()



INCLUDE(${ARCADIA_ROOT}/build/platform/python/resources.inc)

IF (OS_LINUX)
    DECLARE_EXTERNAL_RESOURCE(EXTERNAL_PYTHON37 sbr:${PYTHON37_LINUX})
ELSEIF (OS_DARWIN)
    DECLARE_EXTERNAL_RESOURCE(EXTERNAL_PYTHON37 sbr:${PYTHON37_DARWIN})
ELSEIF (OS_WINDOWS)
    DECLARE_EXTERNAL_RESOURCE(EXTERNAL_PYTHON37 sbr:${PYTHON37_WINDOWS})
ENDIF()

END()
