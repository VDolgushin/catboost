# Generated by devtools/yamaker from nixpkgs 21.11.

LIBRARY()



VERSION(5.3.0)

ORIGINAL_SOURCE(https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2)

LICENSE(
    BSD-2-Clause AND
    Public-Domain
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

ADDINCL(
    contrib/libs/jemalloc/include
    contrib/libs/libunwind/include
)

IF (OS_WINDOWS)
    ADDINCL(
        contrib/libs/jemalloc/include/msvc_compat
    )
ELSE()
    CFLAGS(
        -funroll-loops
    )
    IF (OS_DARWIN OR OS_IOS)
        SRCS(
            GLOBAL reg_zone.cpp
            src/zone.c
        )
    ELSE()
        PEERDIR(
            contrib/libs/libunwind
        )
        CFLAGS(
            -fvisibility=hidden
        )
    ENDIF()
ENDIF()

NO_COMPILER_WARNINGS()

NO_UTIL()

SRCS(
    src/arena.c
    src/background_thread.c
    src/base.c
    src/bin.c
    src/bin_info.c
    src/bitmap.c
    src/buf_writer.c
    src/cache_bin.c
    src/ckh.c
    src/counter.c
    src/ctl.c
    src/decay.c
    src/div.c
    src/ecache.c
    src/edata.c
    src/edata_cache.c
    src/ehooks.c
    src/emap.c
    src/eset.c
    src/exp_grow.c
    src/extent.c
    src/extent_dss.c
    src/extent_mmap.c
    src/fxp.c
    src/hook.c
    src/hpa.c
    src/hpa_hooks.c
    src/hpdata.c
    src/inspect.c
    src/jemalloc.c
    src/jemalloc_cpp.cpp
    src/large.c
    src/log.c
    src/malloc_io.c
    src/mutex.c
    src/nstime.c
    src/pa.c
    src/pa_extra.c
    src/pac.c
    src/pages.c
    src/pai.c
    src/peak_event.c
    src/prof.c
    src/prof_data.c
    src/prof_log.c
    src/prof_recent.c
    src/prof_stats.c
    src/prof_sys.c
    src/psset.c
    src/rtree.c
    src/safety_check.c
    src/san.c
    src/san_bump.c
    src/sc.c
    src/sec.c
    src/stats.c
    src/sz.c
    src/tcache.c
    src/test_hooks.c
    src/thread_event.c
    src/ticker.c
    src/tsd.c
    src/witness.c
)

END()
