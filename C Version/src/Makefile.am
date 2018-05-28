lib_LTLIBRARIES = libccl.la

libccl_la_SOURCES = ccl_learn_nhat.c \
                    ccl_learn_alpha.c\
                    ccl_learn_lambda.c\
                    ccl_learn_ncl.c\
                    ccl_learn_policy.c\
                    ccl_math.c

libccl_la_CFLAGS = -I$(top_srcdir)/include
libccl_la_LIBADD =  -lgsl -lgslcblas -lm
libccl_la_LDFLAGS = -version-info 1:1:0

bin_PROGRAMS = ccl
ccl_SOURCES = ccl.c
ccl_LDADD   = libccl.la -lm
ccl_CFLAGS  = -I$(top_srcdir)/include
