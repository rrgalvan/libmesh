BUILT_SOURCES = mesh.xda

mesh.xda: $(top_srcdir)/examples/adaptivity/adaptivity_ex2/mesh.xda
	$(AM_V_GEN)$(LN_S) -f $< $@

DISTCLEANFILES = $(BUILT_SOURCES)

example_name  = keller_segel_ex3
check_SCRIPTS = run.sh
install_dir   = $(examples_install_path)/keller_segel/ex3
data          = keller_segel_ex3.C exact_solution.C keller_segel_ex3.in ../../adaptivity/adaptivity_ex2/mesh.xda run.sh
sources       = $(data) run.sh

CLEANFILES = out_*.gmv keller_segel_ex3.e

##############################################
# include common example environment
include $(top_srcdir)/examples/Make.common
