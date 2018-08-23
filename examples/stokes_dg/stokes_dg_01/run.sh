#!/bin/bash

#set -x

source $LIBMESH_DIR/examples/run_common.sh

example_name=stokes_dg_01

run_example "$example_name" 
