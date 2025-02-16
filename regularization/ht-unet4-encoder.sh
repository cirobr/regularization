#!/bin/bash
script_name=$(basename "$0" .sh)
rm "${script_name}.csv"
julia -t1 "${script_name}.jl" 0 100 false
