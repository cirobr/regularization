#!/bin/bash
rm -rf models/unet5/
rm -rf tblogs/unet5/
julia +lts unet5.jl 1 400 false
