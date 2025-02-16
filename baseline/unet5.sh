#!/bin/bash
rm -rf models/unet5/
rm -rf tblogs/unet5/
julia +lts unet5.jl 0 400 false
