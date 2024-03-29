#!/bin/bash

echo "Tenha certeza de que está na pasta do arquivo .simg ao executar este script, pois ele usa o pwd para localizar os arquivos."

dir=$(pwd)

mkdir $dir/tmp
export SINGULARITY_TMPDIR="$dir/tmp"
export TMPDIR="$dir/tmp"
sudo -E singularity build "$dir/urnai_default_singularity_container_$(date +%F).sif" "$dir/urnai_default.simg"
