#!/bin/bash

echo "Tenha certeza de que está na pasta do arquivo .simg ao executar este script, pois ele usa o pwd para localizar os arquivos."

dir=$(pwd)

sudo singularity build "$dir/urnai_full_singularity_container.sif" "$dir/urnai_full.simg"
