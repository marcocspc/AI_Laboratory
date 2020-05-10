#!/bin/bash

echo "Tenha certeza de que está na pasta do arquivo .simg ao executar este script, pois ele usa o pwd para localizar os arquivos."

dir=$(pwd)

docker run -e TZ=America/Fortaleza --privileged -v $dir:/mnt singularity build /mnt/urnai_full_singularity_container.sif /mnt/urnai_full.simg
