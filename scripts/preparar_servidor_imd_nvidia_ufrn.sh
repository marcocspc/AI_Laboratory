#!/bin/bash

#carregar git
module load softwares/git/2.11.0-gnu-4.4

#carregar python 3 no qual esta instalado o tensorflow
module load softwares/python/3.6-anaconda-5.0.1

#carregar virtualenv na qual o tensorflow FUNCIONA
source activate iaPy3

#carregar glibc
module load libraries/glibc/2.14-pre-comp

#carregar cmake
module load softwares/cmake/3.6.2

#visualizar bibliotecas disponíveis
module ava -t |& grep -i <biblioteca>
