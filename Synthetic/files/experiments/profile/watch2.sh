#!/bin/bash
# PERFIL (RÉGUA) E VISIBILIDADE DO RÓTULO DE CADA DATASET DE EXPERIMENTO ASSIM QUE O DataBase.csv DELE APARECE; PARA SOZINHO QUANDO
# NÃO HÁ MAIS GERAÇÃO
cd /home/grva-mint/Projects/Falhas/Synthetic/files/experiments/profile
while true; do
    for csv in ../datasets/*/DataBase.csv; do
        tag=$(basename $(dirname $csv))
        [ -f "$tag.pkl" ] || nice -n 10 ~/anaconda3/envs/torch-gpu/bin/python run_profile.py $tag
        grep -q "^$tag," visibility.csv 2>/dev/null || nice -n 10 ~/anaconda3/envs/torch-gpu/bin/python visibility.py $tag 2>&1 | grep -v Warning
    done
    ps -eo args | grep -v grep | grep -q "make_dataset.py" || break
    sleep 600
done
echo fim
