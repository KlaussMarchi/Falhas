RUN=/home/grva-mint/Projects/Falhas/Marlim/Regions/Synthetic/files/run
PY=/home/grva-mint/anaconda3/envs/torch-gpu/bin/python

exec 9> $RUN/lock
flock -n 9 || exit 0
[ -f $RUN/DONE ] && exit 0

cd /home/grva-mint/Projects/Falhas/Marlim/Regions/Synthetic
echo "$(date '+%F %T') início" >> $RUN/history.log
chrt --idle 0 $PY -m papermill Analysis.ipynb $RUN/Analysis_run.ipynb --cwd . --log-output --autosave-cell-every 60 > $RUN/papermill.log 2>&1
code=$?
echo "$(date '+%F %T') fim, código $code" >> $RUN/history.log

if [ $code -eq 0 ]; then
    touch $RUN/DONE
    crontab -l | grep -v "files/run/resume.sh" | crontab -
fi
