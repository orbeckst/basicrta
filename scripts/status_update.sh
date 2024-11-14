#!/usr/bin/bash                                                                 
                                                                                
sacct -n -X --format jobname,jobid -s RUNNING > .running_jobs.txt               
awk -v col1=1 -v col2=2 '{printf "basicrta-7.0/%s/slurm-%s.out \n", $col1, $col2}' < .running_jobs.txt | xargs tail -n 1 | sed 's@[^a-z A-Z0-9.:%|</]@@;s/\].*/\]/'; echo
