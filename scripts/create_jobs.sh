#!/usr/bin/bash                                                                 
                                                                                
if [[ $1 = "--rerun" ]];                                                        
then                                                                            
    echo "submitting rerun jobs"                                                
    #./get_rerun_residues.py                                                    
    IFS=,                                                                       
    read -r -a residues < rerun_residues.csv                                    
else                                                                            
    echo "submitting all jobs"                                                  
    IFS=,                                                                       
    read -r -a residues < residue_list.csv                                      
fi                                                                              
                                                                                
                                                                                
for res in ${residues[@]}; do                                                   
    sed -r "s/RESIDUE/${res}/g;s/RESID/${res:1}/g" submit_tmp.slu > submit.slu;
    sbatch submit.slu                                                           
    done                                                                        

