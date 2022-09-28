SCRIPT_DIR=$(cd $(dirname $0); pwd)
for grid_path in $1/*; do 
    if [[ ! "${grid_path}" -ef "${grid_path}/../DEF" ]]; then 
        for cell_path in ${grid_path}/*; do 
            echo ${SCRIPT_DIR}/${cell_path}
            "../build/RTLib/Src/Ext/OPX7/Release/RTLib-Ext-OPX7-Test-Comp.exe" --base_dir ${SCRIPT_DIR}/${cell_path} --base_smp 1000000 --comp_dir ${SCRIPT_DIR}/${grid_path}/../ > ${SCRIPT_DIR}/${cell_path}/result.csv; 
        done
    fi;
done