SCRIPT_DIR=$(cd $(dirname $0); pwd)
for grid_path in $1/*; do 
    if [[ ! "${grid_path}" -ef "${grid_path}/../DEF" ]] && [ -d ${grid_path} ]; then 
        "${SCRIPT_DIR}/../build/RTLib/Src/Ext/OPX7/Release/RTLib-Ext-OPX7-Test-Comp.exe" --base_dir ${SCRIPT_DIR}/${grid_path} --base_smp 1000000 --comp_dir ${SCRIPT_DIR}/${grid_path}/.. > ${SCRIPT_DIR}/${grid_path}/result.csv; 
    fi;
done