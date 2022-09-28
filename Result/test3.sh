SCRIPT_DIR=$(cd $(dirname $0); pwd)
"${SCRIPT_DIR}/../build/RTLib/Src/Ext/OPX7/Release/RTLib-Ext-OPX7-Test-Comp.exe" --base_dir ${SCRIPT_DIR}/$1 --base_smp 1000000  > ${SCRIPT_DIR}/$1/result.csv