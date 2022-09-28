arr=( "DEF" "RIS" "PGRIS" "HTDEF" "HTRIS" )
enableGrid="false"
enableTree="false"
for Tracer in ${arr[@]};do
    enableGrid="false"
    enableTree="false"
    if [ $Tracer = "PGDEF" ] || [ $Tracer = "PGRIS" ]; then
        echo "$Tracer: --EnableTree true"
        enableTree="true"
    fi
    if [ $Tracer = "HTDEF" ] || [ $Tracer = "HTRIS" ]; then
        echo "$Tracer: --EnableGrid true"
        enableGrid="true"
    fi
    echo "$Tracer $(( 1))"
    ./"/../build/RTLib/Src/Ext/OPX7/Release/RTLib-Ext-OPX7-Test-Main" --DefTracer $Tracer --SamplesPerSave 1 --MaxSamples 1 --EnableGrid ${enableGrid} --EnableTree ${enableTree}
    for((i=0;i<5;i++));do
        echo "$Tracer: EnableGrid: ${enableGrid} EnableTree: ${enableTree}"
        echo "$Tracer: $(( 5*10**$i))"
        ./"/../build/RTLib/Src/Ext/OPX7/Release/RTLib-Ext-OPX7-Test-Main" --DefTracer $Tracer --SamplesPerSave $((5*10**$i)) --MaxSamples $((5*10**$i)) --EnableGrid ${enableGrid} --EnableTree ${enableTree}
        echo "$Tracer: EnableGrid: ${enableGrid} EnableTree: ${enableTree}"
        echo "$Tracer: $((10*10**$i))"
        ./"/../build/RTLib/Src/Ext/OPX7/Release/RTLib-Ext-OPX7-Test-Main" --DefTracer $Tracer --SamplesPerSave $((10*10**$i)) --MaxSamples $((10*10**$i)) --EnableGrid ${enableGrid} --EnableTree ${enableTree}
    done
done