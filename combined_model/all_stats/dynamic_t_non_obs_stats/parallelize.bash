#!/bin/bash

if [ "$#" != "1" ]; then
    echo
    echo "Usage: $0 [num_procs]"
    echo "   This reads a list of commands that can be executed"
    echo "   in parallel and runs them using parallel make."
    echo
    exit
fi

NPROCS="$1"

TARGET=0
MAKEFILE=".Makefile.stuffer.$$"

echo "all : targets" >> ${MAKEFILE}

while read -r command; do
    echo >> ${MAKEFILE}
    echo "target$((TARGET++)) : " >> ${MAKEFILE}
    echo -en "\tbash -c " >> ${MAKEFILE}
    echo \"${command}\" >> ${MAKEFILE}
done

echo >> ${MAKEFILE}
echo -n "targets : " >> ${MAKEFILE}
for ((x=0;x<${TARGET};x++)); do
    echo -n "target${x} " >> ${MAKEFILE}
done
echo >> ${MAKEFILE}

make -f ${MAKEFILE} -j ${NPROCS}
RETVAL="${PIPESTATUS}"
rm ${MAKEFILE}
exit ${RETVAL}
