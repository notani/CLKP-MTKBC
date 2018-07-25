#!/usr/bin/env sh

PID=$$
__FILE__=`basename $0`
__DIR__=`echo $(cd $(dirname $0); pwd)`
__USAGE__="
Usage : ${__FILE__} <OPTION> <dirLog>

  Drop sentence pairs whose length is longer than a threshold

<OPTION>
  --in1, --in2 <pathInput>       Input files
  --out1, --out2 <pathOutput>    Output files
  --length <int>                 length threshold [=50]
"


usage()
{
    echo "${__USAGE__}" 1>&2
    exit $1
}

logging()
{
    local nowExecuteTime=`date +%s`
    local progressTime=`echo $((nowExecuteTime - __STARTTIME__)) | awk '{printf("%02d:%s", $1/3600, strftime("%M:%S",$1))}'`
    if [ "${__LOG__}" ]
    then
        echo "[${__FILE__}] `date +%y/%m/%d,%H:%M:%S` (${progressTime}) $1" >> ${__LOG__}
    else
        echo "[${__FILE__}] `date +%y/%m/%d,%H:%M:%S` (${progressTime}) $1"
    fi
}

DIR_TMP=/data/`whoami`/$PID
mkdir -p $DIR_TMP

pathInput1=
pathInput2=
pathOutput1=
pathOutput2=
length=50

while [ $# -gt 0 ]
do
    case $1 in
    --in1)              pathInput1=$2 ; shift ;;
    --in2)              pathInput2=$2 ; shift ;;
    --out1)             pathOutput1=$2 ; shift ;;
    --out2)             pathOutput2=$2 ; shift ;;
    --length)           length=$2 ; shift ;;
    --)                 shift ; break ;;
    -*)                 usage 1 ;;
    *)                  break ;;
    esac
    shift
done

if [ -z "$pathInput1" ]; then
    logging "ERROR: --in1 is empty"
    exit(1)
fi
if [ -z "$pathInput2" ]; then
    logging "ERROR: --in2 is empty"
    exit(1)
fi
if [ -z "$pathOutput1" ]; then
    logging "ERROR: --out1 is empty"
    exit(1)
fi
if [ -z "$pathOutput2" ]; then
    logging "ERROR: --out2 is empty"
    exit(1)
fi

NICE=19

logging "paste $pathInput1 $pathInput2 | nice -n $NICE python ${__DIR__}/drop_long_sentences.py $length -v > $DIR_TMP/result"
paste $pathInput1 $pathInput2 | nice -n $NICE python ${__DIR__}/drop_long_sentences.py $length -v > $DIR_TMP/result
logging "cut -f 1 $DIR_TMP/result > $pathOutput1"
cut -f 1 $DIR_TMP/result > $pathOutput1
logging "cut -f 2 $DIR_TMP/result > $pathOutput2"
cut -f 2 $DIR_TMP/result > $pathOutput2

rm -r $DIR_TMP
