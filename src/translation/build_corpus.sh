#!/usr/bin/env sh

PID=$$
NICE=19
__FILE__=`basename $0`
__DIR__=`echo $(cd $(dirname $0); pwd)`
__DIR_SCR__=`dirname ${__DIR__}`
__USAGE__="
Usage : ${__FILE__} <OPTION> <dirLog>

  Build a corpus for training NMT with lamtram

<OPTION>
  --dir-data <DIR_DATA>          parallel corpora directory [=paralell_corpora]
  --sources <SOURCES>            sources [=\{BASIC,REIJIRO,Tatoeba,WordNet,YOMIURI\}]
  --out-dir <DIR_OUT>            output director
  --lang-src <LANG>              source language
  --lang-trg <LANG>              target language
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

DIR_TMP=/data/${USERNAME}/${PID}
DIR_DATA=parallel_corpora
DIR_OUT=
LANG_SRC=en
LANG_TRG=ja
SOURCES={BASIC,REIJIRO,Tatoeba,WordNet,YOMIURI}
pathInput1=
pathInput2=
pathOutput1=
pathOutput2=
suffix=

while [ $# -gt 0 ]
do
    case $1 in
    --dir-data)         DIR_DATA=$2 ; shift ;;
    --dir-out)          DIR_OUT=$2 ; shift ;;
    --lang-src)         LANG_SRC=$2 ; shift ;;
    --lang-trg)         LANG_TRG=$2 ; shift ;;
    --sources)          SOURCES=$2 ; shift ;;
    --raw)              suffix=.raw ;;
    --)                 shift ; break ;;
    -*)                 usage 1 ;;
    *)                  break ;;
    esac
    shift
done

if [ -z "$DIR_OUT" ]; then
    logging "ERROR: --dir-out is empty"
    exit(1)
fi

mkdir -p ${DIR_TMP}

# Merge
sources=(`echo ${SOURCES}`)
for corpus in $sources
do
    corpuspath="${DIR_DATA}/${corpus}"
    cmd="cat ${corpuspath}/${LANG_SRC}${suffix} | python3 ${__DIR__}/remove_tab.py >> ${DIR_TMP}/${LANG_SRC}"
    logging ${cmd}
    eval ${cmd}
    cmd="cat ${corpuspath}/${LANG_TRG}${suffix} | python3 ${__DIR__}/remove_tab.py>> ${DIR_TMP}/${LANG_TRG}"
    logging ${cmd}
    eval ${cmd}
done

logging "wc -l ${DIR_TMP}/${LANG_SRC}: `wc -l ${DIR_TMP}/${LANG_SRC} | cut -d ' ' -f 1`"

# Uniq
cmd="nice -n ${NICE} paste ${DIR_TMP}/{${LANG_SRC},${LANG_TRG}} | LC_ALL=C nice -n ${NICE} sort | LC_ALL=C nice -n ${NICE} uniq > ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq"
logging ${cmd}
eval ${cmd}

logging "wc -l ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq: `wc -l ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq | cut -d ' ' -f 1`"

# Shuffle and split
logging "Shuffle and split"
cmd="nice -n ${NICE} python ${__DIR_SCR__}/shuffle.py < ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq > ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0"
logging ${cmd}
eval ${cmd}

head -n 1000 ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0 > ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.dev
sed -e '1,1000d' ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0 > ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.train

logging "wc -l ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.*\n`wc -l ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.*`"

logging "mkdir -p ${DIR_OUT}"
mkdir -p ${DIR_OUT}
logging "Copying *train and *dev"
cut -f 1 ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.dev > ${DIR_OUT}/dev.${LANG_SRC}
cut -f 1 ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.train > ${DIR_OUT}/train.${LANG_SRC}
cut -f 2 ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.dev > ${DIR_OUT}/dev.${LANG_TRG}
cut -f 2 ${DIR_TMP}/${LANG_SRC}${LANG_TRG}.uniq.shuf0.train > ${DIR_OUT}/train.${LANG_TRG}

logging "nice -n ${NICE} xz ${DIR_TMP}/* -v"
nice -n ${NICE} xz ${DIR_TMP}/* -v
logging "mv ${DIR_TMP} ${DIR_OUT}/work"
mv ${DIR_TMP} ${DIR_OUT}/work
