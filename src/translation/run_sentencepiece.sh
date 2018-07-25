#!/usr/bin/env sh

PID=$$
NICE=19
__FILE__=`basename $0`
__DIR__=`echo $(cd $(dirname $0); pwd)`
__DIR_SCR__=`dirname ${__DIR__}`
__USAGE__="
Usage : ${__FILE__} <OPTION> <dirLog>

  Build a BPE model and parse corpora using sentencepiece

  To install sentencepiece, see the github page:
  https://github.com/google/sentencepiece

<OPTION>
  --dir-data <DIR_DATA>          parallel corpora directory
  --size <VOCAB_SIZE>            sources [=8003]
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
DIR_DATA=
LANG_SRC=en
LANG_TRG=ja
pathInput1=
pathInput2=
pathOutput1=
pathOutput2=
modelType=
suffix=
PYTHON=python
VOCAB_SIZE=8003

while [ $# -gt 0 ]
do
    case $1 in
    --dir-data)         DIR_DATA=$2 ; shift ;;
    --lang-src)         LANG_SRC=$2 ; shift ;;
    --lang-trg)         LANG_TRG=$2 ; shift ;;
    --size)             VOCAB_SIZE=$2 ; shift ;;
    --python)           PYTHON=$2 ; shift ;;
    --model-type)       modelType=$2 ; shift ;;
    --)                 shift ; break ;;
    -*)                 usage 1 ;;
    *)                  break ;;
    esac
    shift
done

if [ -z "$DIR_DATA" ]; then
    logging "ERROR: --dir-data is empty"
    exit(1)
fi

logging "mkdir -p ${DIR_TMP}"
mkdir -p ${DIR_TMP}

for lang in ${LANG_TRG} ${LANG_SRC}
do
    MODEL_PREFIX=${lang}.${VOCAB_SIZE}
    if [ ! -e "${DIR_DATA}/sp/${MODEL_PREFIX}.model" ]; then
        # Build a model
        cmd="nice -n ${NICE} spm_train --input=${DIR_DATA}/train.${lang} --model_prefix=${DIR_TMP}/${MODEL_PREFIX} --vocab_size=${VOCAB_SIZE} --model_type=unigram"
        if [ -n "${modelType}" ]; then
            cmd="${cmd} --model_type ${modelType}"
        fi
        logging $cmd
        eval $cmd
        logging "mkdir -p ${DIR_DATA}/sp && cp ${DIR_TMP}/${MODEL_PREFIX}.* ${DIR_DATA}/sp"
        mkdir -p ${DIR_DATA}/sp && mv ${DIR_TMP}/${MODEL_PREFIX}.* ${DIR_DATA}/sp
    fi

    # Encode
    logging "nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=id < ${DIR_DATA}/train.${lang} > ${DIR_TMP}/train.${lang}.sp${VOCAB_SIZE}.id"
    nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=id < ${DIR_DATA}/train.${lang} > ${DIR_TMP}/train.${lang}.sp${VOCAB_SIZE}.id
    logging "nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=piece < ${DIR_DATA}/train.${lang} > ${DIR_TMP}/train.${lang}.sp${VOCAB_SIZE}.piece"
    nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=piece < ${DIR_DATA}/train.${lang} > ${DIR_TMP}/train.${lang}.sp${VOCAB_SIZE}.piece
    logging "nice -n ${NICE} ${PYTHON} ${__DIR__}/replace_piece_with_unk.py --id ${DIR_DATA}/train.${lang}.sp${VOCAB_SIZE}.id --pi ${DIR_DATA}/train.${lang}.sp${VOCAB_SIZE}.piece -v > ${DIR_TMP}/train.${lang}.sp${VOCAB_SIZE}"
    nice -n ${NICE} ${PYTHON} ${__DIR__}/replace_piece_with_unk.py --id ${DIR_TMP}/train.${lang}.sp${VOCAB_SIZE}.id --pi ${DIR_TMP}/train.${lang}.sp${VOCAB_SIZE}.piece -v > ${DIR_DATA}/train.${lang}.sp${VOCAB_SIZE}

    logging "nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=id < ${DIR_DATA}/dev.${lang} > ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.id"
    nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=id < ${DIR_DATA}/dev.${lang} > ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.id
    logging "nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=piece < ${DIR_DATA}/dev.${lang} > ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.piece"
    nice -n ${NICE} spm_encode --model=${DIR_DATA}/sp/${MODEL_PREFIX}.model --output_format=piece < ${DIR_DATA}/dev.${lang} > ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.piece
    logging "nice -n ${NICE} ${PYTHON} ${__DIR__}/replace_piece_with_unk.py --id ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.id --pi ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.piece -v > ${DIR_DATA}/dev.${lang}.sp${VOCAB_SIZE}"
    nice -n ${NICE} ${PYTHON} ${__DIR__}/replace_piece_with_unk.py --id ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.id --pi ${DIR_TMP}/dev.${lang}.sp${VOCAB_SIZE}.piece -v > ${DIR_DATA}/dev.${lang}.sp${VOCAB_SIZE}
done

logging "rm -r ${DIR_TMP}"
rm -r ${DIR_TMP}
