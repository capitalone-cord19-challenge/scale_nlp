DATE=2020-04-24
DATA_DIR=./cord19-"${DATE}"

sh target/appassembler/bin/IndexCollection \
  -collection CovidParagraphCollection -generator CovidGenerator \
  -threads 8 -input "${DATA_DIR}" \
  -index "${DATA_DIR}"/lucene-index-cord19-paragraph-"${DATE}" \
  -storePositions -storeDocvectors -storeContents -storeRaw > log.cord19-paragraph.${DATE}.txt