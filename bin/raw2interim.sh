for f in data/raw/*.pdf; do
    echo $f
    pdftotext -enc ASCII7 $f data/interim/$(basename -- ${f%.*}).txt
done
