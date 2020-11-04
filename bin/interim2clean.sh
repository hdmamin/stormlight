for f in data/interim/*.txt; do
    echo $f
    sed 's//xxbrxx/' $f > data/clean/$(basename -- ${f%.*}).txt
done
