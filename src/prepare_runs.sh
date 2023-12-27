# trec covid
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-trec-covid.multifield \
    --topics beir-v1.0.0-trec-covid-test \
    --output run.beir.bm25-multifield.trec-covid.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

# NFCorpus
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-nfcorpus.multifield \
    --topics beir-v1.0.0-nfcorpus-test \
    --output run.beir.bm25-multifield.nfcorpus.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

# FiQA
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-fiqa.multifield \
    --topics beir-v1.0.0-fiqa-test \
    --output run.beir.bm25-multifield.fiqa.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

# arguana
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-arguana.multifield \
    --topics beir-v1.0.0-arguana-test \
    --output run.beir.bm25-multifield.arguana.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0


# Touche
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-webis-touche2020.multifield \
    --topics beir-v1.0.0-webis-touche2020-test \
    --output run.beir.bm25-multifield.webis-touche2020.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0


# DBPedia
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-dbpedia-entity.multifield \
    --topics beir-v1.0.0-dbpedia-entity-test \
    --output run.beir.bm25-multifield.dbpedia-entity.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

# Scidocs
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-scidocs.multifield \
    --topics beir-v1.0.0-scidocs-test \
    --output run.beir.bm25-multifield.scidocs.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

# climate-fever
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-climate-fever.multifield \
    --topics beir-v1.0.0-climate-fever-test \
    --output run.beir.bm25-multifield.climate-fever.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

# scifact
python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index beir-v1.0.0-scifact.multifield \
    --topics beir-v1.0.0-scifact-test \
    --output run.beir.bm25-multifield.scifact.txt \
    --output-format trec \
    --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

