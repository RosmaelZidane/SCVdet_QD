

cd sourcescripts  

# 1 meta data extraction
python3 -B ./processing/process.py  
python3 -B ./processing/graphdata.py  


# 2 

python3 -B ./embeddmodel/codebert.py  
python3 -B ./embeddmodel/sentencebert.py  
python3 -B ./embeddmodel/word2vec.py  
python3 -B ./processing/graphconstruction.py 



# 3 train test


python3 -B ./model/scvuldetect.py  