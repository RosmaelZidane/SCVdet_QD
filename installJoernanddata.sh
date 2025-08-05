if [[ -d sourcescripts/storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents sourcescripts/storage/external
fi

cd sourcescripts/storage/external

if [[ ! -d joern-cli ]]; then
    # https://github.com/joernio/joern/releases/download/v4.0.398/joern-cli.zip
    wget https://github.com/joernio/joern/releases/download/v2.0.331/joern-cli.zip
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi

# quemu data

# https://drive.google.com/file/d/1oewQvGPwzGR2lMp51NQhqJaFIvL8j1lH/view?usp=sharing
if [[ ! -d "data_FFmpeg+qemu.csv" ]]; then
    gdown https://drive.google.com/uc\?id\=1oewQvGPwzGR2lMp51NQhqJaFIvL8j1lH
    unzip data_FFmpeg+qemu.zip
    rm data_FFmpeg+qemu.zip
else
    echo "Already downloaded Joern"
fi

cd ..

cd processed

# https://drive.google.com/file/d/1C_MdQLEOfXdE8ru2Kermk4tUou2uAuEL/view?usp=sharing
if [[ ! -f "dataset.zip" ]]; then
    gdown https://drive.google.com/uc\?id\=1C_MdQLEOfXdE8ru2Kermk4tUou2uAuEL
    unzip dataset.zip
    rm dataset.zip
else
    echo "Already downloaded dataset data"
fi



# project kb data
# if [[ ! -f "ProjectKB_domain_csv" ]]; then
#     gdown https://drive.google.com/uc\?id\=1W3Truwd_kEAGYwBgHd6hRLELAeyQRBwT
#     unzip ProjectKB_domain_csv.zip
#     rm ProjectKB_domain_csv.zip
# else
#     echo "Already downloaded dataset data"
# fi


# https://drive.google.com/file/d/1W3Truwd_kEAGYwBgHd6hRLELAeyQRBwT/view?usp=sharing


# cd .. 

# cd processed

# # https://drive.google.com/file/d/1wcoXzQEWxCDWmhC4mtCB4nHIGcIbeoNW/view?usp=sharing

# if [[ ! -f "PJkb.zip" ]]; then
#     gdown https://drive.google.com/uc\?id\=1wcoXzQEWxCDWmhC4mtCB4nHIGcIbeoNW
#     unzip PJkb.zip
#     rm PJkb.zip
# else
#     echo "Already downloaded dataset data"
# fi

cd ..
cd cache

if [[ ! -f "codebert_finetuned" ]]; then
    gdown https://drive.google.com/uc?id=1X-QitxtD3Djdg8lJNrzkyzomBSwHu-A1
    unzip codebert_finetuned.zip
    rm codebert_finetuned.zip
else
    echo "Already downloaded codebert finetuned"
fi

mkdir embedmodel

mv codebert_finetuned/ embedmodel/CodeBERT


