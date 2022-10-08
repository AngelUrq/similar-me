rm -r data/train
mkdir data/train
apt update
apt install -y p7zip-full
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1akr3MnRsk94okzpsURaFdusS1j6rmwbi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1akr3MnRsk94okzpsURaFdusS1j6rmwbi" -O data/train/data.zip && rm -rf /tmp/cookies.txt
apt install -y p7zip-full
7z x data/train/data.zip -odata/train/
