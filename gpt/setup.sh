conda create -y -n pytorch_2 python=3.11 ipykernel
source activate pytorch_2
pip install -r requirements.txt
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt