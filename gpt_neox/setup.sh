conda create -y -n pytorch_2 python=3.11 ipykernel

source activate pytorch_2

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/EleutherAI/gpt-neox
pushd gpt-neox/requirements
pip install -r requirements.txt