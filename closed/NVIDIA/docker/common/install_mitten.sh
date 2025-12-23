git clone https://github.com/NVIDIA/mitten.git
git checkout 4b6d18d76a530cfad4c70302d93d9afd68e7d99d
# git checkout ${MITTEN_HASH}
git submodule update --init
sed -i 's/numpy >=1.22.0, <1.24.0/numpy >=1.26.4/' ./setup.cfg
pip install .