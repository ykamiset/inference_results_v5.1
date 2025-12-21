cd /tmp/mitten
git checkout ${MITTEN_HASH}
git submodule update --init
sed -i 's/numpy >=1.22.0, <1.24.0/numpy >=1.26.4/' ./setup.cfg
pip install .