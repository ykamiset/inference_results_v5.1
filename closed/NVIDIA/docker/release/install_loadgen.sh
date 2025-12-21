git clone ${INFERENCE_URL} /opt/inference
cd /opt/inference/loadgen
git checkout ${INFERENCE_HASH}
pip install -e .

