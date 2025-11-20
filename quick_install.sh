git clone ssh://git@gitlab.cern.ch:7999/lhclumi/nx2pd.git
source ./nx2pd/make_it.sh
python -m pip install -r ./nx2pd/requirements.txt
python -m pip install ./nx2pd
python ./nx2pd/examples/001_simple.py 2> log.txt
