# sectorRotationHeatmap

Update: added support to dukascopy datasource which can be download tick data if needed. Install dukascopy-node to use this function.


install node on windows:
1. '$ choco install -y nodejs-lts --force'

1. '$ npm install dukascopy-node --save'

to run all the test cases:
1. '$ cd tests'

1. '$ pytest -s --log-cli-level=INFO'

to get updated data:

1. '$ python update_data.py'

to run the model:

1. '$ python run_model.py'

Visit https://www.linkedin.com/pulse/identifying-sector-rotation-trading-signals-using-heatmap-jacky-lee on detailed description on the model.
