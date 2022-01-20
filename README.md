Main current project for table tennis tracking in FIP

# Team Table Tennis
## Providing Live Tracking, Visualization, and Statistics for Table Tennis Recreation and Research.

<!-- ![Example tracking](https://github.com/ScripteJunkie/AirHeads/blob/main/server/public/tracked.jpg) -->

### Install:
> Clone and open T3 library.
```
git clone --recursive https://github.com/ScripteJunkie/T3.git
cd T3
```

### Setup:
> Setup virtual environment.
```
conda create -n ttennis python=3.8
conda activate ttennis
python ./lib/depthai/install_requirments.py
```

### Alignment:
> Center table with displayed target.
```
python ./src/align.py
```

### Calibration:
> Adjust sliders to determine colors and click table borders to find edge bounds.
> Hit 's' when finished and change trackOUT file values based on console output.
```
python ./src/colorfinder.py
```

### Run:
> Displays both mask and tracked frames.
```
python ./src/trackOUT.py
```

