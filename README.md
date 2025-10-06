steps:

1. clean data --
2. process the data --
3. make a folder to read out and process the data --
---
4. make code for the models and the statistical tests
5. Make a plotter class
6. write down findings
7. statistical dashboard


## INSTALL
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
./unzip.sh
```

## conclusion notes:
  - OLS naive estimation -> o.36 R but very high p values
  - FixedEffectsModel -> singular matrix -> cows don't very enought from eachother?


## NOTES:
Medical history was incorrectly recorded, so data about that is unreliable.
Cow with ID: rexFmUY8QHCvB0TsjnbB had major issues so is left out of the analysis.



