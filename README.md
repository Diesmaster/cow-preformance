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
  - SIMEX to get rid of measurement errors did not work.
  - Kalman filter to get rid of measurement errors -> did work.



## NOTES:
Medical history was incorrectly recorded, so data about that is unreliable.
Cow with ID: rexFmUY8QHCvB0TsjnbB had major issues so is left out of the analysis.

## NOTES:
Neg required = a * weight + adg^2
Neg required (Negr) = a * weight + (Neg/Negr)^2
Neg required (Negr) = a * weight + Neg^2/Negr^2
Neg required (Negr) = a * weight + Neg^2/Negr^2

0 = a * weight + Neg^2/Negr^2 - Negr 

which gives approx:

negr = a * weight neg^2(a * weight)^2

or if weight is small

Negr = Neg^2/3

sinds average pref on the cow farm is 1 adg -> negr = neg per day

would require some sort of hidden state model.

for now Negr is approx 1, -> more advanced methods need more data.


## TODO:

explain the breed split.
explain all the finding with the ratio, and the dmi/dt etc. processes/
explain different values for Negr that failed etc.
make all the in between models so it can easily be checked
make 3 data processing options -> raw, kalhman filter, kahlman filter + tail removal.



Try to come up with a generalized from of wg over long periods.

show difference between OLS and Panel

Conculsions
