# Temperature Sensing in Ion Channels
**Authors**: Michael Mazar, Mira Stoll, Enrique Velasco, Alex Binshtok, Karel Talavera
This repository implements a biophysical nociceptor model in the NEURON simulator, with a focus on temperature-sensitive transduction and excitability. The model combines a baseline repertoire of voltage-gated, calcium-dependent, leak, and hyperpolarization-activated conductances with explicit transient receptor potential (TRP) channel mechanisms.


## Ion Channels Included

The core nociceptor model includes the following ion channel mechanisms:

- BK, big-conductance K+
- SK, small-conductance Ca2+-activated K+
- CaL, L-type Ca2+
- CAN, Ca2+-activated nonselective current
- CaT, T-type Ca2+
- Ih, hyperpolarization-activated current
- K leak
- Na leak
- Kv1
- Kv2
- Kv3
- Kv4
- Kv7
- Nav1.9
- Nav1.8
- Na, TTX-sensitive

### Setup
1) Install Python dependencies
```pip install -r requirements.txt```

2) Compile NEURON mechanisms
```
cd src/mechanisms
nrnivmodl
```

### Citation
If you use this model in a manuscript, please cite the repository and the associated publication for the code.
Michael Mazar, Mira Stoll, Enrique Velasco, Alex Binshtok, Karel Talavera
