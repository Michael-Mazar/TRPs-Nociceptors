# Temperature Sensing in Ion Channels

This repository implements a biophysical nociceptor model in the NEURON simulator, with a focus on temperature-sensitive transduction and excitability. The model combines a baseline repertoire of voltage-gated, calcium-dependent, leak, and hyperpolarization-activated conductances with explicit transient receptor potential (TRP) channel mechanisms.

The central goal of the project is to simulate how thermosensitive ion channels shape nociceptor membrane responses, firing behavior, passive properties, and temperature thresholds. In particular, this model incorporates both TRPV1 and TRPA1 into a nociceptor framework and uses them to study heat- and cold-associated response classes, as well as polymodal behavior when both channels are present.



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

### Requirements
Typical dependencies used by this repository include:

NEURON
NumPy
pandas
matplotlib
seaborn
eFEL
mplcursors

### Citation
If you use this model in a manuscript, please cite the repository and any associated publication or preprint describing the TRPV1/TRPA1 nociceptor framework.