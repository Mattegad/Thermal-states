# Quantum state tomography with reservoir computing

This project aims at :
- Simulating thermal squeezed states in which we can add noise in either the amplitude or the phase quadrature (or both).
- Injecting them into a polaritonic microcavity and solving the dynamics.
- Computing the output quadratures and the balanced sum detection photocurrent.
- Training a regression to predict the parameters of the thermal squeezed states injected bases on the balanced detection measurement at the output of the reservoir.

---

## ⚙️ Installation & Use  

### 1. Clone the project
```bash
git clone https://github.com/Mattegad/Thermal-states.git
cd Training-thermal-states
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```
---

## 🧱 Project organization

**Principal functions**
-`Polariton_Microcavity_OHT.py` → Generate the thermal squeezed states, evolve them in the reservoir and compute the desire features
-`Replot.py` → Replot the saved results 
-`Amplitude_only_reconstruction_from_OHT.py` → Generate a set of test states, train a model and reconstruct the state parameters of test states
-`Replot_amplitude_only.py` → Replot the predictions of the training 

---

## 🙏 Credits
This project has been developed by Matteo Gadani at the Laboratoire Kastler Brossel.
We thank Wouter Verstraelen et al. for the first results. Their contributions were essential to the development of this code.

---

## 📘 Licence & context  
This project illustrates the **quantum states tomography** via simulations and reservoir computing. It complements an experiment done in the LKB where we aim at recognizing quantum states with an exciton polariton reservoir.
Verify the **licence** for use and distribution.  

---

## Physical model 

