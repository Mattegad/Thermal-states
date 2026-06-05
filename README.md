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
-'Polariton_Microcavity_OHT.py'
