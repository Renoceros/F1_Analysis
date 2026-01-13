A comprehensive toolkit for analyzing Formula 1 telemetry, race pace, and strategy data. This project leverages the [FastF1](https://github.com/theOehrly/Fast-F1) library to extract and visualize deeper insights from Grand Prix weekends.

## 📖 Overview

**F1_Analysis** provides scripts and notebooks to process raw F1 timing and telemetry data. It is designed for fans and analysts who want to go beyond the TV broadcast graphics.

**Key capabilities include:**
* **Telemetry Comparison:** Compare speed, throttle, and braking traces between two drivers on their fastest laps.
* **Race Pace Analysis:** Visualize lap time consistency and stint performance using violin plots and scatter charts.
* **Tyre Strategy:** Analyze tyre degradation and pit stop windows.
* **Corner Analysis:** Minisector analysis to see where drivers are gaining or losing time.

## 🛠️ Tech Stack

* **Python 3.8+**
* **[FastF1](https://github.com/theOehrly/Fast-F1):** For retrieving timing, telemetry, and session data.
* **Pandas:** Data manipulation and cleaning.
* **Matplotlib / Seaborn:** Visualization and plotting.
* **NumPy:** Numerical operations.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/Renoceros/F1_Analysis.git](https://github.com/Renoceros/F1_Analysis.git)

# Navigate to the project directory
cd F1_Analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt