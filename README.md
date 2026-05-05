# Musculoskeletal Gait Generation

This repository contains Python code for generating walking motions of a musculoskeletal model using trajectory optimization.

## Environment Setup

1. Install [Anaconda](https://www.anaconda.com/).
2. You will need to create three separate Conda environments for different tasks:

   **1. Main Environment**
   Create your main environment and install the required packages:
   ```bash
   pip install casadi numpy pandas
   ```

   **2. OpenSim Environment**
   Create an environment where OpenSim is available. For installation and setup details, please refer to the official [OpenSim Python Scripting Guide](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python).

   **3. Pinocchio Environment**
   Create an environment where Pinocchio is available. For installation details, please refer to the [Pinocchio GitHub Repository](https://github.com/stack-of-tasks/pinocchio).

## Usage

Before generating the walking motion, you need to prepare some auxiliary files by running specific scripts in their respective environments.

**Step 1: Generate Muscle Kinematics**
Activate your **OpenSim Environment** and run the following command:
```bash
python scripts/muscle_kinematics_polynomial_regression.py
```

**Step 2: Generate and Compile CasADi Functions**
Activate your **Pinocchio Environment** and run the following command:
```bash
python scripts/generate_and_compile_casadi_functions.py
```

**Step 3: Generate Walking Motion**
Finally, activate your **Main Environment** and run the main script to generate the walking motion:
```bash
python main.py
```