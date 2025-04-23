import rpy2.robjects as robjects
import os

# Replace this with the correct .R file you actually have
r_script_path = r"C:\Users\MM3815\Documents\Colgate\Colgatee.R"

# Check if file exists
if not os.path.isfile(r_script_path):
    raise FileNotFoundError(f"R script not found at: {r_script_path}")

# Read and run the R script
with open(r_script_path, "r", encoding="utf-8") as file:
    r_code = file.read()

try:
    robjects.r(r_code)
    print("R script executed successfully.")
except Exception as e:
    print("Error while executing R script:")
    print(e)
