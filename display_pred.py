import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from model import WidthModel
import random
import datasets


# Load the trained model
model_path = "./width_model.eqx"
print(f"Loading model from {model_path}...")
model = eqx.tree_deserialise_leaves(model_path, WidthModel(key=jax.random.PRNGKey(0)))

# Load the plasma diagnostics dataset
plasma_data = datasets.load_from_disk("plasma_diagnostics_dataset")

# Select a random shot
random_index = random.randint(0, len(plasma_data) - 1)
shot = plasma_data[6]
print(f"Loading random shot data from index {random_index}...")

q_profile = np.array(shot["q_profile"])
amplitude = np.array(shot["mhd_amplitude"])
width = np.array(shot["width"])
# # Normalize the inputs (use the same normalization as during training)
q_profile_mean = np.mean(q_profile)
q_profile_std = np.std(q_profile)
amplitude_mean = np.mean(amplitude)
amplitude_std = np.std(amplitude)
width_mean = np.mean(width)
width_std = np.std(width)

q_profile = (q_profile - q_profile_mean) / q_profile_std
amplitude = (amplitude - amplitude_mean) / amplitude_std
width = (width - width_mean) / width_std

# Predict the island width
WINDOW_SIZE = 40
predicted_widths = []
real_widths = []


for t in range(WINDOW_SIZE // 2, len(q_profile) - WINDOW_SIZE // 2):
    window_q = q_profile[t - WINDOW_SIZE // 2 : t + WINDOW_SIZE // 2, :]
    window_amplitude = amplitude[t - WINDOW_SIZE // 2 : t + WINDOW_SIZE // 2]
    real_width = width[t]

    # Model expects batched inputs, so add batch dimension
    # window_q = np.expand_dims(window_q, axis=0)
    # window_amplitude = np.expand_dims(window_amplitude, axis=0)

    pred_width = model(window_q, window_amplitude)
    predicted_widths.append(pred_width.item() * width_std + width_mean)  # Denormalize
    real_widths.append(real_width * width_std + width_mean)
# Extract the shot number directly from the dataset data
shot_number = shot["shot_number"]
print(f"Visualized results for shot number {shot_number} at index {random_index}.")

print(real_widths)

# Adjust x-axis to reflect time in seconds based on dataset time data
time_data = np.array(shot["time"])  # Assuming "time" field exists in the dataset and is in seconds
# Slice time_data to match the length of real_widths and predicted_widths
time_data = time_data[WINDOW_SIZE // 2 : -(WINDOW_SIZE // 2)]
plt.figure(figsize=(12, 8))
plt.plot(time_data, real_widths, label="Real Width", color="blue")
plt.plot(time_data, predicted_widths, label="Predicted Width", color="red", linestyle="--")
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Island Width", fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

