from collections import defaultdict
import datetime
import time
import random
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from model import WidthModel
from tensorboardX import SummaryWriter
import datasets as ds
import numpy as np
import matplotlib.pyplot as plt


random.seed(1)
RANDOM = jax.random.PRNGKey(69)
LEARNING_RATE = 5e-4  # Slightly lower learning rate for stability
BATCH_SIZE = 64       # Larger batch size for better gradient estimates
NUM_EPOCHS = 20   # More epochs for convergence

def calculate_loss(model, q_profile_x: jax.Array, amplitude_x: jax.Array, y: jax.Array):
    pred_y = jax.vmap(model)(q_profile_x, amplitude_x)
    mse_loss = jnp.mean((pred_y - y) ** 2)

    return mse_loss, pred_y

@eqx.filter_jit
def make_step(model, optimizer, opt_state, q_profile_x: jax.Array, amplitude_x: jax.Array, y: jax.Array):
    (total_loss, pred_y), grads = eqx.filter_value_and_grad(calculate_loss, has_aux=True)(model, q_profile_x, amplitude_x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return (
        model,
        opt_state,
        total_loss,
        pred_y
    )


# Function to extract sliding windows
def extract_windows(example):

    q_profile = example["q_profile"]
    # Use 'mhd_amplitude' if present, else fallback to 'amplitude' for processed shots
    amplitude = example["mhd_amplitude"] if "mhd_amplitude" in example else example["amplitude"]
    width = example["width"]

    if len(q_profile) < WINDOW_SIZE:
        return {"q_profile": [], "amplitude": [], "width": []}

    windows_q = []
    windows_amplitude = []
    targets = []
    width_windows = []

    for t in range(WINDOW_SIZE // 2, len(q_profile) - WINDOW_SIZE // 2):
        window_q = q_profile[t - WINDOW_SIZE // 2 : t + WINDOW_SIZE // 2, :]
        window_amplitude = amplitude[t - WINDOW_SIZE // 2 : t + WINDOW_SIZE // 2]
        window_width = width[t - WINDOW_SIZE // 2 : t + WINDOW_SIZE // 2]
        target = width[t]
        if np.isnan(window_q).any() or np.isnan(window_amplitude).any() or np.isnan(target):
            continue
        windows_q.append(window_q)
        windows_amplitude.append(window_amplitude)
        width_windows.append(window_width)
        targets.append(target)

    return {
        "q_profile": windows_q,
        "amplitude": windows_amplitude,
        "width": targets,
        "width_profile": width_windows
    }


# Load the dataset
plasma_data = ds.load_from_disk("plasma_diagnostics_dataset")
plasma_data = plasma_data.with_format("jax")

# Explore the dataset
print("Dataset Features:", plasma_data.features)
print("Dataset Example:", plasma_data[0])

# Check the train and test splits
if "train" in plasma_data and "test" in plasma_data:
    print("Train Dataset Example:", plasma_data["train"][0])
    print("Test Dataset Example:", plasma_data["test"][0])
else:
    print("Dataset does not have train/test splits.")

# Filter out invalid entries
plasma_data = plasma_data.filter(lambda x: len(x['q_profile']) > 0, batched=False)

# Debugging: Print invalid entries
print("Inspecting dataset entries after filtering...")
for i, entry in enumerate(plasma_data):
    if len(entry['q_profile']) == 0:
        print(f"Invalid entry at index {i}: {entry}")

# Define sliding window parameters
WINDOW_SIZE = 40  # Number of timesteps
RADIAL_POINTS = 61  # Number of radial positions


# --- Split shots before window extraction: keep one for test, rest for train ---
all_shot_indices = list(range(len(plasma_data)))

random.shuffle(all_shot_indices)
test_shot_idx = all_shot_indices[0]
train_shot_indices = all_shot_indices[1:]

test_shot = plasma_data[test_shot_idx]
train_shots = [plasma_data[i] for i in train_shot_indices]
# Now process train and test shots separately
def process_shots(shots):
    processed = {"q_profile": [], "amplitude": [], "width": [], "width_profile": []}
    for shot in shots:
        windows = extract_windows(shot)
        if windows["q_profile"]:
            processed["q_profile"].extend(windows["q_profile"])
            processed["amplitude"].extend(windows["amplitude"])
            processed["width"].extend(windows["width"])
            processed["width_profile"].extend(windows["width_profile"])
    processed["q_profile"] = np.array(processed["q_profile"])
    processed["amplitude"] = np.array(processed["amplitude"])
    processed["width"] = np.array(processed["width"])
    processed["width_profile"] = np.array(processed["width_profile"])
    return processed

train_processed = process_shots(train_shots)
test_processed = process_shots([test_shot])

# Check for NaNs in the processed training data
if np.isnan(train_processed["q_profile"]).any():
    print("NaN found in train q_profile data")
if np.isnan(train_processed["amplitude"]).any():
    print("NaN found in train amplitude data")
if np.isnan(train_processed["width"]).any():
    print("NaN found in train width data")
if np.isnan(train_processed["width_profile"]).any():
    print("NaN found in train width_profile data")

# Normalize the training data
print("Normalizing training data...")
width_mean = np.mean(train_processed["width"])
width_std = np.std(train_processed["width"])
print(f"Width - Mean: {width_mean:.6f}, Std: {width_std:.6f}")
train_processed["q_profile"] = (train_processed["q_profile"] - np.mean(train_processed["q_profile"])) / np.std(train_processed["q_profile"])
train_processed["amplitude"] = (train_processed["amplitude"] - np.mean(train_processed["amplitude"])) / np.std(train_processed["amplitude"])
train_processed["width"] = (train_processed["width"] - width_mean) / width_std

# Store normalization parameters for denormalization during inference
q_profile_mean = np.mean(train_processed["q_profile"])
q_profile_std = np.std(train_processed["q_profile"])
amplitude_mean = np.mean(train_processed["amplitude"])
amplitude_std = np.std(train_processed["amplitude"])
normalization_params = {
    "width_mean": width_mean,
    "width_std": width_std
}
print("Training data normalization complete.")

# For compatibility with the rest of the code, create datasets
plasma_data = ds.Dataset.from_dict(train_processed)
train_data = plasma_data
eval_data = ds.Dataset.from_dict(test_processed)


# Plot a few examples from the training data
print("Plotting sample training data...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i in range(3):
    im1 = axes[0, i].imshow(train_processed["q_profile"][i+10].T, aspect='auto', cmap='viridis')
    axes[0, i].set_title(f'Q-Profile Window {i+1}')
    axes[0, i].set_xlabel('Time Steps')
    axes[0, i].set_ylabel('Radial Position')
    plt.colorbar(im1, ax=axes[0, i])
    axes[1, i].plot(train_processed["amplitude"][i+10])
    axes[1, i].set_title(f'MHD Amplitude Window {i+1}')
    axes[1, i].set_xlabel('Time Steps')
    axes[1, i].set_ylabel('Amplitude')
    axes[1, i].grid(True)
    axes[2, i].plot(train_processed["width_profile"][i+10], 'b-', label='Width Window')
    center_idx = len(train_processed["width_profile"][i+10]) // 2
    axes[2, i].plot(center_idx, train_processed["width"][i+10], 'ro', markersize=8, label=f'Target: {train_processed["width"][i+10]:.4f}')
    axes[2, i].set_title(f'Width Window {i+1}')
    axes[2, i].set_xlabel('Time Steps')
    axes[2, i].set_ylabel('Width')
    axes[2, i].legend()
    axes[2, i].grid(True)
plt.tight_layout()
plt.savefig('sample_data_visualization.png', dpi=150, bbox_inches='tight')
plt.show()


# Plot width distribution for training data
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(train_processed["width"], bins=50, alpha=0.7, edgecolor='black')
plt.title('Width Distribution')
plt.xlabel('Width')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(train_processed["width"][:1000])
plt.title('Width Values (First 1000 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Width')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('width_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# Print training data statistics
print("\nTraining Dataset Statistics:")
print(f"Total samples: {len(train_processed['width'])}")
print(f"Q-profile shape per sample: {train_processed['q_profile'][0].shape}")
print(f"Amplitude shape per sample: {train_processed['amplitude'][0].shape}")
print("Width statistics:")
print(f"  Min: {np.min(train_processed['width']):.6f}")
print(f"  Max: {np.max(train_processed['width']):.6f}")
print(f"  Mean: {np.mean(train_processed['width']):.6f}")
print(f"  Std: {np.std(train_processed['width']):.6f}")


model = WidthModel(key=RANDOM)

print("Starting training from scratch")

# current_step = 0
starting_epoch = 0

# Add a cosine decay learning rate scheduler
scheduler = optax.cosine_decay_schedule(init_value=1e-4, decay_steps=NUM_EPOCHS * (len(train_data) // BATCH_SIZE), alpha=0.0)
optimizer = optax.adam(scheduler)
opt_state = optimizer.init(model)

writer = SummaryWriter(
    log_dir="./runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: F821
)

step = 0

# Timing stats dictionary
def data_loader(dataset, batch_size):
    dataset.set_format(type='numpy')
    
    # Get the number of samples in the dataset
    num_samples = len(dataset)
    
    # Create an array of indices and shuffle them
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Yield batches
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield dataset[batch_indices]

timing_stats = defaultdict(list)

for epoch in range(starting_epoch, NUM_EPOCHS):
    epoch_start = time.time()
    wait_start = time.time()

    for i, batch in enumerate(data_loader(train_data, BATCH_SIZE)):
        wait_time = time.time() - wait_start

        # Measure data loading time
        data_load_start = time.time()
        q_profile_x, amplitude_x, y = batch["q_profile"], batch["amplitude"], batch["width"]
        data_load_time = time.time() - data_load_start

        # Measure training step time
        train_start = time.time()
        results = make_step(model, optimizer, opt_state, q_profile_x, amplitude_x, y)
        train_time = time.time() - train_start

        # Measure logging time
        log_start = time.time()
        model, opt_state, total_loss, pred_y = results

        step += 1
        writer.add_scalar("Total loss", total_loss, step)
        log_time = time.time() - log_start

        # Store timing stats
        timing_stats["data_loading"].append(data_load_time)
        timing_stats["training"].append(train_time)
        timing_stats["logging"].append(log_time)
        timing_stats["wait_time"].append(wait_time)

        if i % 10 == 0:
            # Denormalize for logging to see actual values
            target_denorm = y[0] * normalization_params["width_std"] + normalization_params["width_mean"]
            pred_denorm = pred_y[0] * normalization_params["width_std"] + normalization_params["width_mean"]
            
            print(f"Target: {target_denorm:.6f}, Pred: {pred_denorm:.6f} (diff: {abs(target_denorm - pred_denorm):.6f})")
            print(f"Epoch {epoch}, Step {i}, Loss: {total_loss:.4f}, "
                  f"Data Load Time: {data_load_time:.4f}s, "
                  f"Training Time: {train_time:.4f}s, "
                  f"Logging Time: {log_time:.4f}s, "
                  f"Wait Time: {wait_time:.4f}s")

        wait_start = time.time()

    # Save model
    eqx.tree_serialise_leaves("./width_model.eqx", model)
    print(f"Test idx: {test_shot_idx}")

# # Load the trained model
# model_path = "./width_model.eqx"
# print(f"Loading model from {model_path}...")
# model = eqx.tree_deserialise_leaves(model_path, WidthModel(key=jax.random.PRNGKey(0)))
# q_profile = np.array(test_shot["q_profile"])
# amplitude = np.array(test_shot["mhd_amplitude"])
# width = np.array(test_shot["width"])
# # # Normalize the inputs (use the same normalization as during training)
# q_profile_mean = np.mean(q_profile)
# q_profile_std = np.std(q_profile)
# amplitude_mean = np.mean(amplitude)
# amplitude_std = np.std(amplitude)
# width_mean = np.mean(width)
# width_std = np.std(width)

# q_profile = (q_profile - q_profile_mean) / q_profile_std
# amplitude = (amplitude - amplitude_mean) / amplitude_std
# width = (width - width_mean) / width_std

# # Predict the island width
# WINDOW_SIZE = 40
# predicted_widths = []
# real_widths = []

# for t in range(WINDOW_SIZE // 2, len(q_profile) - WINDOW_SIZE // 2):
#     window_q = q_profile[t - WINDOW_SIZE // 2 : t + WINDOW_SIZE // 2, :]
#     window_amplitude = amplitude[t - WINDOW_SIZE // 2 : t + WINDOW_SIZE // 2]
#     real_width = width[t]

#     pred_width = model(window_q, window_amplitude)
#     predicted_widths.append(pred_width.item() * width_std + width_mean)  # Denormalize
#     real_widths.append(real_width * width_std + width_mean)
# # Extract the shot number directly from the dataset data
# shot_number = test_shot["shot_number"]
# # Adjust x-axis to reflect time in seconds based on dataset time data
# time_data = np.array(test_shot["time"])  # Assuming "time" field exists in the dataset and is in seconds
# # Slice time_data to match the length of real_widths and predicted_widths
# time_data = time_data[WINDOW_SIZE // 2 : -(WINDOW_SIZE // 2)]
# plt.figure(figsize=(12, 8))
# plt.plot(time_data, real_widths, label="Real Width", color="blue")
# plt.plot(time_data, predicted_widths, label="Predicted Width", color="red", linestyle="--")
# plt.xlabel("Time (s)", fontsize=16)
# plt.ylabel("Island Width", fontsize=16)
# plt.legend(fontsize=14)
# plt.grid(True)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.show()

