from collections import defaultdict
import datetime
import time
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from model import WidthModel
from tensorboardX import SummaryWriter
import datasets
import numpy as np
import matplotlib.pyplot as plt

RANDOM = jax.random.PRNGKey(69)
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 30

def calculate_loss(model, q_profile_x: jax.Array, amplitude_x: jax.Array, y: jax.Array):
    pred_y = jax.vmap(model)(q_profile_x, amplitude_x)
    mse_loss = jnp.mean((pred_y * 100 - y * 100) ** 2)

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


# Load the dataset
plasma_data = datasets.load_from_disk("plasma_diagnostics_dataset")
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

# Function to extract sliding windows
def extract_windows(example):
    q_profile = example["q_profile"]
    amplitude = example["mhd_amplitude"]
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

# Apply the window extraction to the dataset
# Manually process and flatten the data
processed_data = {"q_profile": [], "amplitude": [], "width": [], "width_profile": []}
for shot in plasma_data:
    windows = extract_windows(shot)
    if windows["q_profile"]:
        processed_data["q_profile"].extend(windows["q_profile"])
        processed_data["amplitude"].extend(windows["amplitude"])
        processed_data["width"].extend(windows["width"])
        processed_data["width_profile"].extend(windows["width_profile"])

processed_data["q_profile"] = np.array(processed_data["q_profile"])
processed_data["amplitude"] = np.array(processed_data["amplitude"])
processed_data["width"] = np.array(processed_data["width"])
processed_data["width_profile"] = np.array(processed_data["width_profile"])

# Check for NaNs in the processed data
if np.isnan(processed_data["q_profile"]).any():
    print("NaN found in processed q_profile data")
if np.isnan(processed_data["amplitude"]).any():
    print("NaN found in processed amplitude data")
if np.isnan(processed_data["width"]).any():
    print("NaN found in processed width data")
if np.isnan(processed_data["width_profile"]).any():
    print("NaN found in processed width_profile data")


plasma_data = datasets.Dataset.from_dict(processed_data)

# Plot a few examples from the processed data
print("Plotting sample data...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Plot first 3 examples
for i in range(3):
    # Plot q_profile heatmap (time vs radial position)
    im1 = axes[0, i].imshow(processed_data["q_profile"][i+10].T, aspect='auto', cmap='viridis')
    axes[0, i].set_title(f'Q-Profile Window {i+1}')
    axes[0, i].set_xlabel('Time Steps')
    axes[0, i].set_ylabel('Radial Position')
    plt.colorbar(im1, ax=axes[0, i])
    
    # Plot amplitude time series
    axes[1, i].plot(processed_data["amplitude"][i+10])
    axes[1, i].set_title(f'MHD Amplitude Window {i+1}')
    axes[1, i].set_xlabel('Time Steps')
    axes[1, i].set_ylabel('Amplitude')
    axes[1, i].grid(True)
    
    # Plot width window with target highlighted
    axes[2, i].plot(processed_data["width_profile"][i+10], 'b-', label='Width Window')
    # Highlight the target value (center of window)
    center_idx = len(processed_data["width_profile"][i+10]) // 2
    axes[2, i].plot(center_idx, processed_data["width"][i+10], 'ro', markersize=8, label=f'Target: {processed_data["width"][i]:.4f}')
    axes[2, i].set_title(f'Width Window {i+1}')
    axes[2, i].set_xlabel('Time Steps')
    axes[2, i].set_ylabel('Width')
    axes[2, i].legend()
    axes[2, i].grid(True)

plt.tight_layout()
plt.savefig('sample_data_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot width distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(processed_data["width"], bins=50, alpha=0.7, edgecolor='black')
plt.title('Width Distribution')
plt.xlabel('Width')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(processed_data["width"][:1000])
plt.title('Width Values (First 1000 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Width')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('width_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print data statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(processed_data['width'])}")
print(f"Q-profile shape per sample: {processed_data['q_profile'][0].shape}")
print(f"Amplitude shape per sample: {processed_data['amplitude'][0].shape}")
print("Width statistics:")
print(f"  Min: {np.min(processed_data['width']):.6f}")
print(f"  Max: {np.max(processed_data['width']):.6f}")
print(f"  Mean: {np.mean(processed_data['width']):.6f}")
print(f"  Std: {np.std(processed_data['width']):.6f}")

# Split the dataset into training and evaluation sets
# Since we don't have predefined splits, we'll create them.
# First, shuffle the data
plasma_data = plasma_data.shuffle(seed=42)

# Then, split into training and testing sets
split_dataset = plasma_data.train_test_split(test_size=0.2)
train_data = split_dataset["train"]
eval_data = split_dataset["test"]

model = WidthModel(key=RANDOM)

print("Starting training from scratch")

# current_step = 0
starting_epoch = 0

optimizer = optax.adam(1e-4)
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

# ... existing code ...

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
            print(f"Target: {y[0]}, Pred: {pred_y[0]}  ")
            print(f"Epoch {epoch}, Step {i}, Loss: {total_loss:.4f}, "
                  f"Data Load Time: {data_load_time:.4f}s, "
                  f"Training Time: {train_time:.4f}s, "
                  f"Logging Time: {log_time:.4f}s, "
                  f"Wait Time: {wait_time:.4f}s")

        wait_start = time.time()

    # Save model
    eqx.tree_serialise_leaves("./width_model.eqx", model)
