import os

# Path to your main folder
main_folder = "//Users/mohammadbilal/Documents/Projects/STM32-InstanceSegmentation/base_model/Dataset"

# List all items in the folder and filter directories
subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

# Count
num_subfolders = len(subfolders)

print(f"Number of subfolders: {num_subfolders}")

organism_dict = {}
for i in range(num_subfolders):
    organism_dict[i] = subfolders[i]

print(organism_dict)