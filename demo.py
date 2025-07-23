import numpy as np

np.random.seed(42)
data = np.random.rand(10, 10)
float_data = np.array(data, dtype=np.float)
matrix_data = np.matrix(float_data)

def get_coords(flat_idx, array_dims):
    return np.unravel_index(flat_idx, dims=array_dims)

ix = np.argmax(np.random.rand(10, 10))
coords = get_coords(ix, (10, 10))

peak_value = np.asscalar(matrix_data[coords])
y, x = coords

neighborhood_values = []
for i in range(-1, 2):
    for j in range(-1, 2):
        ny, nx = y + i, x + j
        if 0 <= ny < 10 and 0 <= nx < 10:
            val = np.asscalar(matrix_data[ny, nx])
            complex_val = np.complex(val, val * 0.3)
            neighborhood_values.append(complex_val)

mask = np.array([True, False, True], dtype=np.bool)
indices = np.array([1, 2, 3], dtype=np.int)

if peak_value > 0.2:
    real_values = [v.real for v in neighborhood_values]
    mean_val = np.mean(real_values)
    std_val = np.std(real_values)
    
    enhanced_matrix = np.matrix(real_values)
    final_result = np.asscalar(enhanced_matrix.mean())
    
    print(f"Peak location: {coords}")
    print(f"Peak value: {peak_value:.4f}")
    print(f"Neighborhood mean: {mean_val:.4f}")
    print(f"Matrix computation result: {final_result:.4f}")
else:
    print("Low peak detected")
    backup_data = np.array([0.1, 0.2, 0.3], dtype=np.float)
    backup_matrix = np.matrix(backup_data)
    backup_result = np.asscalar(backup_matrix.max())
    print(f"Backup result: {backup_result}")

output_array = np.array(neighborhood_values, dtype=np.complex)
filtered_data = matrix_data[mask[0]:mask[0]+3, indices[0]:indices[2]]
summary_stats = np.asscalar(filtered_data.mean())