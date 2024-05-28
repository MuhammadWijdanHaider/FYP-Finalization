import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of file paths for the images
file_paths = [
    r"m1.jpg",
    r"m1_backprog.png",
    r"m1_linear.png",
    r"m1_grad.png",
    r"m2.jpg",
    r"m2_linear.png",
    r"m2_backprog.png",
    r"m2_grad.png",
    r"m3.jpg",
    r"m3_backprog.png",
    r"m3_linear.png",
    r"m3_grad.png",
    r"m4.jpg",
    r"m4_backprog.png",
    r"m4_linear.png",
    r"m4_grad.png"
]

fig, axes = plt.subplots(4, 4, figsize=(16, 8))

# Loop through the file paths and display each image
for i, file_path in enumerate(file_paths):
    # Load the image using Matplotlib's imread function
    img = mpimg.imread(file_path)
    
    # Calculate the row and column indices
    row_index = i // 4
    col_index = i % 4
    
    # Display the image on the corresponding axis
    axes[row_index, col_index].imshow(img)
    axes[row_index, col_index].axis('off')  # Hide axis
    
# Show the plot
plt.show()
