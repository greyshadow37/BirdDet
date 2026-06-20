import matplotlib.pyplot as plt

# Data Definitions
splits = ['Train', 'Validation', 'Test']
birds = ['Ruddy Shelduck', 'Asian Green Bee-Eater', 'Cattle Egret', 'Gray Wagtail', 'Indian Pitta']

train_counts = [567, 412, 390, 352, 281]
val_counts = [192, 124, 170, 136, 75]
test_counts = [269, 212, 172, 146, 84]

# Journal-friendly distinct color palette (colorblind safe)
colors = ['#e63946', '#2a9d8f', '#f4a261', '#e9c46a', '#264653'] 

def plot_donut_standalone(counts, title, total_images, total_birds, output_filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the pie chart with a hole in the middle (donut)
    wedges, texts, autotexts = ax.pie(
        counts, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2), 
        textprops=dict(color="black", fontsize=11, fontweight='bold')
    )
    
    # Add title and metadata above the chart
    ax.set_title(f'{title}\n(Images: {total_images:,} | Birds: {total_birds:,})', fontsize=14, pad=10, fontweight='bold')
    
    # Add a dedicated legend to the right of the pie chart
    ax.legend(wedges, birds, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
    
    # Formatting and saving
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as {output_filename}")
    plt.close() # Close figure to free memory

# Plot each split into its own separate image file
plot_donut_standalone(train_counts, 'Training Set Distribution', 1667, 2002, 'distribution_train.png')
plot_donut_standalone(val_counts, 'Validation Set Distribution', 576, 697, 'distribution_val.png')
plot_donut_standalone(test_counts, 'Test Set Distribution', 717, 883, 'distribution_test.png')
