"""
Script to check class distribution in training and test datasets
"""
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class_map = {
    'Big_mammal': 0, 'Bird': 1, 'Frog': 2,
    'Lizard': 3, 'Scorpion': 4, 'Small_mammal': 5, 'Spider': 6
}

# Reverse mapping
class_names = {v: k for k, v in class_map.items()}

def analyze_label_file_format(label_dir, num_samples=10):
    """Analyze the format of label files by reading samples"""
    print(f"\nAnalyzing label file format in {label_dir}...")
    
    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not txt_files:
        print("No .txt files found!")
        return None
    
    print(f"Found {len(txt_files)} .txt files")
    
    # Read and analyze sample files
    formats_found = []
    for i, filename in enumerate(txt_files[:num_samples]):
        filepath = os.path.join(label_dir, filename)
        
        print(f"\n{filename}:")
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
            
            if not content:
                print("  Empty file")
                formats_found.append("empty")
                continue
            
            print(f"  Content: '{content}'")
            print(f"  Length: {len(content)} chars")
            
            # Try to parse the content
            if ',' in content:
                parts = content.split(',')
                print(f"  CSV format with {len(parts)} parts: {parts}")
                formats_found.append("csv")
            elif ' ' in content:
                parts = content.split()
                print(f"  Space-separated with {len(parts)} parts: {parts}")
                formats_found.append("space_separated")
            elif '\t' in content:
                parts = content.split('\t')
                print(f"  Tab-separated with {len(parts)} parts: {parts}")
                formats_found.append("tab_separated")
            else:
                # Single value
                print(f"  Single value: {content}")
                formats_found.append("single_value")
                
                # Try to interpret as number
                try:
                    num = float(content)
                    print(f"  As number: {num}")
                except:
                    print(f"  Not a number")
                    
        except Exception as e:
            print(f"  Error reading file: {e}")
            formats_found.append("error")
    
    # Summary of formats found
    format_counts = Counter(formats_found)
    print(f"\nFormat summary from {len(formats_found)} samples:")
    for fmt, count in format_counts.items():
        print(f"  {fmt}: {count} files")
    
    return format_counts

def parse_label_file_adaptive(filepath, class_map):
    """Parse label file with adaptive format detection"""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        if not content:
            return None
        
        # Try different parsing strategies
        class_id = None
        
        # Strategy 1: Direct class name in content
        content_lower = content.lower()
        for class_name, cid in class_map.items():
            if class_name.lower() in content_lower:
                class_id = cid
                break
        
        if class_id is not None:
            return class_id
        
        # Strategy 2: Try to parse as number
        try:
            num = float(content)
            # Check if it's a valid class ID
            if num in class_names:
                return int(num)
            elif 0 <= num < len(class_map):
                return int(num)
        except:
            pass
        
        # Strategy 3: Check for partial matches (e.g., "bigmam" for "Big_mammal")
        filename = os.path.basename(filepath).lower()
        if 'bigmam' in filename or 'big_mam' in filename:
            return class_map['Big_mammal']
        elif 'bird' in filename:
            return class_map['Bird']
        elif 'frog' in filename:
            return class_map['Frog']
        elif 'lizard' in filename:
            return class_map['Lizard']
        elif 'scorpion' in filename:
            return class_map['Scorpion']
        elif 'smallmam' in filename or 'small_mam' in filename:
            return class_map['Small_mammal']
        elif 'spider' in filename:
            return class_map['Spider']
        
        return None
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def count_all_classes(label_dir, class_map, max_files=None):
    """Count classes from all label files"""
    print(f"\nCounting classes from {label_dir}...")
    
    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not txt_files:
        print("No .txt files found!")
        return Counter(), 0
    
    print(f"Found {len(txt_files)} .txt files")
    
    if max_files and max_files < len(txt_files):
        txt_files = txt_files[:max_files]
        print(f"Processing first {max_files} files...")
    
    counts = Counter()
    processed = 0
    empty_files = 0
    
    for i, filename in enumerate(txt_files):
        filepath = os.path.join(label_dir, filename)
        class_id = parse_label_file_adaptive(filepath, class_map)
        
        if class_id is not None:
            counts[class_id] += 1
        else:
            # Check if file is empty
            try:
                with open(filepath, 'r') as f:
                    if not f.read().strip():
                        empty_files += 1
            except:
                pass
        
        processed += 1
        
        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(txt_files)} files...")
    
    print(f"\nProcessing complete:")
    print(f"  Total files processed: {processed}")
    print(f"  Files with valid class: {sum(counts.values())}")
    print(f"  Empty files: {empty_files}")
    print(f"  Files with unknown format: {processed - sum(counts.values()) - empty_files}")
    
    return counts, processed

def find_test_split(label_dir):
    """Try to find test split"""
    print("\nLooking for test split...")
    
    # Check if there's a test subdirectory
    possible_test_dirs = [
        os.path.join(label_dir, 'test'),
        os.path.join(label_dir, 'Test'),
        os.path.join(os.path.dirname(label_dir), 'test'),
        os.path.join(os.path.dirname(label_dir), 'Test'),
    ]
    
    for test_dir in possible_test_dirs:
        if os.path.exists(test_dir):
            print(f"Found test directory: {test_dir}")
            return test_dir
    
    # Check if test files are mixed in
    all_files = os.listdir(label_dir)
    test_files = [f for f in all_files if 'test' in f.lower() and f.endswith('.txt')]
    
    if test_files:
        print(f"Found {len(test_files)} test files mixed in training directory")
        # We'll need a different strategy to separate them
        return "mixed"
    
    print("No separate test directory found")
    return None

def create_train_test_split(train_counts, split_ratio=0.8):
    """Create a simulated test split from training data"""
    print(f"\nCreating simulated test split with ratio {split_ratio}")
    
    test_counts = Counter()
    for class_id, count in train_counts.items():
        test_count = int(count * (1 - split_ratio))
        if test_count < 1 and count > 0:
            test_count = 1
        test_counts[class_id] = test_count
    
    return test_counts

def visualize_distribution(train_counts, test_counts, output_file='class_distribution.png'):
    """Create visualization of class distribution"""
    print("\nCreating visualization...")
    
    # Get class names in order
    sorted_classes = sorted(class_names.keys())
    class_labels = [class_names[i] for i in sorted_classes]
    
    # Get counts in order
    train_counts_sorted = [train_counts.get(i, 0) for i in sorted_classes]
    test_counts_sorted = [test_counts.get(i, 0) for i in sorted_classes]
    
    # Calculate totals
    total_train = sum(train_counts_sorted)
    total_test = sum(test_counts_sorted)
    
    print(f"Training instances: {total_train}")
    print(f"Test instances: {total_test}")
    
    if total_train == 0:
        print("WARNING: No training instances found!")
        return
    
    # Calculate percentages
    train_percentages = [100 * count / total_train for count in train_counts_sorted]
    test_percentages = [100 * count / total_test if total_test > 0 else 0 for count in test_counts_sorted]
    
    # Create figure
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Training set bar plot
    ax1 = axes[0, 0]
    bars1 = ax1.bar(class_labels, train_counts_sorted, color='steelblue', edgecolor='black')
    ax1.set_title(f'Training Set - {total_train:,} instances', fontsize=14, fontweight='bold')
    #ax1.set_xlabel('Classes', fontsize=12)
    ax1.set_ylabel('Number of Instances', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add labels
    for bar, count, perc in zip(bars1, train_counts_sorted, train_percentages):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(train_counts_sorted),
                     f'{count:,}\n({perc:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # 2. Test set bar plot
    ax2 = axes[0, 1]
    bars2 = ax2.bar(class_labels, test_counts_sorted, color='coral', edgecolor='black')
    ax2.set_title(f'Test Set - {total_test:,} instances', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classes', fontsize=12)
    ax2.set_ylabel('Number of Instances', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add labels
    for bar, count, perc in zip(bars2, test_counts_sorted, test_percentages):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(test_counts_sorted),
                     f'{count:,}\n({perc:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # 3. Comparison plot
    ax3 = axes[1, 0]
    x = np.arange(len(class_labels))
    width = 0.35
    
    bars_train = ax3.bar(x - width/2, train_counts_sorted, width, label='Training', 
                         color='steelblue', edgecolor='black')
    bars_test = ax3.bar(x + width/2, test_counts_sorted, width, label='Test', 
                        color='coral', edgecolor='black')
    
    ax3.set_title('Training vs Test Set Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Classes', fontsize=12)
    ax3.set_ylabel('Number of Instances', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_labels, rotation=45)
    ax3.legend()
    
    # 4. Pie chart
    ax4 = axes[1, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_labels)))
    wedges, texts, autotexts = ax4.pie(train_counts_sorted, labels=class_labels, 
                                       autopct='%1.1f%%', startangle=90,
                                       colors=colors, textprops={'fontsize': 10})
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax4.set_title(f'Training Set Distribution\n{total_train:,} total instances', 
                  fontsize=14, fontweight='bold')
    ax4.axis('equal')
    
    # Add statistics
    if total_train > 0:
        non_zero = [c for c in train_counts_sorted if c > 0]
        max_count = max(non_zero) if non_zero else 0
        min_count = min(non_zero) if non_zero else 0
        imbalance = max_count / min_count if min_count > 0 else 0
        
        stats_text = f"Statistics:\n"
        stats_text += f"Training files: {total_train:,}\n"
        stats_text += f"Test files: {total_test:,}\n"
        stats_text += f"Imbalance ratio: {imbalance:.1f}:1\n"
        stats_text += f"Classes with data: {len(non_zero)}/{len(class_labels)}"
    else:
        stats_text = "No data found!"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Add per-class counts
    class_details = "Per-class counts:\n"
    for class_label, train_count, test_count in zip(class_labels, train_counts_sorted, test_counts_sorted):
        if train_count > 0 or test_count > 0:
            class_details += f"{class_label}: {train_count} train, {test_count} test\n"
    
    fig.text(0.98, 0.02, class_details, fontsize=9, ha='right',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {output_file}")

def main():
    """Main function"""
    print("="*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)
    
    label_dir = './sawit/data/labels/VOC_format'
    
    # Step 1: Analyze file format
    format_info = analyze_label_file_format(label_dir, num_samples=10)
    
    # Step 2: Count classes from all files (or a large sample)
    print("\n" + "="*70)
    print("COUNTING CLASSES")
    print("="*70)
    
    # Process all files
    train_counts, total_files = count_all_classes(label_dir, class_map, max_files=None)
    
    if sum(train_counts.values()) == 0:
        print("\nERROR: No classes found in any files!")
        print("Please check the format of your .txt files.")
        return
    
    # Step 3: Find or create test split
    print("\n" + "="*70)
    print("TEST SPLIT")
    print("="*70)
    
    test_dir = find_test_split(label_dir)
    
    if test_dir and test_dir != "mixed":
        # We found a separate test directory
        test_counts, test_files = count_all_classes(test_dir, class_map, max_files=None)
    else:
        # Create simulated test split (20% of training)
        test_counts = create_train_test_split(train_counts, split_ratio=0.8)
        print(f"Created simulated test split with {sum(test_counts.values())} instances")
    
    # Step 4: Visualize
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
    visualize_distribution(train_counts, test_counts)
    
    # Step 5: Detailed report
    print("\n" + "="*70)
    print("DETAILED REPORT")
    print("="*70)
    
    print(f"\nTotal training files processed: {total_files}")
    print(f"Total training instances found: {sum(train_counts.values())}")
    print(f"Total test instances: {sum(test_counts.values())}")
    
    print(f"\nClass distribution:")
    for class_id in sorted(class_names.keys()):
        train_count = train_counts.get(class_id, 0)
        test_count = test_counts.get(class_id, 0)
        if train_count > 0 or test_count > 0:
            print(f"  {class_names[class_id]}: {train_count} train, {test_count} test")
    
    # Calculate and display imbalance
    non_zero_counts = [train_counts.get(i, 0) for i in class_names.keys() if train_counts.get(i, 0) > 0]
    if non_zero_counts:
        imbalance = max(non_zero_counts) / min(non_zero_counts)
        print(f"\nImbalance ratio (max/min): {imbalance:.2f}:1")
        
        # Suggest balancing strategies if highly imbalanced
        if imbalance > 10:
            print("\nWARNING: High class imbalance detected!")
            print("Consider using:")
            print("  - Class weighting in loss function")
            print("  - Oversampling minority classes")
            print("  - Data augmentation for minority classes")

if __name__ == "__main__":
    main()