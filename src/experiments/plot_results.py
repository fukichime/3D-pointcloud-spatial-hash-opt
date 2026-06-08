import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(results_dir="experiment_results"):
    hash_data = []
    for f in os.listdir(results_dir):
        if f.startswith("hash_T") and f.endswith(".csv"):
            T = int(f.split("_T")[1].replace(".csv", ""))
            df = pd.read_csv(os.path.join(results_dir, f))
            mem = (T * 8 * 4) / (1024 * 1024)
            acc = df['val_acc'].max()
            collision = df['collision_rate'].mean() if 'collision_rate' in df.columns else 0.0
            hash_data.append({
                'Method': f'Hash T={T}', 
                'Memory_MB': mem, 
                'Accuracy': acc,
                'Collision_Rate': collision
            })
    
    cnn_file = os.path.join(results_dir, "cnn_baseline.csv")
    if os.path.exists(cnn_file):
        df = pd.read_csv(cnn_file)
        mem = df['hash_table_memory_mb'].iloc[0] if 'hash_table_memory_mb' in df.columns else 4.52
        acc = df['val_acc'].max()
        hash_data.append({
            'Method': '3D CNN', 
            'Memory_MB': mem, 
            'Accuracy': acc,
            'Collision_Rate': 0.0
        })
    
    df_results = pd.DataFrame(hash_data).sort_values('Memory_MB')
    
    plt.figure(figsize=(10, 6))
    
    hash_df = df_results[df_results['Method'] != '3D CNN'].sort_values('Memory_MB')
    cnn_df = df_results[df_results['Method'] == '3D CNN']
    
    plt.plot(hash_df['Memory_MB'], hash_df['Accuracy'], 'b-o', linewidth=2, markersize=8, label='Hash Method')
    
    if not cnn_df.empty:
        plt.scatter(cnn_df['Memory_MB'], cnn_df['Accuracy'], color='red', s=150, marker='s', 
                   label='3D CNN', zorder=5, edgecolors='darkred', linewidth=2)
    
    for _, row in df_results.iterrows():
        color = 'red' if row['Method'] == '3D CNN' else 'blue'
        plt.annotate(f"{row['Accuracy']:.3f}", (row['Memory_MB'], row['Accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, color=color)
    
    plt.xscale('log')
    plt.xlabel('Memory (MB) - Log Scale')
    plt.ylabel('Best Validation Accuracy')
    plt.title('Memory-Accuracy Trade-off')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'memory_accuracy_curve.png'), dpi=150)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for f in os.listdir(results_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(results_dir, f))
            label = f.replace(".csv", "")
            plt.plot(df['epoch'], df['val_acc'], label=label, linewidth=2.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150)
    plt.show()
    
    if len(hash_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(hash_df['Collision_Rate'], hash_df['Accuracy'], 'g-o', linewidth=2, markersize=8)
        for _, row in hash_df.iterrows():
            plt.annotate(f"{row['Method']}", (row['Collision_Rate'], row['Accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Collision Rate')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Impact of Collisions on Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'collision_impact.png'), dpi=150)
        plt.show()
    
    print("MEMORY VS ACCURACY VS COLLISION RATE COMPARISON")
    print("="*80)
    print(f"{'Method':<20} {'Memory (MB)':<15} {'Accuracy (%)':<15} {'Collision Rate':<15}")
    print("-"*80)
    for _, row in df_results.iterrows():
        coll_str = f"{row['Collision_Rate']:.1%}" if row['Collision_Rate'] > 0 else "0%"
        print(f"{row['Method']:<20} {row['Memory_MB']:<15.2f} {row['Accuracy']*100:<15.1f} {coll_str:<15}")

if __name__ == "__main__":
    plot_results()