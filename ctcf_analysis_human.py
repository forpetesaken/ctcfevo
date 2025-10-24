import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# Step 1: Run FIMO to scan genome
# ============================================

def run_fimo(motif_file, genome_fasta, output_dir="fimo_output", thresh=1e-4):
    """
    Run FIMO to scan genome for CTCF motifs
    """
    print(f"Running FIMO with threshold {thresh}...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run FIMO
    cmd = [
        "fimo",
        "--thresh", str(thresh),
        "--o", output_dir,
        motif_file,
        genome_fasta
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ FIMO completed. Results in {output_dir}/")
        return f"{output_dir}/fimo.tsv"
    except subprocess.CalledProcessError as e:
        print(f"✗ FIMO failed: {e.stderr}")
        return None
    except FileNotFoundError:
        print("✗ FIMO not found. Please install MEME Suite:")
        print("  conda install -c bioconda meme")
        return None

# ============================================
# Step 2: Convert FIMO output to BED format
# ============================================

def fimo_to_bed(fimo_tsv, output_bed="ctcf_motifs.bed"):
    """
    Convert FIMO TSV output to BED format with strand information
    """
    print(f"\nConverting FIMO output to BED format...")
    
    # Read FIMO output (skip comment lines)
    df = pd.read_csv(fimo_tsv, sep='\t', comment='#')
    
    # Create BED format: chr, start, end, name, score, strand
    bed_df = pd.DataFrame({
        'chr': df['sequence_name'],
        'start': df['start'] - 1,  # Convert to 0-based
        'end': df['stop'],
        'name': df['motif_id'],
        'score': df['score'],
        'strand': df['strand']
    })
    
    # Sort by chromosome and position
    bed_df = bed_df.sort_values(['chr', 'start'])
    
    # Save to BED file
    bed_df.to_csv(output_bed, sep='\t', header=False, index=False)
    
    print(f"✓ Found {len(bed_df)} CTCF motifs")
    print(f"✓ Saved to {output_bed}")
    
    return bed_df

# ============================================
# Step 3: Analyze CTCF directionality at TAD boundaries
# ============================================

def analyze_ctcf_directionality(tads_file, ctcf_bed_file, window=5000):
    """
    Analyze CTCF orientation at TAD boundaries
    Should get ~90% convergent in human
    """
    print(f"\n{'='*50}")
    print("CTCF Directionality Analysis")
    print(f"{'='*50}")
    
    # Load TADs
    tads = pd.read_csv(tads_file, sep='\t', 
                       names=['chr', 'start', 'end'],
                       comment='#')
    
    # Load CTCF motifs
    ctcf = pd.read_csv(ctcf_bed_file, sep='\t',
                       names=['chr', 'start', 'end', 'name', 'score', 'strand'])
    
    print(f"\nLoaded {len(tads)} TADs")
    print(f"Loaded {len(ctcf)} CTCF motifs")
    print(f"Window size: ±{window}bp around boundaries")
    
    results = {
        'convergent': 0,      # → ←  (forward at left, reverse at right)
        'divergent': 0,       # ← →  (reverse at left, forward at right)
        'tandem_forward': 0,  # → →
        'tandem_reverse': 0,  # ← ←
        'no_ctcf': 0,
        'single_side': 0
    }
    
    convergent_tads = []
    
    for idx, tad in tads.iterrows():
        if idx % 500 == 0:
            print(f"  Processing TAD {idx}/{len(tads)}...", end='\r')
        
        # Get CTCF at left boundary
        left_ctcf = ctcf[
            (ctcf['chr'] == tad['chr']) & 
            (ctcf['start'] >= tad['start'] - window) &
            (ctcf['end'] <= tad['start'] + window)
        ].sort_values('score', ascending=False)
        
        # Get CTCF at right boundary
        right_ctcf = ctcf[
            (ctcf['chr'] == tad['chr']) & 
            (ctcf['start'] >= tad['end'] - window) &
            (ctcf['end'] <= tad['end'] + window)
        ].sort_values('score', ascending=False)
        
        # Check if both boundaries have CTCF
        if len(left_ctcf) == 0 or len(right_ctcf) == 0:
            if len(left_ctcf) == 0 and len(right_ctcf) == 0:
                results['no_ctcf'] += 1
            else:
                results['single_side'] += 1
            continue
        
        # Get strongest motif at each side
        left_strand = left_ctcf.iloc[0]['strand']
        right_strand = right_ctcf.iloc[0]['strand']
        
        # Classify orientation
        if left_strand == '+' and right_strand == '-':
            results['convergent'] += 1
            convergent_tads.append(tad)
        elif left_strand == '-' and right_strand == '+':
            results['divergent'] += 1
        elif left_strand == '+' and right_strand == '+':
            results['tandem_forward'] += 1
        elif left_strand == '-' and right_strand == '-':
            results['tandem_reverse'] += 1
    
    print("\n")
    
    # Calculate percentages
    total_with_both = (results['convergent'] + results['divergent'] + 
                       results['tandem_forward'] + results['tandem_reverse'])
    
    if total_with_both > 0:
        conv_pct = (results['convergent'] / total_with_both) * 100
        
        print(f"\n{'='*50}")
        print("Results:")
        print(f"{'='*50}")
        print(f"Total TADs analyzed: {len(tads)}")
        print(f"TADs with CTCF at both boundaries: {total_with_both}")
        print(f"\nOrientation breakdown:")
        print(f"  Convergent (→ ←):  {results['convergent']:5d} ({conv_pct:.1f}%)")
        print(f"  Divergent (← →):   {results['divergent']:5d} ({results['divergent']/total_with_both*100:.1f}%)")
        print(f"  Tandem (→ →):      {results['tandem_forward']:5d} ({results['tandem_forward']/total_with_both*100:.1f}%)")
        print(f"  Tandem (← ←):      {results['tandem_reverse']:5d} ({results['tandem_reverse']/total_with_both*100:.1f}%)")
        print(f"\nTADs with CTCF on only one side: {results['single_side']}")
        print(f"TADs with no CTCF at boundaries: {results['no_ctcf']}")
        
        if conv_pct < 80:
            print(f"\n⚠️  WARNING: Convergent percentage is {conv_pct:.1f}%")
            print("Expected ~90% in human. Possible issues:")
            print("  - TAD boundaries may not be accurate")
            print("  - CTCF motif threshold may be too lenient (try --thresh 1e-5)")
            print("  - Window size may be too large")
            print("  - Using amphioxus or other non-mammalian data?")
        else:
            print(f"\n✅ Convergent percentage looks good for vertebrates!")
    
    return results

# ============================================
# Step 4: Visualize results
# ============================================

def plot_ctcf_orientations(results, output_file="ctcf_orientations.pdf"):
    """
    Create a bar plot of CTCF orientations
    """
    total_with_both = (results['convergent'] + results['divergent'] + 
                       results['tandem_forward'] + results['tandem_reverse'])
    
    if total_with_both == 0:
        print("No data to plot")
        return
    
    # Prepare data
    categories = ['Convergent\n(→ ←)', 'Divergent\n(← →)', 
                  'Tandem\n(→ →)', 'Tandem\n(← ←)']
    counts = [
        results['convergent'],
        results['divergent'],
        results['tandem_forward'],
        results['tandem_reverse']
    ]
    percentages = [c / total_with_both * 100 for c in counts]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    bars = ax.bar(categories, percentages, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, pct, count in zip(bars, percentages, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line at 90%
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, 
               label='Expected for mammals (~90%)')
    
    ax.set_ylabel('Percentage of TAD Pairs (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('CTCF Orientation', fontsize=12, fontweight='bold')
    ax.set_title('CTCF Motif Orientation at TAD Boundaries', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_file}")
    plt.show()

# ============================================
# Main pipeline
# ============================================

def main():
    # File paths - MODIFY THESE FOR YOUR DATA
    motif_file = "MA0139.1.meme"  # Your CTCF motif file
    genome_fasta = "hg38.fa"      # Your genome FASTA
    tads_file = "TADs.bed"        # Your TAD boundaries (chr, start, end)
    
    # Step 1: Run FIMO
    fimo_output = run_fimo(
        motif_file=motif_file,
        genome_fasta=genome_fasta,
        output_dir="fimo_output",
        thresh=1e-4  # Try 1e-5 for stricter threshold
    )
    
    if fimo_output is None:
        return
    
    # Step 2: Convert to BED
    ctcf_bed = fimo_to_bed(fimo_output, output_bed="ctcf_motifs.bed")
    
    # Step 3: Analyze directionality
    results = analyze_ctcf_directionality(
        tads_file=tads_file,
        ctcf_bed_file="ctcf_motifs.bed",
        window=5000
    )
    
    # Step 4: Visualize
    plot_ctcf_orientations(results)

if __name__ == "__main__":
    main()
    motif_file = "MA0139.1.meme"  # Your CTCF motif file
    genome_fasta = "hg38.fa"      # Your genome FASTA
    tads_file = "TADs.bed"        # Your TAD boundaries (chr, start, end)