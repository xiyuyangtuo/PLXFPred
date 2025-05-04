import os
import re
import logging
from collections import defaultdict
import numpy as np

def extract_sample_weights(log_file):
    samples = defaultdict(lambda: {
        'protein_node_weights': defaultdict(list),
        'protein_edge_weights': defaultdict(list),
        'max_position': -1  # Used to infer sequence length
    })

    with open(log_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match "Sample <ID>: Protein edge attention weights" or "Sample <ID>: Protein node attention weights"
        edge_section_match = re.search(r'Sample (\S+): Protein edge attention weights', line)
        node_section_match = re.search(r'Sample (\S+): Protein node attention weights', line)

        if edge_section_match:
            sample_id = edge_section_match.group(1)  # Keep as string
            logging.debug(f"Found protein edge section for Sample {sample_id}")

            # Read subsequent edge weights until a non-matching line is encountered
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                # Match multi-head weights format
                edge_match = re.search(r'Edge:\s*(\w)(\d+)-(\w)(\d+),\s*Attention Weights:\s*Head \d+:\s*([\d.]+),\s*Head \d+:\s*([\d.]+),\s*Head \d+:\s*([\d.]+),\s*Head \d+:\s*([\d.]+)', next_line)
                if edge_match:
                    aa1, pos1, aa2, pos2 = edge_match.group(1), int(edge_match.group(2)), edge_match.group(3), int(edge_match.group(4))
                    # Update maximum position
                    samples[sample_id]['max_position'] = max(samples[sample_id]['max_position'], pos1, pos2)
                    # Check for self-loop edges
                    if aa1 == aa2 and pos1 == pos2:
                        logging.debug(f"Sample {sample_id}: Skipping self-loop edge {aa1}{pos1}-{aa2}{pos2}")
                        i += 1
                        continue
                    # Exclude edges involving the start (pos1 == 0 or pos2 == 0)
                    if pos1 == 0 or pos2 == 0:
                        logging.debug(f"Sample {sample_id}: Skipping edge {aa1}{pos1}-{aa2}{pos2} involving start")
                        i += 1
                        continue
                    weights = [float(edge_match.group(i)) for i in range(5, 9)]  # Extract weights for four heads
                    edge_key = tuple(sorted([(aa1, pos1), (aa2, pos2)]))
                    samples[sample_id]['protein_edge_weights'][edge_key].append(weights)  # Store the complete list of weights
                    logging.debug(f"Sample {sample_id}: Edge {aa1}{pos1}-{aa2}{pos2} added with weights {weights}")
                    i += 1
                else:
                    logging.debug(f"Sample {sample_id}: Edge line not matched: {next_line}")
                    break

        elif node_section_match:
            sample_id = node_section_match.group(1)  # Keep as string
            logging.debug(f"Found protein node section for Sample {sample_id}")

            # Read subsequent node weights until a non-matching line is encountered
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                # Match multi-head weights format
                node_match = re.search(r'Node:\s*(\w)(\d+),\s*Attention Weights:\s*Head \d+:\s*([\d.]+),\s*Head \d+:\s*([\d.]+),\s*Head \d+:\s*([\d.]+),\s*Head \d+:\s*([\d.]+)', next_line)
                if node_match:
                    aa_type, position = node_match.group(1), int(node_match.group(2))
                    # Update maximum position
                    samples[sample_id]['max_position'] = max(samples[sample_id]['max_position'], position)
                    # Exclude nodes at the start (position == 0)
                    if position == 0:
                        logging.debug(f"Sample {sample_id}: Skipping node {aa_type}{position} at start")
                        i += 1
                        continue
                    weights = [float(node_match.group(i)) for i in range(3, 7)]  # Extract weights for four heads
                    samples[sample_id]['protein_node_weights'][(aa_type, position)].append(weights)  # Store the complete list of weights
                    logging.debug(f"Sample {sample_id}: Node {aa_type}{position} added with weights {weights}")
                    i += 1
                else:
                    logging.debug(f"Sample {sample_id}: Node line not matched: {next_line}")
                    break

        else:
            i += 1

    # Process endpoint filtering
    for sample_id, data in samples.items():
        max_pos = data['max_position']
        if max_pos >= 0:  # Ensure valid position data is present
            seq_len = max_pos + 1  # Sequence length is maximum position plus one (zero-based indexing)
            logging.debug(f"Sample {sample_id}: Inferred sequence length = {seq_len}")
            # Filter edges involving the endpoint
            filtered_edges = {}
            for edge_key, weights_list in data['protein_edge_weights'].items():
                aa1, pos1 = edge_key[0]
                aa2, pos2 = edge_key[1]
                if pos1 == max_pos or pos2 == max_pos:
                    logging.debug(f"Sample {sample_id}: Skipping edge {aa1}{pos1}-{aa2}{pos2} involving end (pos={max_pos})")
                    continue
                filtered_edges[edge_key] = weights_list
            data['protein_edge_weights'] = filtered_edges
            # Filter nodes involving the endpoint
            filtered_nodes = {}
            for (aa_type, position), weights_list in data['protein_node_weights'].items():
                if position == max_pos:
                    logging.debug(f"Sample {sample_id}: Skipping node {aa_type}{position} at end (pos={max_pos})")
                    continue
                filtered_nodes[(aa_type, position)] = weights_list
            data['protein_node_weights'] = filtered_nodes
        else:
            logging.warning(f"Sample {sample_id}: No valid positions found, cannot determine sequence length")

        logging.info(f"Sample {sample_id}: Extracted {len(data['protein_node_weights'])} nodes, {len(data['protein_edge_weights'])} edges, Inferred max position: {max_pos}")
    return samples

def calculate_sample_stats(samples):
    """Calculate statistics for protein nodes and edges in each sample based on the complete list of multi-head weights"""
    sample_stats = {}
    for sample_id, data in samples.items():
        # Node statistics
        node_stats = {}
        for (aa_type, position), weights_list in data['protein_node_weights'].items():
            # Flatten all weight lists into a single 1D array
            all_weights = np.concatenate(weights_list)
            node_stats[(aa_type, position)] = {
                'avg_weight': np.mean(all_weights),
                'max_weight': np.max(all_weights),
                'count': len(weights_list)  # Count is still based on the number of weight list occurrences
            }

        # Edge statistics
        edge_stats = {}
        for edge_key, weights_list in data['protein_edge_weights'].items():
            aa1, pos1 = edge_key[0]
            aa2, pos2 = edge_key[1]
            # Flatten all weight lists into a single 1D array
            all_weights = np.concatenate(weights_list)
            edge_stats[(pos1, pos2)] = {
                'aa1': aa1,
                'pos1': pos1,
                'aa2': aa2,
                'pos2': pos2,
                'avg_weight': np.mean(all_weights),
                'max_weight': np.max(all_weights),
                'count': len(weights_list)  # Count is still based on the number of weight list occurrences
            }

        sample_stats[sample_id] = {
            'node_stats': node_stats,
            'edge_stats': edge_stats
        }

    return sample_stats

def generate_formatted_report(sample_stats, output_file):
    """Generate a formatted protein statistics report, outputting only the top ten nodes and edges statistics"""
    with open(output_file, 'w') as f:
        f.write("Protein Weight Statistics Report by Sample (Excluding Self-Loops, Start, and End Nodes/Edges)\n\n")
        
        # Sort by sample_id (if sample_id is a string, sort alphabetically)
        for sample_id in sorted(sample_stats.keys()):
            stats = sample_stats[sample_id]
            f.write(f"Sample {sample_id}\n")
            f.write("---\n")
            
            # Top 10 node statistics (sorted by Avg Weight)
            f.write("Top 10 Protein Nodes by Avg Weight\n")
            f.write("---\n")
            f.write("Node: AA Position | Avg Weight | Max Weight | Count\n")
            f.write("---\n")
            top_avg_nodes = sorted(stats['node_stats'].items(), key=lambda x: x[1]['avg_weight'], reverse=True)[:10]
            for (aa_type, position), node_info in top_avg_nodes:
                f.write(f"Node: {aa_type}{position:3d} | {node_info['avg_weight']:10.4f} | {node_info['max_weight']:10.4f} | {node_info['count']:5d}\n")
            
            # Top 10 node statistics (sorted by Max Weight)
            f.write("\nTop 10 Protein Nodes by Max Weight\n")
            f.write("---\n")
            f.write("Node: AA Position | Avg Weight | Max Weight | Count\n")
            f.write("---\n")
            top_max_nodes = sorted(stats['node_stats'].items(), key=lambda x: x[1]['max_weight'], reverse=True)[:10]
            for (aa_type, position), node_info in top_max_nodes:
                f.write(f"Node: {aa_type}{position:3d} | {node_info['avg_weight']:10.4f} | {node_info['max_weight']:10.4f} | {node_info['count']:5d}\n")
            
            # Top 10 node statistics (sorted by Count)
            f.write("\nTop 10 Protein Nodes by Count\n")
            f.write("---\n")
            f.write("Node: AA Position | Avg Weight | Max Weight | Count\n")
            f.write("---\n")
            top_count_nodes = sorted(stats['node_stats'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]
            for (aa_type, position), node_info in top_count_nodes:
                f.write(f"Node: {aa_type}{position:3d} | {node_info['avg_weight']:10.4f} | {node_info['max_weight']:10.4f} | {node_info['count']:5d}\n")
            
            # Top 10 edge statistics (sorted by Avg Weight)
            f.write("\nTop 10 Protein Edges by Avg Weight\n")
            f.write("---\n")
            f.write("Edge: AA1Pos1-AA2Pos2 | Avg Weight | Max Weight | Count\n")
            f.write("---\n")
            top_avg_edges = sorted(stats['edge_stats'].items(), key=lambda x: x[1]['avg_weight'], reverse=True)[:10]
            for (pos1, pos2), edge_info in top_avg_edges:
                f.write(f"Edge: {edge_info['aa1']}{edge_info['pos1']:3d}-{edge_info['aa2']}{edge_info['pos2']:3d} | {edge_info['avg_weight']:10.4f} | {edge_info['max_weight']:10.4f} | {edge_info['count']:5d}\n")
            
            # Top 10 edge statistics (sorted by Max Weight)
            f.write("\nTop 10 Protein Edges by Max Weight\n")
            f.write("---\n")
            f.write("Edge: AA1Pos1-AA2Pos2 | Avg Weight | Max Weight | Count\n")
            f.write("---\n")
            top_max_edges = sorted(stats['edge_stats'].items(), key=lambda x: x[1]['max_weight'], reverse=True)[:10]
            for (pos1, pos2), edge_info in top_max_edges:
                f.write(f"Edge: {edge_info['aa1']}{edge_info['pos1']:3d}-{edge_info['aa2']}{edge_info['pos2']:3d} | {edge_info['avg_weight']:10.4f} | {edge_info['max_weight']:10.4f} | {edge_info['count']:5d}\n")
            
            # Top 10 edge statistics (sorted by Count)
            f.write("\nTop 10 Protein Edges by Count\n")
            f.write("---\n")
            f.write("Edge: AA1Pos1-AA2Pos2 | Avg Weight | Max Weight | Count\n")
            f.write("---\n")
            top_count_edges = sorted(stats['edge_stats'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]
            for (pos1, pos2), edge_info in top_count_edges:
                f.write(f"Edge: {edge_info['aa1']}{edge_info['pos1']:3d}-{edge_info['aa2']}{edge_info['pos2']:3d} | {edge_info['avg_weight']:10.4f} | {edge_info['max_weight']:10.4f} | {edge_info['count']:5d}\n")
            
            f.write("\n\n")  # Two blank lines between samples

    logging.info(f"Formatted protein weight statistics report saved to: {output_file}")

def process_weights():
    log_dir = "/home/nc307/workspace/PDBbind/script/test/logs/prediction1"
    log_file = os.path.join(log_dir, "prediction_log.txt")
    report_file = os.path.join(log_dir, "formatted_protein_weight_report.txt")
    
    logging.basicConfig(filename=os.path.join(log_dir, 'protein_weight_processing.log'),
                       level=logging.DEBUG,
                       format='%(asctime)s [%(levelname)s] %(message)s')
    
    try:
        samples = extract_sample_weights(log_file)
        logging.info(f"Extracted protein data for {len(samples)} samples")
        
        if not samples:
            logging.warning("No protein data extracted. Exiting.")
            return
        
        sample_stats = calculate_sample_stats(samples)
        
        generate_formatted_report(sample_stats, report_file)
        
    except Exception as e:
        logging.error(f"Error processing protein weights: {e}")
        raise

if __name__ == "__main__":
    process_weights()