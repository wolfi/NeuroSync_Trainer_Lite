import os
import pandas as pd

# Configuration
NUM_CHUNKS = 4  # Number of equal parts to split into

def split_csv_by_frames(csv_path, num_chunks):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Determine the total number of frames
    total_frames = len(df)
    
    # Calculate the size of each chunk
    chunk_size = total_frames // num_chunks
    remainder = total_frames % num_chunks

    # Pad the dataframe with mirrored values if necessary
    if remainder != 0:
        padding_size = num_chunks - remainder
        padding_df = df.iloc[-padding_size:].iloc[::-1]
        df = pd.concat([df, padding_df], ignore_index=True)
    
    # Recalculate the total frames after padding
    total_frames = len(df)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(csv_path), 'chunks')
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into equal chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else total_frames
        chunk_df = df[start:end]
        chunk_path = os.path.join(output_dir, f'chunk_{i+1}.csv')
        chunk_df.to_csv(chunk_path, index=False)
        print(f"Chunk {i+1} saved to {chunk_path}")