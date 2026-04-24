import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import psutil
import gc
import time
from contextlib import contextmanager

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()  # If don't pass a parameter, it returns the object of the currently executing Python process
    memory_mb = process.memory_info().rss / 1024 / 1024.   # RSS: Resident Set Size: the actual amount of memory currently occupied by the process in physical memory.  B-KB-MB
    return memory_mb

@contextmanager
def memory_monitor(step_name):
    """Memory monitoring context manager"""
    start_memory = get_memory_usage()
    start_time = time.time()
    print(f"[{step_name}] begin - memory: {start_memory:.1f} MB")
    
    # try: execute the code in the 'with' block
    try:
        yield
    finally:
        end_memory = get_memory_usage()
        end_time = time.time()
        memory_diff = end_memory - start_memory
        time_diff = end_time - start_time
        print(f"[{step_name}] completed - memory: {end_memory:.1f} MB ({memory_diff:+.1f} MB), time elapsed: {time_diff:.2f}s")

def chunked_star_search(fits_path, ra0, dec0, radius_deg, gmag_limit, 
                       chunk_size=1000000, memory_limit_gb=8):
    """
    Astronomical Searches Using Block Processing for Massive FITS Files
    
    Parameters:
    -----------
    fits_path : str
        FITS file path
    ra0, dec0 : float
        Search center coordinates (degrees)
    radius_deg : float
        Search radius (degrees)
    gmag_limit : float
        Magnitude limit
    chunk_size : int
        Number of rows processed per chunk (default 1 million)
    memory_limit_gb : float
        Memory limit (GB)
    
    Returns:
    --------
    astropy.table.Table
        Filtered results
    """
    
    print(f"=== Block Processing for Massive Datasets ===")
    print(f"Memory limit: {memory_limit_gb} GB")
    print(f"Chunk size: {chunk_size:,} rows")
    
    # Precompute search parameters
    ra0_rad = np.radians(ra0)   # Replace them all with arcs
    dec0_rad = np.radians(dec0)
    radius_rad = np.radians(radius_deg)
    cos_dec0 = np.cos(dec0_rad)
    sin_dec0 = np.sin(dec0_rad)
    
    results = []
    total_found = 0
    
    with memory_monitor("File Open"):
        hdul = fits.open(fits_path, memmap=True)  # Memory Mapping
        data = hdul[1].data
        total_rows = len(data)
        
    print(f"Total rows: {total_rows:,}")
    
    # Dynamically adjust chunk_size
    available_memory_gb = psutil.virtual_memory().available / (1024**3)  # psutil.virtual_memory()：Retrieve the system's virtual memory information，.available：Retrieve the currently available physical memory (in bytes), convert bytes to GB
    if available_memory_gb < memory_limit_gb:
        chunk_size = min(chunk_size, int(chunk_size * available_memory_gb / memory_limit_gb))
        ## Therefore, the initially preset chunk_size and memory_limit_gb should be appropriately matched.
        print(f"Insufficient memory, adjusting chunk size to: {chunk_size:,} rows")
    
    try:
        # Block processing
        for start_idx in range(0, total_rows, chunk_size):  #(start(involve), stop(don't involve), step)
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_id = start_idx // chunk_size + 1
            total_chunks = (total_rows + chunk_size - 1) // chunk_size  # Convert floor division to ceiling division
            
            with memory_monitor(f"Chunk {chunk_id}/{total_chunks}"):
                # Read current chunk
                chunk_data = data[start_idx:end_idx]
                
                # Extract necessary columns
                ra_chunk = np.asarray(chunk_data['ra'], dtype=np.float64)
                dec_chunk = np.asarray(chunk_data['dec'], dtype=np.float64)
                wmag_chunk = np.asarray(chunk_data['wmag_tianyu_syn'], dtype=np.float64)
                
                # Quick pre-filtering: magnitude + rough position
                mag_mask = wmag_chunk <= gmag_limit
                
                # Rough rectangular filtering (quickly exclude most data)
                rough_ra_mask = np.abs(ra_chunk - ra0) < radius_deg
                rough_dec_mask = np.abs(dec_chunk - dec0) < radius_deg
                
                rough_mask = mag_mask & rough_ra_mask & rough_dec_mask
                rough_count = np.sum(rough_mask)
                
                if rough_count == 0:
                    print(f"  Chunk {chunk_id}: No candidates after rough filtering")
                    del chunk_data, ra_chunk, dec_chunk, wmag_chunk
                    gc.collect()  # Force Python's garbage collection mechanism
                    continue
                
                print(f"  Chunk {chunk_id}: Rough filtering retained {rough_count:,} candidate stars")
                
                # Precise position filtering for candidates
                ra_candidates = ra_chunk[rough_mask]
                dec_candidates = dec_chunk[rough_mask]
                wmag_candidates = wmag_chunk[rough_mask]
                
                # Precise spherical angular distance calculation
                ra_candidates_rad = np.radians(ra_candidates)
                dec_candidates_rad = np.radians(dec_candidates)
                
                if radius_deg < 5.0:  # Small radius fast calculation
                    delta_ra = ra_candidates_rad - ra0_rad
                    # Handle boundary crossing
                    delta_ra = np.where(delta_ra > np.pi, delta_ra - 2*np.pi, delta_ra)
                    delta_ra = np.where(delta_ra < -np.pi, delta_ra + 2*np.pi, delta_ra)
                    
                    angular_sep_sq = ((dec_candidates_rad - dec0_rad)**2 + 
                                     (cos_dec0 * np.cos(dec_candidates_rad) * delta_ra)**2)
                    precise_mask = angular_sep_sq < radius_rad**2
                else:  # Large radius precise calculation
                    cos_angular_sep = (sin_dec0 * np.sin(dec_candidates_rad) + 
                                      cos_dec0 * np.cos(dec_candidates_rad) * 
                                      np.cos(ra_candidates_rad - ra0_rad))
                    cos_angular_sep = np.clip(cos_angular_sep, -1, 1)
                    angular_sep = np.arccos(cos_angular_sep)
                    precise_mask = angular_sep < radius_rad
                
                chunk_found = np.sum(precise_mask)
                total_found += chunk_found
                
                if chunk_found > 0:
                    print(f"  Chunk {chunk_id}: Precise filtering found {chunk_found} stars")
                    
                    # Save results
                    result_dict = {
                        'ra': ra_candidates[precise_mask],
                        'dec': dec_candidates[precise_mask],
                        'phot_g_mean_mag': wmag_candidates[precise_mask]
                    }
                    
                    # Try to extract other columns
                    try:
                        wmag_err_chunk = np.asarray(chunk_data['wmag_tianyu_err_syn'][rough_mask][precise_mask], 
                                                   dtype=np.float64)
                        result_dict['phot_g_mean_mag_error'] = wmag_err_chunk
                    except:
                        pass
                    
                    try:
                        designation_chunk = chunk_data['designation'][rough_mask][precise_mask]
                        # Split each string, take the last part and convert to integer
                        source_id_chunk = [int(s.split()[-1]) for s in designation_chunk]
                        result_dict['SOURCE_ID'] = np.array(source_id_chunk, dtype=np.int64)
                    except (KeyError, IndexError, ValueError):
                        # If any exception occurs, skip this field
                        pass
                    
                    chunk_result = Table(result_dict)
                    results.append(chunk_result)
                else:
                    print(f"  Chunk {chunk_id}: No stars found after precise filtering")
                
                # Clear memory
                del (chunk_data, ra_chunk, dec_chunk, wmag_chunk, 
                     ra_candidates, dec_candidates, wmag_candidates)
                if 'ra_candidates_rad' in locals():
                    del ra_candidates_rad, dec_candidates_rad
                gc.collect()
                
                # Check memory usage
                current_memory = get_memory_usage()
                if current_memory > memory_limit_gb * 1024:
                    print(f"High memory usage: {current_memory:.1f} MB")
                    # If there are results, merge them once to free memory
                    if len(results) > 10:
                        print("Merging intermediate results to free memory...")
                        results = [vstack(results)]
                        gc.collect()
        
    finally:
        hdul.close()
    
    # Merge all results
    if results:
        with memory_monitor("Merging results"):
            final_result = vstack(results) if len(results) > 1 else results[0]
        
        print(f"\nSearch completed!")
        print(f"Total found: {total_found} stars")
        print(f"Final result: {len(final_result)} rows")
        
        return final_result
    else:
        print("\nNo stars found matching the criteria")
        return Table()

def streaming_star_search(fits_path, ra0, dec0, radius_deg, gmag_limit, 
                         output_file="search_results.fits"):
    """
    Streaming version - search and save simultaneously, most memory efficient
    
    Parameters:
    -----------
    fits_path : str
        Input FITS file path
    output_file : str
        Output results file path
    
    Returns:
    --------
    int
        Total number of stars found
    """
    
    print(f"=== Streaming search mode (most memory efficient) ===")
    
    # Precompute search parameters
    ra0_rad = np.radians(ra0)
    dec0_rad = np.radians(dec0) 
    radius_rad = np.radians(radius_deg)
    cos_dec0 = np.cos(dec0_rad)
    sin_dec0 = np.sin(dec0_rad)
    
    # Dynamically adjust chunk size
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    chunk_size = min(500000, int(available_memory_gb * 100000))  # Adjust based on available memory
    
    print(f"Available memory: {available_memory_gb:.1f} GB")
    print(f"Chunk size: {chunk_size:,} rows")
    
    total_found = 0
    output_tables = []
    
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        total_rows = len(data)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        print(f"Total data: {total_rows:,} rows, divided into {total_chunks} chunks")
        
        for chunk_id in range(total_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            
            print(f"Processing chunk {chunk_id+1}/{total_chunks}...", end=" ")
            
            # Read chunk
            chunk_data = data[start_idx:end_idx]
            
            # Quick filtering
            ra_chunk = np.asarray(chunk_data['ra'], dtype=np.float64)
            dec_chunk = np.asarray(chunk_data['dec'], dtype=np.float64)
            wmag_chunk = np.asarray(chunk_data['wmag_tianyu_syn'], dtype=np.float64)
            
            # Combine filter conditions
            mag_mask = wmag_chunk < gmag_limit
            rough_pos_mask = ((np.abs(ra_chunk - ra0) < radius_deg * 1) & 
                             (np.abs(dec_chunk - dec0) < radius_deg * 1))
            rough_mask = mag_mask & rough_pos_mask
            
            if np.sum(rough_mask) > 0:
                # Precise position filtering
                ra_cand = ra_chunk[rough_mask]
                dec_cand = dec_chunk[rough_mask]
                
                ra_cand_rad = np.radians(ra_cand)
                dec_cand_rad = np.radians(dec_cand)
                
                # Angular distance calculation
                if radius_deg < 5.0:
                    delta_ra = ra_cand_rad - ra0_rad
                    delta_ra = np.where(delta_ra > np.pi, delta_ra - 2*np.pi, delta_ra)
                    delta_ra = np.where(delta_ra < -np.pi, delta_ra + 2*np.pi, delta_ra)
                    
                    angular_sep_sq = ((dec_cand_rad - dec0_rad)**2 + 
                                     (cos_dec0 * np.cos(dec_cand_rad) * delta_ra)**2)
                    final_mask = angular_sep_sq < radius_rad**2
                else:
                    cos_sep = (sin_dec0 * np.sin(dec_cand_rad) + 
                              cos_dec0 * np.cos(dec_cand_rad) * 
                              np.cos(ra_cand_rad - ra0_rad))
                    cos_sep = np.clip(cos_sep, -1, 1)
                    final_mask = np.arccos(cos_sep) < radius_rad
                
                chunk_found = np.sum(final_mask)
                
                if chunk_found > 0:
                    # Creation results
                    rough_indices = np.where(rough_mask)[0]
                    final_indices = rough_indices[final_mask]
                    
                    result_data = {}
                    for col_name in ['ra', 'dec']:
                        if col_name in chunk_data.dtype.names:
                            result_data[col_name] = chunk_data[col_name][final_indices]
                    result_data['phot_g_mean_mag'] = chunk_data['wmag_tianyu_syn'][final_indices]
                    result_data['phot_g_mean_mag_error'] = chunk_data['wmag_tianyu_err_syn'][final_indices]
                    designation = chunk_data['designation'][final_indices]
                    source_id_chunk = [int(s.split()[-1]) for s in designation]
                    result_data['SOURCE_ID'] = np.array(source_id_chunk, dtype=np.int64)

                    chunk_table = Table(result_data)
                    output_tables.append(chunk_table)  # output_tables used for periodic saving
                    total_found += chunk_found  # used for final saving
                    
                    print(f"Found {chunk_found} stars")
                else:
                    print("No results")
            else:
                print("No candidates")
            
            # Clear memory
            del chunk_data, ra_chunk, dec_chunk, wmag_chunk
            gc.collect()
            
            # Periodically save intermediate results
            if len(output_tables) >= 20:  # Save every 20 chunks
                print("  Saving intermediate results...")
                intermediate_result = vstack(output_tables)
                if chunk_id == 19:  # First save
                    intermediate_result.write(output_file, overwrite=True)
                else:  # Append save
                    existing = Table.read(output_file)
                    combined = vstack([existing, intermediate_result])
                    combined.write(output_file, overwrite=True)
                output_tables = []
                gc.collect()
    
    # Save the final result
    if output_tables:
        final_chunk = vstack(output_tables)
        if total_found > len(final_chunk):  # Previously saved
            existing = Table.read(output_file)
            final_result = vstack([existing, final_chunk])
        else:
            final_result = final_chunk
        final_result.write(output_file, overwrite=True)
    
    print(f"\n Streaming search completed!")
    print(f"Total found: {total_found} stars")
    print(f"Results saved to: {output_file}")

    return final_result

# Usage example
if __name__ == "__main__":
    fits_path = '/Users/kexin_li/Documents/vs_py/tianyu_parameters/reference_star_catalog/ref_star_catalog/all_Tianyu_standards_North.fits'
    
    # Method 1: Chunked processing (recommended)
    print("=== Starting chunked search ===")
    start_time = time.time()
    result = chunked_star_search(
        fits_path=fits_path,
        ra0=47.36893066,
        dec0=30.67357305,
        radius_deg=0.484,
        gmag_limit=16.0,
        chunk_size=5000000,  # 5 million rows per chunk
        memory_limit_gb=2   # 2GB memory limit
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    if len(result) > 0:
        print(f"Sample search results:")
        print(result[:5])
        
        # Save results
        result.write('search_wasp_11_results.fits', overwrite=True)
        print("Results saved to .fits file")
    
    # # Method 2: Streaming processing (minimal memory)
    # streaming_star_search(
    #     fits_path=fits_path,
    #     ra0=120.0,
    #     dec0=40.0,
    #     radius_deg=1.0,
    #     gmag_limit=18.0,
    #     output_file="streaming_results.fits"
    # )