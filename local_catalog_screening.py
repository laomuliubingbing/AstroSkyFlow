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
    print(f"[{step_name}] 开始 - 内存: {start_memory:.1f} MB")
    
    # try: execute the code in the 'with' block
    try:
        yield
    finally:
        end_memory = get_memory_usage()
        end_time = time.time()
        memory_diff = end_memory - start_memory
        time_diff = end_time - start_time
        print(f"[{step_name}] 完成 - 内存: {end_memory:.1f} MB ({memory_diff:+.1f} MB), 耗时: {time_diff:.2f}s")

def chunked_star_search(fits_path, ra0, dec0, radius_deg, gmag_limit, 
                       chunk_size=1000000, memory_limit_gb=8):
    """
    分块处理超大FITS文件的天体搜索
    
    Parameters:
    -----------
    fits_path : str
        FITS文件路径
    ra0, dec0 : float
        搜索中心坐标（度）
    radius_deg : float
        搜索半径（度）
    gmag_limit : float
        星等限制
    chunk_size : int
        每次处理的行数（默认100万行）
    memory_limit_gb : float
        内存限制（GB）
    
    Returns:
    --------
    astropy.table.Table
        筛选后的结果
    """
    
    print(f"=== 超大数据集分块搜索 ===")
    print(f"内存限制: {memory_limit_gb} GB")
    print(f"分块大小: {chunk_size:,} 行")
    
    # 预计算搜索参数
    ra0_rad = np.radians(ra0)   # Replace them all with arcs
    dec0_rad = np.radians(dec0)
    radius_rad = np.radians(radius_deg)
    cos_dec0 = np.cos(dec0_rad)
    sin_dec0 = np.sin(dec0_rad)
    
    results = []
    total_found = 0
    
    with memory_monitor("文件打开"):
        hdul = fits.open(fits_path, memmap=True)  # Memory Mapping
        data = hdul[1].data
        total_rows = len(data)
        
    print(f"总数据量: {total_rows:,} 行")
    
    # 动态调整chunk_size
    available_memory_gb = psutil.virtual_memory().available / (1024**3)  # psutil.virtual_memory()：获取系统的虚拟内存信息，.available：获取当前可用的物理内存量（单位：字节），将字节转换为GB
    if available_memory_gb < memory_limit_gb:
        chunk_size = min(chunk_size, int(chunk_size * available_memory_gb / memory_limit_gb))
        ## 所以一开始预设的chunk_size和memory_limit_gb应该是对应好
        print(f"内存不足，调整分块大小为: {chunk_size:,} 行")
    
    try:
        # 分块处理
        for start_idx in range(0, total_rows, chunk_size):  #(start(involve), stop(don't involve), step)
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_id = start_idx // chunk_size + 1
            total_chunks = (total_rows + chunk_size - 1) // chunk_size  # 将向下取整变为向上取整
            
            with memory_monitor(f"分块 {chunk_id}/{total_chunks}"):
                # 读取当前分块
                chunk_data = data[start_idx:end_idx]
                
                # 提取必要的列
                ra_chunk = np.asarray(chunk_data['ra'], dtype=np.float64)
                dec_chunk = np.asarray(chunk_data['dec'], dtype=np.float64)
                wmag_chunk = np.asarray(chunk_data['wmag_tianyu_syn'], dtype=np.float64)
                
                # 快速预筛选：星等 + 粗略位置
                mag_mask = wmag_chunk <= gmag_limit
                
                # 粗略的矩形筛选（快速排除大部分数据）
                rough_ra_mask = np.abs(ra_chunk - ra0) < radius_deg
                rough_dec_mask = np.abs(dec_chunk - dec0) < radius_deg
                
                rough_mask = mag_mask & rough_ra_mask & rough_dec_mask
                rough_count = np.sum(rough_mask)
                
                if rough_count == 0:
                    print(f"  分块 {chunk_id}: 粗筛后无候选天体")
                    del chunk_data, ra_chunk, dec_chunk, wmag_chunk
                    gc.collect()  # 强制启动 Python 的垃圾回收（garbage collection）机制
                    continue
                
                print(f"分块 {chunk_id}: 粗筛保留 {rough_count:,} 颗候选天体")
                
                # 对候选天体进行精确位置筛选
                ra_candidates = ra_chunk[rough_mask]
                dec_candidates = dec_chunk[rough_mask]
                wmag_candidates = wmag_chunk[rough_mask]
                
                # 精确的球面角距离计算
                ra_candidates_rad = np.radians(ra_candidates)
                dec_candidates_rad = np.radians(dec_candidates)
                
                if radius_deg < 5.0:  # 小半径快速计算
                    delta_ra = ra_candidates_rad - ra0_rad
                    # 处理跨越边界
                    delta_ra = np.where(delta_ra > np.pi, delta_ra - 2*np.pi, delta_ra)
                    delta_ra = np.where(delta_ra < -np.pi, delta_ra + 2*np.pi, delta_ra)
                    
                    angular_sep_sq = ((dec_candidates_rad - dec0_rad)**2 + 
                                     (cos_dec0 * np.cos(dec_candidates_rad) * delta_ra)**2)
                    precise_mask = angular_sep_sq < radius_rad**2
                else:  # 大半径精确计算
                    cos_angular_sep = (sin_dec0 * np.sin(dec_candidates_rad) + 
                                      cos_dec0 * np.cos(dec_candidates_rad) * 
                                      np.cos(ra_candidates_rad - ra0_rad))
                    cos_angular_sep = np.clip(cos_angular_sep, -1, 1)
                    angular_sep = np.arccos(cos_angular_sep)
                    precise_mask = angular_sep < radius_rad
                
                chunk_found = np.sum(precise_mask)
                total_found += chunk_found
                
                if chunk_found > 0:
                    print(f"  分块 {chunk_id}: 精筛找到 {chunk_found} 颗天体")
                    
                    # 保存结果
                    result_dict = {
                        'ra': ra_candidates[precise_mask],
                        'dec': dec_candidates[precise_mask],
                        'phot_g_mean_mag': wmag_candidates[precise_mask]
                    }
                    
                    # 尝试提取其他列
                    try:
                        wmag_err_chunk = np.asarray(chunk_data['wmag_tianyu_err_syn'][rough_mask][precise_mask], 
                                                   dtype=np.float64)
                        result_dict['phot_g_mean_mag_error'] = wmag_err_chunk
                    except:
                        pass
                    
                    try:
                        designation_chunk = chunk_data['designation'][rough_mask][precise_mask]
                        # 拆分每个字符串，取最后一节并转为整数
                        source_id_chunk = [int(s.split()[-1]) for s in designation_chunk]
                        result_dict['SOURCE_ID'] = np.array(source_id_chunk, dtype=np.int64)
                    except (KeyError, IndexError, ValueError):
                        # 如果有任何异常，就跳过这个字段
                        pass
                    
                    chunk_result = Table(result_dict)
                    results.append(chunk_result)
                else:
                    print(f"  分块 {chunk_id}: 精筛后无天体")
                
                # 清理内存
                del (chunk_data, ra_chunk, dec_chunk, wmag_chunk, 
                     ra_candidates, dec_candidates, wmag_candidates)
                if 'ra_candidates_rad' in locals():
                    del ra_candidates_rad, dec_candidates_rad
                gc.collect()
                
                # 检查内存使用
                current_memory = get_memory_usage()
                if current_memory > memory_limit_gb * 1024:
                    print(f"内存使用过高: {current_memory:.1f} MB")
                    # 如果有结果，先合并一次以释放内存
                    if len(results) > 10:
                        print("合并中间结果以释放内存...")
                        results = [vstack(results)]
                        gc.collect()
        
    finally:
        hdul.close()
    
    # 合并所有结果
    if results:
        with memory_monitor("合并结果"):
            final_result = vstack(results) if len(results) > 1 else results[0]
        
        print(f"\n搜索完成!")
        print(f"总计找到: {total_found} 颗天体")
        print(f"最终结果: {len(final_result)} 行")
        
        return final_result
    else:
        print("\n未找到符合条件的天体")
        return Table()

def streaming_star_search(fits_path, ra0, dec0, radius_deg, gmag_limit, 
                         output_file="search_results.fits"):
    """
    流式处理版本 - 边搜索边保存，最节省内存
    
    Parameters:
    -----------
    fits_path : str
        输入FITS文件路径
    output_file : str
        输出结果文件路径
    
    Returns:
    --------
    int
        找到的天体总数
    """
    
    print(f"=== 流式搜索模式（最节省内存）===")
    
    # 预计算
    ra0_rad = np.radians(ra0)
    dec0_rad = np.radians(dec0) 
    radius_rad = np.radians(radius_deg)
    cos_dec0 = np.cos(dec0_rad)
    sin_dec0 = np.sin(dec0_rad)
    
    # 动态调整分块大小
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    chunk_size = min(500000, int(available_memory_gb * 100000))  # 根据可用内存调整
    
    print(f"可用内存: {available_memory_gb:.1f} GB")
    print(f"分块大小: {chunk_size:,} 行")
    
    total_found = 0
    output_tables = []
    
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        total_rows = len(data)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        print(f"总数据: {total_rows:,} 行，分为 {total_chunks} 块")
        
        for chunk_id in range(total_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            
            print(f"处理分块 {chunk_id+1}/{total_chunks}...", end=" ")
            
            # 读取分块
            chunk_data = data[start_idx:end_idx]
            
            # 快速筛选
            ra_chunk = np.asarray(chunk_data['ra'], dtype=np.float64)
            dec_chunk = np.asarray(chunk_data['dec'], dtype=np.float64)
            wmag_chunk = np.asarray(chunk_data['wmag_tianyu_syn'], dtype=np.float64)
            
            # 组合筛选条件
            mag_mask = wmag_chunk < gmag_limit
            rough_pos_mask = ((np.abs(ra_chunk - ra0) < radius_deg * 1) & 
                             (np.abs(dec_chunk - dec0) < radius_deg * 1))
            rough_mask = mag_mask & rough_pos_mask
            
            if np.sum(rough_mask) > 0:
                # 精确位置筛选
                ra_cand = ra_chunk[rough_mask]
                dec_cand = dec_chunk[rough_mask]
                
                ra_cand_rad = np.radians(ra_cand)
                dec_cand_rad = np.radians(dec_cand)
                
                # 角距离计算
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
                    # 创建结果
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
                    output_tables.append(chunk_table)  # output_tables用于定期保存
                    total_found += chunk_found  # 用于最终保存结果
                    
                    print(f"找到 {chunk_found} 颗")
                else:
                    print("无结果")
            else:
                print("无候选")
            
            # 清理内存
            del chunk_data, ra_chunk, dec_chunk, wmag_chunk
            gc.collect()
            
            # 定期保存中间结果
            if len(output_tables) >= 20:  # 每20个分块保存一次
                print("  保存中间结果...")
                intermediate_result = vstack(output_tables)
                if chunk_id == 19:  # 第一次保存
                    intermediate_result.write(output_file, overwrite=True)
                else:  # 追加保存
                    existing = Table.read(output_file)
                    combined = vstack([existing, intermediate_result])
                    combined.write(output_file, overwrite=True)
                output_tables = []
                gc.collect()
    
    # 保存最终结果
    if output_tables:
        final_chunk = vstack(output_tables)
        if total_found > len(final_chunk):  # 之前有保存过
            existing = Table.read(output_file)
            final_result = vstack([existing, final_chunk])
        else:
            final_result = final_chunk
        final_result.write(output_file, overwrite=True)
    
    print(f"\n🎉 流式搜索完成!")
    print(f"总计找到: {total_found} 颗天体")
    print(f"结果已保存到: {output_file}")

    return final_result

# 使用示例
if __name__ == "__main__":
    fits_path = '/Users/kexin_li/Documents/vs_py/tianyu_parameters/reference_star_catalog/ref_star_catalog/all_Tianyu_standards_North.fits'
    
    # 方案1: 分块处理（推荐）
    print("=== 开始分块搜索 ===")
    start_time = time.time()
    result = chunked_star_search(
        fits_path=fits_path,
        ra0=47.36893066,
        dec0=30.67357305,
        radius_deg=0.484,
        gmag_limit=16.0,
        chunk_size=5000000,  # 500万行一块
        memory_limit_gb=2   # 2GB内存限制
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总用时: {elapsed_time:.2f} 秒")
    
    if len(result) > 0:
        print(f"搜索结果样例:")
        print(result[:5])
        
        # 保存结果
        result.write('search_wasp_11_results.fits', overwrite=True)
        print("结果已保存到.fits文件")
    
    # # 方案2: 流式处理（内存最少）
    # streaming_star_search(
    #     fits_path=fits_path,
    #     ra0=120.0,
    #     dec0=40.0,
    #     radius_deg=1.0,
    #     gmag_limit=18.0,
    #     output_file="streaming_results.fits"
    # )