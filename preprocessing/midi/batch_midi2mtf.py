import os
import math
import mido
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Convert MIDI files to MTF format.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the folder containing MIDI (.midi, .mid) files"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the folder where converted MTF files will be saved"
    )
    parser.add_argument(
        "--m3_compatible",
        action="store_true",
        help="Enable M3 compatibility (remove metadata like text, copyright, lyrics, etc.)"
    )
    parser.add_argument(
        "--max_messages",
        type=int,
        default=None,
        help="Maximum number of MIDI messages to include (truncate longer files)"
    )
    parser.add_argument(
        "--max_ticks",
        type=int,
        default=None,
        help="Maximum number of ticks (truncate by time)"
    )
    return parser.parse_args()

def msg_to_str(msg):
    str_msg = ""
    for key, value in msg.dict().items():
        str_msg += " " + str(value)
    return str_msg.strip().encode('unicode_escape').decode('utf-8')

def load_midi(filename, m3_compatible, max_messages=None, max_ticks=None):
    """
    Load a MIDI file and convert it to MTF format.
    
    Args:
        filename: Path to MIDI file
        m3_compatible: Remove metadata for M3 compatibility
        max_messages: Maximum number of messages to include (None = no limit)
        max_ticks: Maximum tick time to include (None = no limit)
    """
    mid = mido.MidiFile(filename)
    msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

    current_tick = 0
    message_count = 0
    
    # Traverse the MIDI file
    for msg in mid.merged_track:
        # Track cumulative time
        if hasattr(msg, 'time'):
            current_tick += msg.time
        
        # Check tick limit
        if max_ticks is not None and current_tick > max_ticks:
            break
        
        # Check message limit
        if max_messages is not None and message_count >= max_messages:
            break
        
        if m3_compatible:
            if msg.is_meta:
                if msg.type in ["text", "copyright", "track_name", "instrument_name", 
                                "lyrics", "marker", "cue_marker", "device_name"]:
                    continue
        
        str_msg = msg_to_str(msg)
        msg_list.append(str_msg)
        message_count += 1
    
    return "\n".join(msg_list)

def convert_midi2mtf(file_list, input_dir, output_dir, m3_compatible, max_messages=None, max_ticks=None, progress_dict=None, lock=None, process_id=0):
    """
    Converts MIDI files to MTF format.
    """
    for file in file_list:
        # Construct the output directory by replacing input_dir with output_dir
        relative_path = os.path.relpath(os.path.dirname(file), input_dir)
        output_folder = os.path.join(output_dir, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        try:
            output = load_midi(file, m3_compatible, max_messages, max_ticks)

            if not output:
                with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                    f.write(file + '\n')
                continue
            else:
                output_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '.mtf')
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                f.write(file + " " + str(e) + '\n')
            pass
        
        # Update shared progress counter
        if progress_dict is not None and lock is not None:
            with lock:
                progress_dict['count'] += 1

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)  # Ensure absolute path
    output_dir = os.path.abspath(args.output_dir)  # Ensure absolute path
    m3_compatible = args.m3_compatible  # Get M3 compatibility flag
    max_messages = args.max_messages  # Get max messages limit
    max_ticks = args.max_ticks  # Get max ticks limit

    file_list = []
    os.makedirs("logs", exist_ok=True)

    print("Scanning for MIDI files...")
    # Traverse the specified input folder for MIDI files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith((".mid", ".midi")):
                continue
            filename = os.path.join(root, file).replace("\\", "/")
            file_list.append(filename)

    print(f"Found {len(file_list)} MIDI files")
    
    # Prepare for multiprocessing
    num_processes = os.cpu_count()
    print(f"Using {num_processes} CPU cores")
    
    file_lists = []
    random.shuffle(file_list)
    for i in range(num_processes):
        start_idx = int(math.floor(i * len(file_list) / num_processes))
        end_idx = int(math.floor((i + 1) * len(file_list) / num_processes))
        file_lists.append(file_list[start_idx:end_idx])

    # Create shared progress tracking
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['count'] = 0
    lock = manager.Lock()

    # Use multiprocessing to speed up conversion
    pool = Pool(processes=num_processes)
    
    # Start async conversion
    result = pool.starmap_async(
        convert_midi2mtf, 
        [(file_list_chunk, input_dir, output_dir, m3_compatible, max_messages, max_ticks, progress_dict, lock, i) 
         for i, file_list_chunk in enumerate(file_lists)]
    )
    
    # Monitor progress with a single progress bar
    with tqdm(total=len(file_list), desc="Converting MIDI to MTF") as pbar:
        last_count = 0
        while not result.ready():
            current_count = progress_dict['count']
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            result.wait(0.1)
        
        # Final update
        current_count = progress_dict['count']
        if current_count > last_count:
            pbar.update(current_count - last_count)
    
    pool.close()
    pool.join()
    
    print(f"\nConversion complete! Processed {progress_dict['count']} files")

