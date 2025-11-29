import os
import sys
import math
import random
import subprocess
import csv
from tqdm import tqdm
from multiprocessing import Pool
import argparse

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Convert XML files to ABC format")
    parser.add_argument('input_dir', type=str, help="Folder containing XML files")
    parser.add_argument('output_dir', type=str, help="Folder to save ABC files")
    parser.add_argument('--csv', type=str, required=True,
                        help="CSV file containing metadata (title, composer, etc.)")
    return parser.parse_args()


# Load metadata CSV indexed by XML filename (basename)
def load_metadata(csv_path):
    meta = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract XML filename only (basename of the path in the CSV)
            xml_path = row.get('mxl') or row.get('xml') or row.get('file')
            if not xml_path:
                continue

            xml_basename = os.path.basename(xml_path)

            meta[xml_basename] = {
                'title': row.get('title', '').strip(),
                'subtitle': row.get('subtitle', '').strip(),
                'song_name': row.get('song_name', '').strip(),
                'artist_name': row.get('artist_name', '').strip(),
                'composer_name': row.get('composer_name', '').strip(),
                'publisher': row.get('publisher', '').strip(),
            }

    return meta


def make_abc_header(meta):
    """Return ABC header text for one song."""
    header = []

    # Required: Title field
    if meta.get('title'):
        header.append(f"T:{meta['title']}")
    elif meta.get('song_name'):
        header.append(f"T:{meta['song_name']}")

    # Optional subtitle
    if meta.get('subtitle'):
        header.append(f"T:{meta['subtitle']}")

    # Composer field
    if meta.get('composer_name'):
        header.append(f"C:{meta['composer_name']}")
    elif meta.get('artist_name'):
        header.append(f"C:{meta['artist_name']}")

    # Publisher as ABC comment
    if meta.get('publisher'):
        header.append(f"%Publisher: {meta['publisher']}")

    return "\n".join(header) + "\n\n"


def convert_xml2abc(file_list, input_dir, output_root_dir, metadata):
    cmd = sys.executable + " utils/xml2abc.py -d 8 -x "
    for file in tqdm(file_list):
        relative_path = os.path.relpath(os.path.dirname(file), input_dir)
        output_dir = os.path.join(output_root_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        try:
            p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            output = result[0].decode('utf-8')

            if not output:
                with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(file + '\n')
                continue

            # MATCH: use the XML file's basename to look up metadata
            xml_basename = os.path.basename(file)
            abc_header = ""

            if xml_basename in metadata:
                abc_header = make_abc_header(metadata[xml_basename])

            final_output = abc_header + output

            output_file_path = os.path.join(output_dir, os.path.splitext(xml_basename)[0] + '.abc')
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(final_output)

        except Exception as e:
            with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                f.write(file + ' ' + str(e) + '\n')
            pass


if __name__ == '__main__':
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_root_dir = os.path.abspath(args.output_dir)

    os.makedirs("logs", exist_ok=True)

    # Load CSV metadata
    metadata = load_metadata(args.csv)

    # Build full list of XML files
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".mxl", ".xml", ".musicxml")):
                file_list.append(os.path.join(root, file).replace("\\", "/"))

    # Split input among CPUs
    random.shuffle(file_list)
    cpu = os.cpu_count()
    chunks = [
        file_list[int(i * len(file_list) / cpu):int((i + 1) * len(file_list) / cpu)]
        for i in range(cpu)
    ]

    # Parallel execution
    pool = Pool(processes=cpu)
    pool.starmap(convert_xml2abc, [(chunk, input_dir, output_root_dir, metadata) for chunk in chunks])
    pool.close()
    pool.join()

