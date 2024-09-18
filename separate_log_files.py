import os
import re
import sys

from tqdm import tqdm


def extract_timestamp(line):
    """라인에서 타임스탬프를 추출"""
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    return match.group(1) if match else None


def split_log_file(input_file, output_dir, max_size=100 * 1024 * 1024):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_size = os.path.getsize(input_file)  # 파일 전체 크기
    current_size = 0  # 현재까지 처리한 크기
    file_count = 0
    output_file = None
    current_timestamp = None
    output_filename = None

    with open(input_file, 'r', encoding='utf-8') as infile:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing") as pbar:
            for line in infile:
                if current_size == 0:  # 새로운 파일 시작
                    current_timestamp = extract_timestamp(line)
                    if current_timestamp is None:
                        # 파일의 첫 번째 타임스탬프 찾기
                        for _ in infile:
                            current_timestamp = extract_timestamp(line)
                            if current_timestamp:
                                break
                    if current_timestamp:
                        file_count += 1
                        output_filename = os.path.join(
                            output_dir,
                            f"{os.path.basename(input_file).replace('.log', '')}_{current_timestamp.replace(' ', '_').replace(':', '-')}.log"
                        )
                        output_file = open(output_filename, 'w', encoding='utf-8')

                if current_size + len(line.encode('utf-8')) > max_size:
                    output_file.close()
                    current_size = 0
                    current_timestamp = extract_timestamp(line)
                    if current_timestamp is None:
                        for _ in infile:
                            current_timestamp = extract_timestamp(line)
                            if current_timestamp:
                                break
                    if current_timestamp:
                        file_count += 1
                        output_filename = os.path.join(
                            output_dir,
                            f"{os.path.basename(input_file).replace('.log', '')}_{current_timestamp.replace(' ', '_').replace(':', '-')}.txt"
                        )
                        output_file = open(output_filename, 'w', encoding='utf-8')

                output_file.write(line)
                line_size = len(line.encode('utf-8'))
                current_size += line_size
                pbar.update(line_size)  # 진행 상황 업데이트

    if output_file:
        output_file.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_log_file.py <log_file_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(input_file), "split_logs")
    split_log_file(input_file, output_dir)
