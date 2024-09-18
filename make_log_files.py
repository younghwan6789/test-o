import random
from datetime import datetime, timedelta


def generate_random_timestamp():
    """랜덤한 타임스탬프 생성"""
    start_date = datetime.now() - timedelta(days=30)
    random_date = start_date + timedelta(seconds=random.randint(0, 30 * 24 * 60 * 60))
    return random_date.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]  # YYYY-mm-dd hh:MM:ss,SSS


def generate_log_message():
    """임의의 로그 메시지 생성"""
    log_levels = ['INFO', 'DEBUG', 'ERROR', 'WARN']
    level = random.choice(log_levels)
    message = f"Sample log message with {level} level."
    return f"{generate_random_timestamp()} - {level} - {message}"


def generate_log_file(file_path, target_size_mb):
    """로그 파일 생성 (MB 단위)"""
    target_size = target_size_mb * 1024 * 1024  # MB -> Bytes
    current_size = 0
    with open(file_path, 'w', encoding='utf-8') as log_file:
        while current_size < target_size:
            log_message = generate_log_message() + "\n"
            log_file.write(log_message)
            current_size += len(log_message.encode('utf-8'))
            if current_size % (10 * 1024 * 1024) < len(log_message.encode('utf-8')):
                print(f"Generated {current_size // (1024 * 1024)} MB of {target_size_mb} MB")

    print(f"Log file {file_path} generated with size {current_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    output_file = "test_log_file.log"  # 생성할 로그 파일 이름
    target_size_mb = 500  # 생성할 로그 파일 크기 (MB)
    generate_log_file(output_file, target_size_mb)
