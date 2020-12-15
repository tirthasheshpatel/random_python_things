import time

def follow(file):
    file.seek(0, 2)
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

if __name__ == '__main__':
    logs = open('logs.txt', 'r')
    try:
        for line in follow(logs):
            print(line, end='')
    except KeyboardInterrupt:
        print("exiting...")
    logs.close()
