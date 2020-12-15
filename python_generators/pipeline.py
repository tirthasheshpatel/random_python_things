from tail import follow

def grep(pattern, lines):
    for line in lines:
        if pattern in line:
            yield line

if __name__ == '__main__':
    logs = open('logs.txt', 'r')
    try:
        for line in grep('python', follow(logs)):
            print(line, end='')
    except KeyboardInterrupt:
        print("exiting...")
    logs.close()
