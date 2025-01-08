import paramiko

def ssh_connect(hostname, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)
    return client

if __name__ == '__main__':
    hostname = '192.168.175.164'
    username = 'nao'
    password = 'nao'
    client = ssh_connect(hostname, username, password)
    print(client)
    stdin, stdout, stderr = client.exec_command('python pycall_NaoGrasp.py')
    print(stdout.read().decode())
    client.close()
