def SEQ(client_ids, starting_port=8081, host='localhost'):
    port = starting_port
    iports = {}
    for i in client_ids:
        iports[i] = ('localhost', port)
        port += 1
    return iports
