import socket
host = "127.0.0.1"
port = 65432 # 일반적으로 가장 많이 쓰는 port 번호


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_server: #IPv4 // Ssocket.SOCKSTREAM tcp/ip
     socket_server.bind((host, port)) # host, port 지정 후, 클라이언트로부터 연결을 대기
     
     socket_server.listen() # listen으로 클라이언트를 받아들일 준비를 한다.
     print("서버가 연결이 되었습니다.")
     
     conn, addr = socket_server.accept() # 서버에서 값을 받아오자
     
     with conn:
        print(f"연결된 소켓은 {addr}")
        
        while True:
            
            data = conn.recv(1024) # 데이터를 불러오는데 1024만 불러오겠다.
            
            # 만약 데이터가 없다면 종료시키자.
            if not data: 
                break
            
            # 받은 데이터를 그대로 보내주자. -> 에코서버
            conn.sendall(data)