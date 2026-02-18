"""
Simple test script to verify the connection works
"""

import socket
import json

def test_server():
    """Test that we can connect and receive data"""
    HOST = '127.0.0.1'
    PORT = 8888

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)

    print(f"Test server listening on {HOST}:{PORT}")
    print("Connect from game with: MLAgent::start()")

    conn, addr = sock.accept()
    print(f"Connected from {addr}")

    buffer = ""
    message_count = 0

    try:
        while message_count < 10:  # Just test 10 messages
            data = conn.recv(4096).decode('utf-8')
            if not data:
                break

            buffer += data

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)

                if line.strip():
                    message_count += 1

                    # Parse observation
                    obs = json.loads(line.strip())
                    print(f"\nMessage {message_count}:")
                    print(f"  Observation dims: {len(obs)}")
                    print(f"  First 5 values: {obs[:5]}")

                    # Send back random action (all zeros for now)
                    action = "0,0,0,0,0,0\n"
                    conn.send(action.encode('utf-8'))

        print("\n✅ Test successful! Received 10 observations.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()
        sock.close()

if __name__ == '__main__':
    test_server()
