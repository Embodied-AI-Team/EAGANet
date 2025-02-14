import socket

"""
    使用Dashboard服务端口
    dashboard基本语法:
        Load: Load a robot program.
        Play: Start the execution of a robot program.
        Pause: Pause the execution of a robot program.
        Stop: Stop the execution of a robot program.
        Set User Access Level: Change the user access level.
        Receive Feedback: Receive feedback about the robot state.
"""

class RG2:
    def __init__(self, host_name="192.168.1.102", port_num="29999"):
        # init variables
        self.host_name = host_name
        self.port_num = port_num
        
        print("gripper init")

    def gripper_control(self, command):
        """
            控制夹爪
            command: 提前在robot中设置好的urp文件
        """
        self.ClientSocket = socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ClientSocket.connect(self.host_name, self.port_num)
        
        # receive port29999 state return value
        self.upperMessage = self.ClientSocket.recv(1024).decode() 
        print("The message from Server:" + self.upperMessage)
        
        # load the robot program
        message = "load %s\r\n" % command
        self.ClientSocket.send(message.encode())
        print("The message from Server:"+upperMessage)
        
        # start the execution
        message ="play\r\n"
        self.ClientSocket.send(message.encode())
        upperMessage = self.ClientSocket.recv(1024).decode() 
        print("The message from Server:"+upperMessage)

        self.ClientSocket.close()

        

