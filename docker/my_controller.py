from tdw.controller import Controller

port = 1071  # This is the default port. You can change this.
c = Controller(launch_build=False, port=port) 
print("Hello world!")
c.communicate({"$type": "terminate"})
