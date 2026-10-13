import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mrp0712/Downloads/ros2_ws_mecanum_estudiantes/install/mi_mecanum_control'
