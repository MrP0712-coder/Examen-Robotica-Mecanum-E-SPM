#!/usr/bin/env python3
"""
cinematica.py — Cinemática del robot Mecanum (Parte 4)

    ξ = [u, v, r]ᵀ                       velocidades en el marco del robot
    ω = [ω_di, ω_dd, ω_ti, ω_td]ᵀ        ruedas: delantera izq/der, trasera izq/der

    Directa : ξ = W ω     (W de 3x4, ver enunciado)
    Inversa : ω = W⁺ ξ

Lazo:
    /cmd_vel --W⁺--> /vel_rueda_* --> Gazebo (JointController) --> ruedas
    /joint_states --W--> ξ --η̇ = J(φ)ξ, Euler--> /odom + TF odom -> base_link

    ros2 run mi_mecanum_control cinematica --ros-args -p use_sim_time:=true
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

# ===== Parámetros físicos de la Variante E (coinciden con el xacro) =====
A_RUEDA = 0.070    # a  : rueda_radio (m)
LX = 0.21          # lx : distancia_ejes / 2 = 0.42 / 2 (m)
LY = 0.19          # ly : separacion_ruedas / 2 = 0.38 / 2 (m)

RUEDAS = ['del_izq', 'del_der', 'tra_izq', 'tra_der']
JOINTS = [f'joint_rueda_{p}' for p in RUEDAS]
TOPICOS = [f'/vel_rueda_{p}' for p in RUEDAS]


def matriz_W(a, lx, ly):
    """Cinemática directa del Mecanum: ξ = W ω  (3x4)."""
    l_sum = lx + ly
    return (a / 4.0) * np.array([
        [1.0,  1.0,  1.0,  1.0],
        [-1.0, 1.0,  1.0, -1.0],
        [-1.0 / l_sum, 1.0 / l_sum, -1.0 / l_sum, 1.0 / l_sum]
    ])


def envolver(a):
    """Envuelve un ángulo a [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def integrar(x, y, phi, u, v, r, dt):
    """η̇ = J(φ) ξ e integración de Euler. Devuelve (x, y, φ) con φ en [-π, π]."""
    x_nuevo = x + (u * math.cos(phi) - v * math.sin(phi)) * dt
    y_nuevo = y + (u * math.sin(phi) + v * math.cos(phi)) * dt
    phi_nuevo = envolver(phi + r * dt)
    return x_nuevo, y_nuevo, phi_nuevo


class Cinematica(Node):
    def __init__(self):
        super().__init__('cinematica')
        self.W = matriz_W(A_RUEDA, LX, LY)
        self.W_pinv = np.linalg.pinv(self.W)

        # Inversa
        self.create_subscription(Twist, '/cmd_vel', self.cb_cmd_vel, 10)
        self.pubs = [self.create_publisher(Float64, t, 10) for t in TOPICOS]

        # Directa + odometría
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.tf = TransformBroadcaster(self)

        self.x = self.y = self.phi = 0.0
        self.t_ant = None

        np.set_printoptions(precision=4, suppress=True)
        self.get_logger().info(f'\nW =\n{self.W}\nW⁺ =\n{self.W_pinv}')

    # ---------- Cinemática inversa: ω = W⁺ ξ ----------
    def cb_cmd_vel(self, msg):
        xi = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        omega = self.W_pinv @ xi
        for pub, w in zip(self.pubs, omega):
            pub.publish(Float64(data=float(w)))

    # ---------- Cinemática directa: ξ = W ω, y odometría ----------
    def cb_joints(self, msg):
        # ω en el orden de JOINTS
        try:
            omega = np.array([msg.velocity[msg.name.index(j)] for j in JOINTS])
        except (ValueError, IndexError):
            return

        ahora = self.get_clock().now()
        if self.t_ant is None:
            self.t_ant = ahora
            return
        dt = (ahora - self.t_ant).nanoseconds * 1e-9
        self.t_ant = ahora
        if dt <= 0.0:
            return

        xi = self.W @ omega
        u, v, r = float(xi[0]), float(xi[1]), float(xi[2])
        self.x, self.y, self.phi = integrar(self.x, self.y, self.phi, u, v, r, dt)

        qz, qw = math.sin(self.phi / 2.0), math.cos(self.phi / 2.0)
        stamp = ahora.to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = u
        odom.twist.twist.linear.y = v
        odom.twist.twist.angular.z = r

        self.pub_odom.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    nodo = Cinematica()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()