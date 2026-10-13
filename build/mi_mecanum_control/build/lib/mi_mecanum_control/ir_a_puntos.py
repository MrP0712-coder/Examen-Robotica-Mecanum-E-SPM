#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

PUNTOS = [
    (2.5, 0.0, math.pi / 2.0),
    (2.5, 1.0, math.pi / 2.0),
    (0.0, 1.0, -math.pi / 2.0),
    (0.0, 0.0, 0.0)
]


def envolver(a):
    return math.atan2(math.sin(a), math.cos(a))


class IrAPuntos(Node):
    def __init__(self):
        super().__init__('ir_a_puntos')

        self.Lambda = np.diag([0.8, 0.8, 1.2])
        self.idx_punto = 0

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.cb_odom, 10)

        self.get_logger().info(f'Iniciando secuencia de puntos. Punto 1: {PUNTOS[0]}')

    def cb_odom(self, msg):
        if self.idx_punto >= len(PUNTOS):
            self.pub_cmd_vel.publish(Twist())
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        phi = 2.0 * math.atan2(qz, qw)

        xd, yd, phid = PUNTOS[self.idx_punto]

        ex = xd - x
        ey = yd - y
        ephi = envolver(phid - phi)

        distancia = math.hypot(ex, ey)

        # Matriz de transformación del marco global al del robot
        u_m_world = self.Lambda[0, 0] * ex
        v_m_world = self.Lambda[1, 1] * ey

        # Convertir a velocidades en el marco del robot
        u = u_m_world * math.cos(phi) + v_m_world * math.sin(phi)
        v = -u_m_world * math.sin(phi) + v_m_world * math.cos(phi)
        r = self.Lambda[2, 2] * ephi

        # Separar en fases: Traslación sin rotar -> Rotación en el sitio
        if distancia >= 0.05:
            # Fase 1: Desplazamiento puro (sin rotar)
            r = 0.0
            norma_uv = math.hypot(u, v)
            if norma_uv > 0.4:
                u = (u / norma_uv) * 0.4
                v = (v / norma_uv) * 0.4
        elif abs(ephi) >= 0.05:
            # Fase 2: Llegó a (x, y), ahora rota sobre su sitio
            u = 0.0
            v = 0.0
            r = max(min(r, 1.0), -1.0)
        else:
            # Punto completamente alcanzado (posición y orientación)
            self.get_logger().info(f'Punto {PUNTOS[self.idx_punto]} alcanzado')
            self.idx_punto += 1
            if self.idx_punto < len(PUNTOS):
                self.get_logger().info(f'Siguiente punto: {PUNTOS[self.idx_punto]}')
            else:
                self.get_logger().info('Secuencia de puntos completada.')
            self.pub_cmd_vel.publish(Twist())
            return

        cmd = Twist()
        cmd.linear.x = float(u)
        cmd.linear.y = float(v)
        cmd.angular.z = float(r)
        self.pub_cmd_vel.publish(cmd)


def main():
    rclpy.init()
    nodo = IrAPuntos()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()