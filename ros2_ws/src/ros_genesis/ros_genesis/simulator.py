import genesis as gs
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import JointState

class GenesisPublisher(Node):

    def __init__(self, fps=30):
        super().__init__('joint_state_publisher')

        print('Hi from ros-genesis.')
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(show_viewer=True)
        plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )
        self.scene.build()

        self.joint_names = []
        self.dofs_idx = []
        for x in self.robot.joints:
            name = x.name
            dof_idx_local = x.dof_idx_local
            if dof_idx_local is not None:
                self.joint_names.append(name)
                self.dofs_idx.append(dof_idx_local)
        print(self.joint_names)
        print(self.dofs_idx)

        self.joint_publishers = [self.create_publisher(JointState, f'genesis/{name}', 10) for name in self.joint_names]
        self.timer_period = 1.0 / fps  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        self.scene.step()
        self.i += self.timer_period
        joint_pos = self.robot.get_dofs_position(self.dofs_idx).tolist()
        joint_vel = self.robot.get_dofs_velocity(self.dofs_idx).tolist()
        joint_f = self.robot.get_dofs_force(self.dofs_idx).tolist()
        for name, pos, vel, f, pub in zip(self.joint_names, joint_pos, joint_vel, joint_f, self.joint_publishers):
            msg = JointState()
            msg.name = [name]
            msg.position = [pos]
            msg.velocity = [vel]
            msg.effort = [f]
            pub.publish(msg)
        self.get_logger().info(f'{name} publishing: {msg}')


def main(args=None):
    rclpy.init(args=args)

    genesis_publisher = GenesisPublisher()

    rclpy.spin(genesis_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    genesis_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
