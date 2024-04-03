import math

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class DroneState:
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

def get_homogenous_transform(droneState):
    # Conversion of Euler angles to rotation matrix
    cos_roll = np.cos(droneState.roll)
    sin_roll = np.sin(droneState.roll)
    cos_pitch = np.cos(droneState.pitch)
    sin_pitch = np.sin(droneState.pitch)
    cos_yaw = np.cos(droneState.yaw)
    sin_yaw = np.sin(droneState.yaw)

    rotation_matrix = np.array([[cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                                 cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll, droneState.x],
                                [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                                 sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll, droneState.y],
                                [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll, droneState.z],
                                [0, 0, 0, 1]])

    return rotation_matrix

def point_drone_to_opti(droneState, drone_point):
    return get_homogenous_transform(droneState) @ np.array([*drone_point, 1])

def vector_drone_to_opti(droneState, drone_point):
    return get_homogenous_transform(droneState) @ np.array([*drone_point, 0])


def plot(droneState, ground_height, points, image_fov, point_line_length=6.2, camera_line_length=1.3):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111,projection='3d')

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)

    x, y = np.meshgrid(x, y)
    z = np.full_like(x, ground_height)

    ax.plot_surface(x, y, z, alpha=0.2, label='Ground Plane')

    #draw the drone reference frame
    ax.scatter(droneState.x, droneState.y, droneState.z, color="red")
    drone_x_drone = np.array([1, 0, 0])
    drone_x_opti = point_drone_to_opti(droneState, drone_x_drone)
    drone_y_drone = np.array([0, 1, 0])
    drone_y_opti = point_drone_to_opti(droneState, drone_y_drone)
    drone_z_drone = np.array([0, 0, 1])
    drone_z_opti = point_drone_to_opti(droneState, drone_z_drone)

    ax.plot([droneState.x, drone_x_opti[0]], [droneState.y, drone_x_opti[1]], [droneState.z, drone_x_opti[2]], color='red')
    ax.plot([droneState.x, drone_y_opti[0]], [droneState.y, drone_y_opti[1]], [droneState.z, drone_y_opti[2]], color='green')
    ax.plot([droneState.x, drone_z_opti[0]], [droneState.y, drone_z_opti[1]], [droneState.z, drone_z_opti[2]], color='blue')
    ax.scatter(droneState.x, droneState.y, droneState.z, label="Body", color="red")

    #draw the opti reference frame
    opti_x_opti = np.array([1, 0, 0])
    opti_y_opti = np.array([0, 1, 0])
    opti_z_opti = np.array([0, 0, 1])

    ax.plot([0, opti_x_opti[0]], [0, opti_x_opti[1]], [0, opti_x_opti[2]], color='red')
    ax.plot([0, opti_y_opti[0]], [0, opti_y_opti[1]], [0, opti_y_opti[2]], color='green')
    ax.plot([0, opti_z_opti[0]], [0, opti_z_opti[1]], [0, opti_z_opti[2]], color='blue')
    ax.scatter(0, 0, 0, label="NED", color="blue")

    #draw camera fov
    camera_corners = [[-1,-1], [-1, 1], [1,1], [1,-1],[-1,-1]]
    prev_endpoint = None
    for idx, camera_corner in enumerate(camera_corners):
        longitude = camera_corner[0] * image_fov[0]/2
        latitude = camera_corner[1] * image_fov[1]/2

        sin_lat = math.sin(-latitude)
        cos_lat = math.cos(-latitude)
        sin_lon = math.sin(longitude)
        cos_lon = math.cos(longitude)


        # Direction vector of the line
        direction_vector_drone = np.array([cos_lat * cos_lon,
                                     cos_lat * sin_lon,
                                     -cos_lon * sin_lat])

        direction_vector_opti = vector_drone_to_opti(droneState, direction_vector_drone)[:3]


        line_opti = direction_vector_opti * camera_line_length


        ax.plot([droneState.x, droneState.x + line_opti[0]], [droneState.y, droneState.y + line_opti[1]],
                [droneState.z, droneState.z + line_opti[2]],
                color='black')
        if prev_endpoint is not None:
            ax.plot([droneState.x + prev_endpoint[0], droneState.x + line_opti[0]],
                    [droneState.y + prev_endpoint[1], droneState.y + line_opti[1]],
                    [droneState.z + prev_endpoint[2], droneState.z + line_opti[2]], color='black')

        prev_endpoint = line_opti

        print(idx)
        if idx == 2:
            # draw points
            for point in points:
                longitude = point[0] * image_fov[0] / 2
                latitude = point[1] * image_fov[1] / 2

                sin_lat = math.sin(-latitude)
                cos_lat = math.cos(-latitude)
                sin_lon = math.sin(longitude)
                cos_lon = math.cos(longitude)

                # Direction vector of the line
                direction_vector_drone = np.array([cos_lat * cos_lon,
                                                   cos_lat * sin_lon,
                                                   -cos_lon * sin_lat])

                direction_vector_opti = vector_drone_to_opti(droneState, direction_vector_drone)[:3]

                line_opti = direction_vector_opti * point_line_length

                # Plane equation coefficients (for a plane parallel to xy-plane)
                a, b, c, d = 0, 0, 1, -ground_height

                point = np.array([droneState.x, droneState.y, droneState.z])

                # Intersection parameter
                t = (-d - np.dot(point, [a, b, c])) / np.dot(direction_vector_opti, [a, b, c])

                ax.plot([droneState.x, droneState.x + line_opti[0]], [droneState.y, droneState.y + line_opti[1]],
                        [droneState.z, droneState.z + line_opti[2]],
                        color='pink')

                # prevent points behind the drone
                if t >= 0:
                    # Intersection point
                    intersection_point = point + t * direction_vector_opti

                    ax.scatter(*intersection_point, color="black")




    fs = 14
    #label axes
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)

    ax.tick_params(labelsize=9)

    #set azimuth
    ax.view_init(elev=15, azim=-122, roll=0)


    ax.set_zticks([0, -1, -2])

    #set axis ranges
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_zlim(-2, .1)

    ax.invert_zaxis()
    ax.invert_yaxis()

    ax.legend(fontsize=fs, bbox_to_anchor=(.6, .8), loc='upper left')

    #equal axis scale
    ax.set_aspect('equal')

    plt.show()





droneState = DroneState(-2, -2, -1, roll=0, pitch=-.12, yaw=.7)

ground_height = 0 # m

#horizontal, vertical
image_fov = [np.deg2rad(70), np.deg2rad(40)]  #
#latitude, longitude in fraction from center  -1, -1  is top left; 1,1 is bottom right, 0, 0 is center
points = [[.3, .2], [-.9, .4]]

plot(droneState, ground_height, points, image_fov)



