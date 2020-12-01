# Subsample the scan. For each point, find a closest point on the
# wall of the arena.
# 05_a_find_wall_pairs
# Claus Brenner, 17 NOV 2012
from lego_robot import *
from slam_b_library import filter_step, compute_cartesian_coordinates
from slam_04_a_project_landmarks import write_cylinders


# Takes one scan and subsamples the measurements, so that every sampling'th
# point is taken. Returns a list of (x, y) points in the scanner's
# coordinate system.
def get_subsampled_points(scan, sampling = 10):
    # Subsample from scan
    index_range_tuples = []
    for i in range(0, len(scan), sampling):
        index_range_tuples.append( (i, scan[i]) )
    return compute_cartesian_coordinates(index_range_tuples, 0.0)

# Given a set of points, checks for every point p if it is closer than
# eps to the left, right, upper or lower wall of the arena. If so,
# adds the point to left_list, and the closest point on the wall to
# right_list.
def get_corresponding_points_on_wall(points,
                                     arena_left = 0.0, arena_right = 2000.0,
                                     arena_bottom = 0.0, arena_top = 2000.0,
                                     eps = 150.0):
    left_list = []
    right_list = []
    
    for point in points:
        new_point = [point[0], point[1]]
        
        if abs(point[0]) < eps:
            new_point[0] = arena_left
            left_list.append(point)
            right_list.append((new_point[0], new_point[1]))
        elif abs(point[0] - arena_right) < eps:
            new_point[0] = arena_right
            left_list.append(point)
            right_list.append((new_point[0], new_point[1]))
        elif abs(point[1]) < eps:
            new_point[1] = arena_bottom
            left_list.append(point)
            right_list.append((new_point[0], new_point[1]))
        elif abs(point[1] - arena_right) < eps:
            new_point[1] = arena_top
            left_list.append(point)
            right_list.append((new_point[0], new_point[1]))
        

    # ---> Implement your code here.

    return left_list, right_list


if __name__ == '__main__':
    # The constants we used for the filter_step.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 150.0

    # The start pose we obtained miraculously.
    pose = (1850.0, 1897.0, 3.717551306747922)

    # Read the logfile which contains all scans.
    logfile = LegoLogfile()
    logfile.read("robot4_motors.txt")
    logfile.read("robot4_scan.txt")

    # Iterate over all positions.
    out_file = open("find_wall_pairs.txt", "w")
    for i in range(len(logfile.scan_data)):
        # Compute the new pose.
        pose = filter_step(pose, logfile.motor_ticks[i],
                           ticks_to_mm, robot_width,
                           scanner_displacement)

        # Subsample points.
        subsampled_points = get_subsampled_points(logfile.scan_data[i])
        world_points = [LegoLogfile.scanner_to_world(pose, c)
                        for c in subsampled_points]

        # Get corresponding points on wall.
        left, right = get_corresponding_points_on_wall(world_points)

        # Write to file.
        # The pose.
        print("F %f %f %f" % pose, file=out_file)

        # Write the scanner points and corresponding points.
        write_cylinders(out_file, "W C", left + right)

    out_file.close()