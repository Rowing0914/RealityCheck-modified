# realitycheck_cleaned.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D


def parse_config(config_path):
    with open(config_path + ".txt") as config_file:
        lines = [line.strip().split(":")[1].strip() for line in config_file.readlines()]

    cam_vals = list(map(float, lines[0].split(',')))
    camera = np.array(cam_vals, dtype=np.float32).reshape(3, 3)

    real_dict = cv2.aruco.getPredefinedDictionary(int(lines[1]))
    real_rows, real_cols = map(int, lines[2].split(','))
    real_size = float(lines[3])
    real_space = float(lines[4])
    real_board = cv2.aruco.GridBoard_create(real_rows, real_cols, real_size, real_space, real_dict)

    virtual_dict = cv2.aruco.getPredefinedDictionary(int(lines[5]))
    virtual_rows, virtual_cols = map(int, lines[6].split(','))
    virtual_size = float(lines[7])
    virtual_space = float(lines[8])
    virtual_board = cv2.aruco.GridBoard_create(virtual_rows, virtual_cols, virtual_size, virtual_space, virtual_dict)

    return camera, real_dict, virtual_dict, real_board, virtual_board


def get_board_transforms(image_path, camera, real_dict, virtual_dict, real_board, virtual_board, rotate=False):
    img = cv2.imread(image_path)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    dist_coeffs = np.zeros(5)

    def detect_pose(dictionary, board):
        corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
        corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(img, board, corners, ids, None, camera, dist_coeffs)
        ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera, dist_coeffs, np.zeros(3), np.zeros(3))
        return ret, rvec, tvec

    v_val, v_rvec, v_tvec = detect_pose(virtual_dict, virtual_board)
    r_val, r_rvec, r_tvec = detect_pose(real_dict, real_board)

    return r_val, r_rvec, r_tvec, v_val, v_rvec, v_tvec


def write_results(output_path, results):
    with open(output_path, 'w') as f:
        for i in range(len(results['distances'])):
            row = [
                results['distances'][i],
                results['v_tvecs'][i].flatten().tolist(),
                results['r_tvecs'][i].flatten().tolist(),
                results['v_rvecs'][i].flatten().tolist(),
                results['r_rvecs'][i].flatten().tolist(),
                results['v_vals'][i],
                results['r_vals'][i]
            ]
            f.write(', '.join(map(str, row)) + '\n')


def graph_trace(v_tvecs, r_tvecs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.plot(*np.array(v_tvecs).squeeze().T, color='blue', label='Virtual')
    ax.plot(*np.array(r_tvecs).squeeze().T, color='red', label='Real')
    plt.legend()
    plt.savefig('trace_plot.png')
    plt.clf()

def graph_distances(distances):
    plt.scatter(range(len(distances)), distances, label='RealityCheck', color='blue')
    plt.xlabel("Frame")
    plt.ylabel("Distance (m)")
    plt.title("Distance Between Boards")
    plt.legend()
    plt.savefig('distance_plot.png')
    plt.clf()


def reality_check(image_dir, config_file, rotate=False):
    camera, real_dict, virtual_dict, real_board, virtual_board = parse_config(config_file)
    start = time.time()

    results = {
        'distances': [], 'v_tvecs': [], 'r_tvecs': [],
        'v_rvecs': [], 'r_rvecs': [], 'v_vals': [], 'r_vals': []
    }

    file_list = sorted(os.listdir(image_dir))
    for fname in tqdm(file_list, desc="Processing frames"):
        image_path = os.path.join(image_dir, fname)

        r_val, r_rvec, r_tvec, v_val, v_rvec, v_tvec = get_board_transforms(
            image_path, camera, real_dict, virtual_dict, real_board, virtual_board, rotate
        )
        dist = np.linalg.norm(v_tvec - r_tvec)
        results['distances'].append(dist)
        results['v_tvecs'].append(v_tvec)
        results['r_tvecs'].append(r_tvec)
        results['v_rvecs'].append(v_rvec)
        results['r_rvecs'].append(r_rvec)
        results['v_vals'].append(v_val)
        results['r_vals'].append(r_val)
    print(f"--- {time.time() - start:.2f} seconds to execute ---")

    graph_trace(results['v_tvecs'], results['r_tvecs'])
    graph_distances(results['distances'])

    output_path = config_file + "_Output.txt"
    write_results(output_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate spatial drift between real and virtual marker boards.")
    parser.add_argument("image_dir", help="Directory containing image frames")
    parser.add_argument("config_file", help="Path to configuration file (without .txt extension)")
    parser.add_argument("--rotate", action="store_true", help="Rotate images 90 degrees counterclockwise")
    args = parser.parse_args()

    import pudb; pudb.start()
    reality_check(args.image_dir, args.config_file, args.rotate)
    