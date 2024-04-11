import numpy as np
import cv2 as cv



# The given video and calibration data
video_file = '/Users/user/Downloads/IMG_0531.mp4'
K = np.array([[935.64197163, 0, 963.02886755],
              [0, 924.48881325, 513.33345825],
              [0, 0, 1]])
dist_coeff = np.array([-0.01728153,  0.11668051, -0.0026063,   0.00025818, -0.12958552])
board_pattern = (8, 6)
board_cellsize = 0.029
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D box for simple AR
box_lower = board_cellsize * np.array([[1, 1,  0], [2, 1,  0], [2, 2.5, 0], [3, 2.5, 0], [3, 1, 0], [4, 1, 0], [4, 5, 0], [3, 5, 0], [3, 3.5, 0], [2, 3.5, 0], [2, 5,  0], [1, 5,  0]])
box_upper = board_cellsize * np.array([[1, 1,  -1], [2, 1,  -1], [2, 2.5, -1], [3, 2.5, -1], [3, 1, -1], [4, 1, -1], [4, 5, -1], [3, 5, -1], [3, 3.5, -1], [2, 3.5, -1], [2, 5,  -1], [1, 5,  -1]])

box_lower2 = board_cellsize * np.array([[5, 1, 0], [6, 1, 0], [6, 2, 0], [5, 2, 0]])
box_upper2 = board_cellsize * np.array([[5, 1, -1], [6, 1, -1], [6, 2, -1], [5, 2, -1]])

box_lower3 = board_cellsize * np.array([[5, 2.5, 0], [6, 2.5, 0], [6, 5, 0], [5, 5, 0]])
box_upper3 = board_cellsize * np.array([[5, 2.5, -1], [6, 2.5, -1], [6, 5, -1], [5, 5, -1]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
        for b, t in zip(line_lower, line_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        line_lower2, _ = cv.projectPoints(box_lower2, rvec, tvec, K, dist_coeff)
        line_upper2, _ = cv.projectPoints(box_upper2, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower2)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper2)], True, (0, 0, 255), 2)
        for b, t in zip(line_lower2, line_upper2):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        line_lower3, _ = cv.projectPoints(box_lower3, rvec, tvec, K, dist_coeff)
        line_upper3, _ = cv.projectPoints(box_upper3, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower3)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper3)], True, (0, 0, 255), 2)
        for b, t in zip(line_lower3, line_upper3):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()