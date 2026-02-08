def calibrate_camera():
    # Define checkerboard dimensions
    checkerboard_size = (11, 11)  # Number of internal corners
    square_size = 1.4
    # Arrays to store points
    objpoints = []
    imgpoints = []
   
    # 3D points in real space
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    print(objp)
    images = glob.glob('image/damier*.jpg')
   
    # Counters for statistics
    total_images = len(images)
    detected_images = 0
    # Go through all images
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
       
        # Try to detect the chessboard
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
       
        # Display the image with detected corners (or not)
        output_img = img.copy()
       
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Add points
            objpoints.append(objp)
            imgpoints.append(corners2)
            output_img = img.copy()
            # Draw corners
            cv2.drawChessboardCorners(output_img, checkerboard_size, corners2, ret)
            detected_images += 1
            status = "DETECTED"
        else:
            status = "NOT DETECTED"
       
        # Display results
        print(f"Image {idx+1}/{total_images} ({fname}): {status}")
       
        # Resize for display if needed
        display_img = cv2.resize(output_img, (800, 600))
       
        # Show image
        cv2.imshow('Detection Test', display_img)
        key = cv2.waitKey(0)
       
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx)
    #np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # Display statistics
    print(f"\nSummary: {detected_images}/{total_images} images with chessboard detected")
    print(f"Success rate: {detected_images/total_images*100:.1f}%")
    # 1. Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
   
    total_error = mean_error/len(objpoints)
    print(f"\n----- CALIBRATION ANALYSIS -----")
    print(f"Average reprojection error: {total_error} pixels")
   
    # Error interpretation
     if total_error < 0.5:
        print("Calibration quality: EXCELLENT (< 0.5 pixels error)")
    elif total_error < 1.0:
        print("Calibration quality: GOOD (< 1.0 pixel error)")
    elif total_error < 1.5:
        print("Calibration quality: ACCEPTABLE (< 1.5 pixels error)")
    else:
        print("Calibration quality: POOR (> 1.5 pixels error)")
   
    # 2. Distortion parameters analysis
    print("\nDistortion parameters:", dist.ravel())
    print("Radial distortion (k1, k2, k3):", dist.ravel()[0:3])
    print("Tangential distortion (p1, p2):", dist.ravel()[3:5])
   
    # Check if distortion is within normal ranges
    k1, k2 = dist.ravel()[0:2]
    if abs(k1) > 0.5 or abs(k2) > 0.5:
        print("\nWARNING: Very high radial distortion. Possible calibration problem.")