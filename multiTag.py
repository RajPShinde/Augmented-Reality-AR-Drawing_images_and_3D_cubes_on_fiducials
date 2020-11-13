import numpy as np
# import imutils
import os, sys
import argparse
import copy
# This try-catch is a workaround for Python3 when used with ROS; 
# it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

# Import OpenCV library
import cv2 as cv

# Use Argument Parser to get the video
ap = argparse.ArgumentParser()
ap.add_argument("--vid", required=True, help="Path to video file")
ap.add_argument("--func", required=True, help="1 (for detection) / 2 (for lena superimpose) / 3 (for tracking)")

args = vars(ap.parse_args())

vid = cv.VideoCapture(args["vid"])

func = args["func"]

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
print("Do you want a video:")
val = input("Give 1 for Yes, 0 for No: ") 

if val == "1":
	out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
	print("Video recording will start accordingly.")
	

# Camera configurations from yaml file.
# given as lower triangular matrix.
KT = np.array([[1406.08415449821, 0 , 0],
               [2.20679787308599, 1417.99930662800,0],
               [1014.13643417416, 566.347754321696,1]])

# Intrinsic camera matrix is given by K
K = KT.T

def read_lena():
	img = cv.imread('Lena.png', cv.IMREAD_COLOR)
	return img

def read_marker():
	img = cv.imread('ref_marker.png', cv.IMREAD_COLOR)
	return img

def homography_estimation(src, tgt):
	A = np.array([
					[-src[0][0], -src[0][1], -1, 0, 0, 0, src[0][0] * tgt[0][0], src[0][1] * tgt[0][0], tgt[0][0]],
					[0, 0, 0, -src[0][0], -src[0][1], -1, src[0][0] * tgt[0][1], src[0][1] * tgt[0][1], tgt[0][1]],
					[-src[1][0], -src[1][1], -1, 0, 0, 0, src[1][0] * tgt[1][0], src[1][1] * tgt[1][0], tgt[1][0]],
					[0, 0, 0, -src[1][0], -src[1][1], -1, src[1][0] * tgt[1][1], src[1][1] * tgt[1][1], tgt[1][1]],
					[-src[2][0], -src[2][1], -1, 0, 0, 0, src[2][0] * tgt[2][0], src[2][1] * tgt[2][0], tgt[2][0]],
					[0, 0, 0, -src[2][0], -src[2][1], -1, src[2][0] * tgt[2][1], src[2][1] * tgt[2][1], tgt[2][1]],
					[-src[3][0], -src[3][1], -1, 0, 0, 0, src[3][0] * tgt[3][0], src[3][1] * tgt[3][0], tgt[3][0]],
					[0, 0, 0, -src[3][0], -src[3][1], -1, src[3][0] * tgt[3][1], src[3][1] * tgt[3][1], tgt[3][1]],
				])
	U,S,V = np.linalg.svd(A)
	X = V[:][8]/V[8][8]
	Hinv = np.reshape(X,(3,3))
	H = np.linalg.inv(Hinv)
	H = H/H[2][2]
	return H

def homography_estimation_cube(src, tgt):
	A = np.array([
					[-src[0][0], -src[0][1], -1, 0, 0, 0, src[0][0] * tgt[0][0], src[0][1] * tgt[0][0], tgt[0][0]],
					[0, 0, 0, -src[0][0], -src[0][1], -1, src[0][0] * tgt[0][1], src[0][1] * tgt[0][1], tgt[0][1]],
					[-src[1][0], -src[1][1], -1, 0, 0, 0, src[1][0] * tgt[1][0], src[1][1] * tgt[1][0], tgt[1][0]],
					[0, 0, 0, -src[1][0], -src[1][1], -1, src[1][0] * tgt[1][1], src[1][1] * tgt[1][1], tgt[1][1]],
					[-src[2][0], -src[2][1], -1, 0, 0, 0, src[2][0] * tgt[2][0], src[2][1] * tgt[2][0], tgt[2][0]],
					[0, 0, 0, -src[2][0], -src[2][1], -1, src[2][0] * tgt[2][1], src[2][1] * tgt[2][1], tgt[2][1]],
					[-src[3][0], -src[3][1], -1, 0, 0, 0, src[3][0] * tgt[3][0], src[3][1] * tgt[3][0], tgt[3][0]],
					[0, 0, 0, -src[3][0], -src[3][1], -1, src[3][0] * tgt[3][1], src[3][1] * tgt[3][1], tgt[3][1]],
				])
	U,S,V = np.linalg.svd(A, full_matrices=True)
	V = (copy.deepcopy(V))/(copy.deepcopy(V[8][8]))
	H = V[8,:].reshape(3, 3)
	return H

def calculate_projection_matrix(h_matrix,k_matrix):
    h1 = h_matrix[:,0]
    h2 = h_matrix[:,1]
    h3 = h_matrix[:,2]
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(k_matrix),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(k_matrix),h2)))
    btilda = lamda * np.matmul(np.linalg.inv(k_matrix),h_matrix)

    d = np.linalg.det(btilda)
    if d > 0:
        b = btilda
    else:
        b = -1 * btilda
    row1 = b[:, 0]
    row2 = b[:, 1]
    row3 = np.cross(row1, row2)
    l = b[:, 2]
    R = np.column_stack((row1, row2, row3, l))
    P_matrix = np.matmul(k_matrix,R)
    return P_matrix


def tag_orientation(tag):
	orientation = None
	if(tag[2][2] == 0 and tag[5][2] == 0 and tag[5][5] == 1 and tag[2][5] == 0):
		orientation = 0
	elif(tag[2][2] == 0 and tag[5][2] == 1 and tag[5][5] == 0 and tag[2][5] == 0): 
		orientation = 90
	elif(tag[2][2] == 1 and tag[5][2] == 0 and tag[5][5] == 0 and tag[2][5] == 0):
		orientation = 180
	elif(tag[2][2] == 0 and tag[5][2] == 0 and tag[5][5] == 0 and tag[2][5] == 1):
		orientation = -90
	# print(orientation)
	return orientation

def get_all_contours():
	ret, frame = vid.read()
	tagCnt = None
	if frame is not None and len(frame) != 0:
		gblur = cv.GaussianBlur(frame, (7,7), 0)
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		ret2, threshold = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
		contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		# contours = sorted(contours, key = cv.contourArea, reverse = True)[1:]
		tagCnt = []
		for h, cnt in zip(hierarchy[0], contours):
			perimeter = cv.arcLength(cnt, True)
			cnt = cv.approxPolyDP(cnt, 0.04*perimeter, True)
			if cv.isContourConvex(cnt) and len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.contourArea(cnt) < 10500:
				cnt = cnt.reshape(-1, 2)
				if h[0] == -1 and h[1] == -1 and h[3] != -1:
					tagCnt.append(cnt)
		return tagCnt, frame

def find_tag_id(frame):
	new_ref = []
	frame_h, frame_w, l = frame.shape
	segment_h = int(frame_h/8)
	segment_w = int(frame_w/8)
	black_count = 0
	white_count = 0
	gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	count = 0
	for itemx in range(0, frame_h, segment_h):
		for itemy in range(0, frame_w, segment_w):
			black_count = 0
			white_count = 0
			for pixx in range(0, segment_h):
				for pixy in range(0, segment_w):
					if frame[itemx + pixx][itemy + pixy].all() == np.all([255,255,255]):
						white_count += 1
					else:
						black_count += 1
			count += 1
			
			if(white_count > black_count):
				new_ref.append(1)
			else:
				new_ref.append(0)
	
	new_arr = np.reshape(new_ref, (8, 8))
	print(new_arr)
	orientation = tag_orientation(new_arr)
	
	if orientation == 0:
		tag_id = new_arr[3][3] * 0 + new_arr[4][3] * 2 + new_arr[4][4] * 4 + new_arr[3][4] * 8
	elif orientation == 90:
		tag_id = new_arr[3][4] * 0 + new_arr[3][3] * 2 + new_arr[4][3] * 4 + new_arr[4][4] * 8
	elif orientation == 180:
		tag_id = new_arr[4][4] * 0 + new_arr[3][4] * 2 + new_arr[3][3] * 4 + new_arr[4][3] * 8
	elif orientation == -90:
		tag_id = new_arr[4][3] * 0 + new_arr[4][4] * 2 + new_arr[3][4] * 4 + new_arr[3][3] * 8
	else:
		tag_id = None
		orientation = None
	# print(tag_id,orientation)

	return orientation, tag_id

def warp_inv(h_matrix, contour, source, copy):
	coordinates = np.indices((copy.shape[1], copy.shape[0]))
	coordinates = coordinates.reshape(2, -1)
	coordinates = np.vstack((coordinates, np.ones(coordinates.shape[1])))
	temp_x, temp_y = coordinates[0], coordinates[1]

	warp_coordinates = (h_matrix@coordinates)

	x1, y1 ,z = warp_coordinates[0, :]/warp_coordinates[2, :], warp_coordinates[1, :]/warp_coordinates[2, :], warp_coordinates[2, :]/warp_coordinates[2, :]
	temp_x, temp_y = temp_x.astype(int), temp_y.astype(int)
	x1, y1 = x1.astype(int), y1.astype(int)
	if x1.all() >= 0 and x1.all() < 1920 and y1.all() >= 0 and y1.all() < 1080:
		copy[temp_y, temp_x] = source[y1,x1]
	return copy


def warp(h_matrix, contour, source, copy):
	coordinates = np.indices((copy.shape[1], copy.shape[0]))
	coordinates = coordinates.reshape(2, -1)
	coordinates = np.vstack((coordinates, np.ones(coordinates.shape[1])))
	temp_x, temp_y = coordinates[0], coordinates[1]

	warp_coordinates = (h_matrix@coordinates)
	x1, y1 ,z= warp_coordinates[0, :]/warp_coordinates[2, :], warp_coordinates[1, :]/warp_coordinates[2, :], warp_coordinates[2, :]/warp_coordinates[2, :]
	temp_x, temp_y = temp_x.astype(int),temp_y.astype(int)
	x1, y1 = x1.astype(int), y1.astype(int)
	if x1.all() >= 0 and x1.all() < 1920 and y1.all() >= 0 and y1.all() < 1080:
		source[y1,x1] = copy[temp_y, temp_x]
	return source

def tag_detection_1(contours, target, frame):
	dest = target
	source = np.float32([[contours[0][0],contours[0][1]],[contours[1][0],contours[1][1]],
                [contours[2][0],contours[2][1]],[contours[3][0],contours[3][1]]])
	
	h_matrix = homography_estimation(source, dest)
	
	frame_copy = np.zeros((200, 200, 3), np.uint8)
	warped_frame_1 = warp_inv(h_matrix, contours, frame, frame_copy)
	
	_,warped_threshold_1= cv.threshold(warped_frame_1, 200, 255, cv.THRESH_BINARY)
	tag_orientation_1, tag_id_1 = find_tag_id(warped_threshold_1)
	
	print("Tag Orientation 1: ", tag_orientation_1)
	print("Tag ID 1: ", tag_id_1)
	cv.imshow("Tag 1", warped_frame_1)
	font = cv.FONT_HERSHEY_SIMPLEX
	x = contours[0][0]
	y = contours[0][1] - 100 
	str_text = "ID: " + str(tag_id_1) + " Orientation: " + str(tag_orientation_1)
	cv.putText(frame, str_text, (x, y), font, 0.8, (0, 0, 255), 2, cv.LINE_AA)
	
def tag_detection_2(contours, target, frame):
	dest = target
	
	source = np.float32([[contours[0][0],contours[0][1]],[contours[1][0],contours[1][1]],
                [contours[2][0],contours[2][1]],[contours[3][0],contours[3][1]]])
	
	h_matrix = homography_estimation(source, dest)
	
	frame_copy = np.zeros((200, 200, 3), np.uint8)
	warped_frame_2 = warp_inv(h_matrix, contours, frame, frame_copy)
	_,warped_threshold_2 = cv.threshold(warped_frame_2, 200, 255, cv.THRESH_BINARY)
	tag_orientation_2, tag_id_2 = find_tag_id(warped_threshold_2)
	
	print("Tag Orientation 2: ", tag_orientation_2)
	print("Tag ID 2: ", tag_id_2)
	cv.imshow("Tag 2", warped_frame_2)
	font = cv.FONT_HERSHEY_SIMPLEX
	x = contours[0][0]
	y = contours[0][1] - 100 
	str_text = "ID: " + str(tag_id_2) + " Orientation: " + str(tag_orientation_2)
	cv.putText(frame, str_text, (x, y), font, 0.8, (0, 0, 255), 2, cv.LINE_AA)
	
def tag_detection_3(contours, target, frame):
	dest = target
	
	source = np.float32([[contours[0][0],contours[0][1]],[contours[1][0],contours[1][1]],
                [contours[2][0],contours[2][1]],[contours[3][0],contours[3][1]]])
	
	h_matrix = homography_estimation(source, dest)
	
	frame_copy = np.zeros((200, 200, 3), np.uint8)
	warped_frame_3 = warp_inv(h_matrix, contours, frame, frame_copy)
	_,warped_threshold_3= cv.threshold(warped_frame_3, 200, 255, cv.THRESH_BINARY)
	tag_orientation_3, tag_id_3 = find_tag_id(warped_threshold_3)
	
	print("Tag Orientation 3: ", tag_orientation_3)
	print("Tag ID 3: ", tag_id_3)
	cv.imshow("Tag 3",warped_frame_3)
	font = cv.FONT_HERSHEY_SIMPLEX
	x = contours[0][0]
	y = contours[0][1] - 100 
	str_text = "ID: " + str(tag_id_3) + " Orientation: " + str(tag_orientation_3)
	cv.putText(frame, str_text, (x, y), font, 0.8, (0, 0, 255), 2, cv.LINE_AA)
	
def lena_transform_1(var, contours, target, frame):    
	po = var
	if frame is not None:
		source = np.float32([[contours[0][0],contours[0][1]],[contours[1][0],contours[1][1]], [contours[2][0],contours[2][1]],[contours[3][0],contours[3][1]]])
		reference_corners = np.float32([[0,0],[200,0],[200,200],[0,200]])

		h_matrix = homography_estimation(source, reference_corners)
		h_matrix_lena = homography_estimation(source, target)
		lena = read_lena()
		frame_copy = np.zeros((200,200,3), np.uint8)

		warped_frame_inv = warp_inv(h_matrix,contours,frame,frame_copy)
		_, warped_threshold = cv.threshold(warped_frame_inv, 180, 255, cv.THRESH_BINARY)
		
		
		orientation, tag_id = find_tag_id(warped_threshold)
		if orientation==None:
			if po==0:
				Lena=warp(h_matrix_lena,contours,frame,lena)
			elif po==90:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_90_CLOCKWISE))
			elif po==180:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_180))
			elif po==-90:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena,cv.ROTATE_90_COUNTERCLOCKWISE))
			
		if orientation==0:
			Lena=warp(h_matrix_lena,contours,frame,lena)
		elif orientation==90:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_90_CLOCKWISE))
		elif orientation==180:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_180))
		elif orientation==-90:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena,cv.ROTATE_90_COUNTERCLOCKWISE))
		if orientation!=None:
			po=orientation

	return po

def lena_transform_2(var, contours, target, frame):    
	po = var
	if frame is not None:
		source = np.float32([[contours[0][0],contours[0][1]],[contours[1][0],contours[1][1]], [contours[2][0],contours[2][1]],[contours[3][0],contours[3][1]]])
		reference_corners = np.float32([[0,0],[200,0],[200,200],[0,200]])

		h_matrix = homography_estimation(source, reference_corners)
		h_matrix_lena = homography_estimation(source, target)
		lena = read_lena()
		frame_copy = np.zeros((200,200,3), np.uint8)

		warped_frame_inv = warp_inv(h_matrix,contours,frame,frame_copy)
		_, warped_threshold = cv.threshold(warped_frame_inv, 180, 255, cv.THRESH_BINARY)
		
		
		orientation, tag_id = find_tag_id(warped_threshold)
		if orientation==None:
			if po==0:
				Lena=warp(h_matrix_lena,contours,frame,lena)
			elif po==90:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_90_CLOCKWISE))
			elif po==180:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_180))
			elif po==-90:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena,cv.ROTATE_90_COUNTERCLOCKWISE))
			
		if orientation==0:
			Lena=warp(h_matrix_lena,contours,frame,lena)
		elif orientation==90:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_90_CLOCKWISE))
		elif orientation==180:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_180))
		elif orientation==-90:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena,cv.ROTATE_90_COUNTERCLOCKWISE))
		if orientation!=None:
			po=orientation

	return po

def lena_transform_3(var, contours, target, frame):    
	po = var
	if frame is not None:
		source = np.float32([[contours[0][0],contours[0][1]],[contours[1][0],contours[1][1]], [contours[2][0],contours[2][1]],[contours[3][0],contours[3][1]]])
		reference_corners = np.float32([[0,0],[200,0],[200,200],[0,200]])

		h_matrix = homography_estimation(source, reference_corners)
		h_matrix_lena = homography_estimation(source, target)
		lena = read_lena()
		frame_copy = np.zeros((200,200,3), np.uint8)

		warped_frame_inv = warp_inv(h_matrix,contours,frame,frame_copy)
		_, warped_threshold = cv.threshold(warped_frame_inv, 180, 255, cv.THRESH_BINARY)
		
		
		orientation, tag_id = find_tag_id(warped_threshold)
		if orientation==None:
			if po==0:
				Lena=warp(h_matrix_lena,contours,frame,lena)
			elif po==90:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_90_CLOCKWISE))
			elif po==180:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_180))
			elif po==-90:
				Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena,cv.ROTATE_90_COUNTERCLOCKWISE))
			
		if orientation==0:
			Lena=warp(h_matrix_lena,contours,frame,lena)
		elif orientation==90:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_90_CLOCKWISE))
		elif orientation==180:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena, cv.ROTATE_180))
		elif orientation==-90:
			Lena=warp(h_matrix_lena,contours,frame,cv.rotate(lena,cv.ROTATE_90_COUNTERCLOCKWISE))
		if orientation!=None:
			po=orientation

	return po

def draw_cube_multi(contours, frame):
    if contours is not None:
        target = np.array([[0,0], [0,79], [79,79], [79,0]])
        print(str([[contours[0][0][0],contours[0][0][1]], [contours[0][1][0],contours[0][1][1]], [contours[0][2][0],contours[0][2][1]], [contours[0][3][0],contours[0][3][1]]]))
        if (len(contours) == 3):
        	contours_1 = np.float32([[contours[0][0][0],contours[0][0][1]], [contours[0][1][0],contours[0][1][1]], [contours[0][2][0],contours[0][2][1]], [contours[0][3][0],contours[0][3][1]]])
        	contours_2 = np.float32([[contours[1][0][0],contours[1][0][1]], [contours[1][1][0],contours[1][1][1]], [contours[1][2][0],contours[1][2][1]], [contours[1][3][0],contours[1][3][1]]])
        	contours_3 = np.float32([[contours[2][0][0],contours[2][0][1]], [contours[2][1][0],contours[2][1][1]], [contours[2][2][0],contours[2][2][1]], [contours[2][3][0],contours[2][3][1]]])
                
        	h_matrix_1 = homography_estimation_cube(target, contours_1)
          	projection_matrix_1 = calculate_projection_matrix(h_matrix_1, K)
            h_matrix_2 = homography_estimation_cube(target, contours_2)
            projection_matrix_2 = calculate_projection_matrix(h_matrix_2, K)
            h_matrix_3 = homography_estimation_cube(target, contours_3)
            projection_matrix_3 = calculate_projection_matrix(h_matrix_3, K)



        	x01,y01,z01 = np.matmul(projection_matrix_1,[0,0,0,1])
        	x02,y02,z02 = np.matmul(projection_matrix_1,[79,0,0,1])
        	x03,y03,z03 = np.matmul(projection_matrix_1,[79,79,0,1])
        	x04,y04,z04 = np.matmul(projection_matrix_1,[0,79,0,1])
        	x05,y05,z05 = np.matmul(projection_matrix_1,[0,0,-79,1])
        	x06,y06,z06 = np.matmul(projection_matrix_1,[79,0,-79,1])
        	x07,y07,z07 = np.matmul(projection_matrix_1,[79,79,-79,1])
        	x08,y08,z08 = np.matmul(projection_matrix_1,[0,79,-79,1])

        	x11,y11,z11 = np.matmul(projection_matrix_2,[0,0,0,1])
        	x12,y12,z12 = np.matmul(projection_matrix_2,[79,0,0,1])
        	x13,y13,z13 = np.matmul(projection_matrix_2,[79,79,0,1])
        	x14,y14,z14 = np.matmul(projection_matrix_2,[0,79,0,1])
        	x15,y15,z15 = np.matmul(projection_matrix_2,[0,0,-79,1])
        	x16,y16,z16 = np.matmul(projection_matrix_2,[79,0,-79,1])
        	x17,y17,z17 = np.matmul(projection_matrix_2,[79,79,-79,1])
        	x18,y18,z18 = np.matmul(projection_matrix_2,[0,79,-79,1])

        	x21,y21,z21 = np.matmul(projection_matrix_3,[0,0,0,1])
        	x22,y22,z22 = np.matmul(projection_matrix_3,[79,0,0,1])
        	x23,y23,z23 = np.matmul(projection_matrix_3,[79,79,0,1])
        	x24,y24,z24 = np.matmul(projection_matrix_3,[0,79,0,1])
        	x25,y25,z25 = np.matmul(projection_matrix_3,[0,0,-79,1])
        	x26,y26,z26 = np.matmul(projection_matrix_3,[79,0,-79,1])
        	x27,y27,z27 = np.matmul(projection_matrix_3,[79,79,-79,1])
        	x28,y28,z28 = np.matmul(projection_matrix_3,[0,79,-79,1])
                

        	cv.line(frame,(int(x01/z01), int(y01/z01)), (int(x02/z02), int(y02/z02)), (0, 0, 255), 6)
        	cv.line(frame,(int(x02/z02), int(y02/z02)), (int(x03/z03), int(y03/z03)), (0, 0, 255), 6)
        	cv.line(frame,(int(x03/z03), int(y03/z03)), (int(x04/z04), int(y04/z04)), (0, 0, 255), 6)
        	cv.line(frame,(int(x04/z04), int(y04/z04)), (int(x01/z01), int(y01/z01)), (0, 0, 255), 6)
        	cv.line(frame,(int(x01/z01), int(y01/z01)), (int(x05/z05), int(y05/z05)), (0, 0, 255), 6)
        	cv.line(frame,(int(x02/z02), int(y02/z02)), (int(x06/z06), int(y06/z06)), (0, 0, 255), 6)
        	cv.line(frame,(int(x03/z03), int(y03/z03)), (int(x07/z07), int(y07/z07)), (0, 0, 255), 6)
        	cv.line(frame,(int(x04/z04), int(y04/z04)), (int(x08/z08), int(y08/z08)), (0, 0, 255), 6)
        	cv.line(frame,(int(x05/z05), int(y05/z05)), (int(x06/z06), int(y06/z06)), (0, 0, 255), 6)
        	cv.line(frame,(int(x06/z06), int(y06/z06)), (int(x07/z07), int(y07/z07)), (0, 0, 255), 6)
        	cv.line(frame,(int(x07/z07), int(y07/z07)), (int(x08/z08), int(y08/z08)), (0, 0, 255), 6)
        	cv.line(frame,(int(x08/z08), int(y08/z08)), (int(x05/z05), int(y05/z05)), (0, 0, 255), 6)

        	cv.line(frame,(int(x11/z11), int(y11/z11)), (int(x12/z12), int(y12/z12)), (0, 0, 255), 6)
        	cv.line(frame,(int(x12/z12), int(y12/z12)), (int(x13/z13), int(y13/z13)), (0, 0, 255), 6)
        	cv.line(frame,(int(x13/z13), int(y13/z13)), (int(x14/z14), int(y14/z14)), (0, 0, 255), 6)
        	cv.line(frame,(int(x14/z14), int(y14/z14)), (int(x11/z11), int(y11/z11)), (0, 0, 255), 6)
        	cv.line(frame,(int(x11/z11), int(y11/z11)), (int(x15/z15), int(y15/z15)), (0, 0, 255), 6)
        	cv.line(frame,(int(x12/z12), int(y12/z12)), (int(x16/z16), int(y16/z16)), (0, 0, 255), 6)
        	cv.line(frame,(int(x13/z13), int(y13/z13)), (int(x17/z17), int(y17/z17)), (0, 0, 255), 6)
        	cv.line(frame,(int(x14/z14), int(y14/z14)), (int(x18/z18), int(y18/z18)), (0, 0, 255), 6)
        	cv.line(frame,(int(x15/z15), int(y15/z15)), (int(x16/z16), int(y16/z16)), (0, 0, 255), 6)
        	cv.line(frame,(int(x16/z16), int(y16/z16)), (int(x17/z17), int(y17/z17)), (0, 0, 255), 6)
        	cv.line(frame,(int(x17/z17), int(y17/z17)), (int(x18/z18), int(y18/z18)), (0, 0, 255), 6)
        	cv.line(frame,(int(x18/z18), int(y18/z18)), (int(x15/z15), int(y15/z15)), (0, 0, 255), 6)

        	cv.line(frame,(int(x21/z21), int(y21/z21)), (int(x22/z22), int(y22/z22)), (0, 0, 255), 6)
        	cv.line(frame,(int(x22/z22), int(y22/z22)), (int(x23/z23), int(y23/z23)), (0, 0, 255), 6)
        	cv.line(frame,(int(x23/z23), int(y23/z23)), (int(x24/z24), int(y24/z24)), (0, 0, 255), 6)
        	cv.line(frame,(int(x24/z24), int(y24/z24)), (int(x21/z21), int(y21/z21)), (0, 0, 255), 6)
        	cv.line(frame,(int(x21/z21), int(y21/z21)), (int(x25/z25), int(y25/z25)), (0, 0, 255), 6)
        	cv.line(frame,(int(x22/z22), int(y22/z22)), (int(x26/z26), int(y26/z26)), (0, 0, 255), 6)
        	cv.line(frame,(int(x23/z23), int(y23/z23)), (int(x27/z27), int(y27/z27)), (0, 0, 255), 6)
        	cv.line(frame,(int(x24/z24), int(y24/z24)), (int(x28/z28), int(y28/z28)), (0, 0, 255), 6)
        	cv.line(frame,(int(x25/z25), int(y25/z25)), (int(x26/z26), int(y26/z26)), (0, 0, 255), 6)
        	cv.line(frame,(int(x26/z26), int(y26/z26)), (int(x27/z27), int(y27/z27)), (0, 0, 255), 6)
        	cv.line(frame,(int(x27/z27), int(y27/z27)), (int(x28/z28), int(y28/z28)), (0, 0, 255), 6)
        	cv.line(frame,(int(x28/z28), int(y28/z28)), (int(x25/z25), int(y25/z25)), (0, 0, 255), 6)
        	
        	cv.imshow("cubeline", frame)
        	return frame

def main():
	frame_count = 0
	lena = read_lena()
	lena_shape = lena.shape
	lena_before = np.float32([[0,0],[512,0],[512,512],[0,512]])

	reference = read_marker()
	reference_h, reference_w, l = reference.shape
	
	reference_corners = np.float32([[0,0],[200,0],[200,200],[0,200]])
	po1 = 0
	po2 = 0
	po3 = 0
	if func == "1":
		print("Detecting AR Tag and finding its ID-")
		while(vid.isOpened()):
			contours, frame=get_all_contours()
			if contours is not None:
				if len(contours) == 3:
					tag_detection_1(contours[0], reference_corners, frame)
					tag_detection_2(contours[1], reference_corners, frame)
					tag_detection_3(contours[2], reference_corners, frame)
			cv.imshow("Video", frame)
			if val == "1":
				out.write(frame)
			key = cv.waitKey(1)
			if key == 27:
				break
	elif func == "2":
		count = 0
		print("Superimposing Lena")
		while(vid.isOpened()):
			contours, frame = get_all_contours()
			if contours is not None:
				if len(contours) == 3:
					po1 = lena_transform_1(po1, contours[0], lena_before, frame)
					po2 = lena_transform_2(po2, contours[1], lena_before, frame)
					po3 = lena_transform_3(po3, contours[2], lena_before, frame)
			cv.imshow("Video", frame)
			if val == "1":
				out.write(frame)
			key = cv.waitKey(1)
			if key == 27:
				break
	elif func == "3":
		print("Tracking AR tag and drawing Virtual Cube")
		while(vid.isOpened()):
			contours, frame = get_all_contours()
			if contours is not None:
				cube_frames = draw_cube_multi(contours, frame)
				if val == "1":
					out.write(cube_frames)
			key = cv.waitKey(1)
			if key == 27:
				break
	else:
		print("Please Try Again with either 1 or 2 or 3")
		main()

	vid.release()
	if val == "1":
		out.release()
	cv.destroyAllWindows()

main()