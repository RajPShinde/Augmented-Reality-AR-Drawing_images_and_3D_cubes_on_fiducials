import numpy as np
import copy
import os, sys
import argparse
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
	U,S,V = np.linalg.svd(A, full_matrices=True)
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
		gblur = cv.GaussianBlur(frame, (7, 7), 0)
		gray = cv.cvtColor(gblur, cv.COLOR_BGR2GRAY)
		ret2, threshold = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
		contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv.contourArea, reverse = True)[1:]
		for i in contours:
			perimeter = cv.arcLength(i, True)
			approx = cv.approxPolyDP(i, 0.02*perimeter, True)
			if len(approx) == 4:
				tagCnt = approx
				break
		cv.drawContours(frame, tagCnt, -1, (0,0,255), 3)
	return tagCnt,frame

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
	# to print the array matrix
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
	if x1.all() >= 0 and x1.all() < 1920 and y1.all() >= 0 and y1.all() < 1080: # and len(x1) < 1920 and len(y1) < 1080:
		copy[temp_y, temp_x] = source[y1, x1]
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
		source[y1, x1] = copy[temp_y, temp_x]
	return source

def tag_detection(contours, target, frame):
	dest = target
	source = np.float32([[contours[0][0][0],contours[0][0][1]],[contours[1][0][0],contours[1][0][1]],
                [contours[2][0][0],contours[2][0][1]],[contours[3][0][0],contours[3][0][1]]])
	
	h_matrix = homography_estimation(source, dest)
	
	frame_copy = np.zeros((200, 200, 3), np.uint8)
	warped_frame = warp_inv(h_matrix, contours, frame, frame_copy)
	_,warped_threshold= cv.threshold(warped_frame, 200, 255, cv.THRESH_BINARY)
	tag_orientation, tag_id = find_tag_id(warped_threshold)
	
	print("Tag Orientation: ", tag_orientation)
	print("Tag ID: ", tag_id)
	font = cv.FONT_HERSHEY_SIMPLEX
	x = contours[0][0][0]
	y = contours[0][0][1] - 100 
	str_text = "ID: " + str(tag_id) + " Orientation: " + str(tag_orientation)
	cv.putText(frame, str_text, (x, y), font, 0.8, (0, 0, 255), 2, cv.LINE_AA)
	cv.imshow("Tag",warped_frame)
	cv.imshow("Video", frame)
	if val == "1":
		out.write(frame)
	
def lena_transform(var, contours, target, frame):    
	po = var
	if frame is not None:
		source = np.float32([[contours[0][0][0],contours[0][0][1]],[contours[1][0][0],contours[1][0][1]], [contours[2][0][0],contours[2][0][1]],[contours[3][0][0],contours[3][0][1]]])
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

			cv.imshow("Lena",Lena)
			cv.imshow("Tag",warped_threshold)
	if val == "1":
		out.write(Lena)
	return po

def draw_cube(contours, frame):
    if contours is not None:
        target = np.array([[0,0], [0,79], [79,79], [79,0]])
        if (len(contours) != 0):
            contours = np.float32([[contours[0][0][0],contours[0][0][1]],[contours[1][0][0],contours[1][0][1]],
                [contours[2][0][0],contours[2][0][1]],[contours[3][0][0],contours[3][0][1]]])
            h_matrix = homography_estimation_cube(target, contours)
            projection_matrix = calculate_projection_matrix(h_matrix, K)

            x1,y1,z1 = np.matmul(projection_matrix,[0,0,0,1])
            x2,y2,z2 = np.matmul(projection_matrix,[79,0,0,1])
            x3,y3,z3 = np.matmul(projection_matrix,[79,79,0,1])
            x4,y4,z4 = np.matmul(projection_matrix,[0,79,0,1])
            x5,y5,z5 = np.matmul(projection_matrix,[0,0,-79,1])
            x6,y6,z6 = np.matmul(projection_matrix,[79,0,-79,1])
            x7,y7,z7 = np.matmul(projection_matrix,[79,79,-79,1])
            x8,y8,z8 = np.matmul(projection_matrix,[0,79,-79,1])

           
            cv.line(frame,(int(x1/z1), int(y1/z1)), (int(x2/z2), int(y2/z2)), (0, 0, 255), 6)
            cv.line(frame,(int(x2/z2), int(y2/z2)), (int(x3/z3), int(y3/z3)), (0, 0, 255), 6)          
            cv.line(frame,(int(x3/z3), int(y3/z3)), (int(x4/z4), int(y4/z4)), (0, 0, 255), 6)
            cv.line(frame,(int(x4/z4), int(y4/z4)), (int(x1/z1), int(y1/z1)), (0, 0, 255), 6)
            cv.line(frame,(int(x1/z1), int(y1/z1)), (int(x5/z5), int(y5/z5)), (0, 0, 255), 6)
            cv.line(frame,(int(x2/z2), int(y2/z2)), (int(x6/z6), int(y6/z6)), (0, 0, 255), 6)
            cv.line(frame,(int(x3/z3), int(y3/z3)), (int(x7/z7), int(y7/z7)), (0, 0, 255), 6)
            cv.line(frame,(int(x4/z4), int(y4/z4)), (int(x8/z8), int(y8/z8)), (0, 0, 255), 6)
            cv.line(frame,(int(x5/z5), int(y5/z5)), (int(x6/z6), int(y6/z6)), (0, 0, 255), 6)
            cv.line(frame,(int(x6/z6), int(y6/z6)), (int(x7/z7), int(y7/z7)), (0, 0, 255), 6)          
            cv.line(frame,(int(x7/z7), int(y7/z7)), (int(x8/z8), int(y8/z8)), (0, 0, 255), 6)
            cv.line(frame,(int(x8/z8), int(y8/z8)), (int(x5/z5), int(y5/z5)), (0, 0, 255), 6)
           
            cv.imshow("cubeline", frame)

def main():
	frame_count = 0
	lena = read_lena()
	lena_shape = lena.shape
	lena_before = np.float32([[0,0],[512,0],[512,512],[0,512]])

	reference = read_marker()
	reference_h, reference_w, l = reference.shape
	reference_corners = np.float32([[0,0],[200,0],[200,200],[0,200]])
	po = 0
	if func == "1":
		print("Detecting AR Tag and finding its ID-")
		while(vid.isOpened()):
			contours, frame=get_all_contours()
			if contours is not None:
				tag_detection(contours, reference_corners, frame)
			key = cv.waitKey(1)
			if key == 27:
				break
	elif func == "2":
		count = 0
		print("Superimposing Lena")
		while(vid.isOpened()):
			contours, frame = get_all_contours()
			if contours is not None:
				po = lena_transform(po, contours, lena_before, frame)
			count = 1
			key = cv.waitKey(1)
			if key == 27:
				break
	elif func == "3":
		print("Tracking AR tag and drawing Virtual Cube")
		while(vid.isOpened()):
			contours, frame = get_all_contours()
			if contours is not None:
				draw_cube(contours, frame)
			count = 1
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