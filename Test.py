# TrainAndTest.py
# github/imneonizer
import cv2
import numpy as np
import operator
import os

test_image = "img4.jpg"

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def crop_img(test_image):
    imgTest = cv2.imread(test_image)          # read in testing numbers image
    img = imgTest.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold the grayscale image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # use morphology erode to blur horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    # use morphology open to remove thin lines from dotted lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    cv2.namedWindow('Overlay Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Overlay Image', morph)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = morph.copy()
    #cntrs, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = img.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)

    cv2.floodFill(result, mask, (0, 50), (0, 0, 0), (3, 151, 65), (3, 151, 65), flags=8)

    
    morph = result.copy()
    cv2.namedWindow('Overlay Image1', cv2.WINDOW_NORMAL)
    cv2.imshow('Overlay Image1', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    # find the topmost box
 
    ythresh = 1000000
    for c in cntrs:
        box = cv2.boundingRect(c)
        x,y,w,h = box
        if y < ythresh:
            topbox = box
            ythresh = y
 
    # Draw contours excluding the topmost box
    result = img.copy()
    crop_folder = './crop/'
    files = os.listdir(crop_folder)
    for file in files:
        file_path = os.path.join(crop_folder, file)
        os.remove(file_path)
        print(f"Đã xóa file: {file_path}")

    for c in cntrs:
        box = cv2.boundingRect(c)
        if box != topbox:
            x,y,w,h = box
            a = result.copy()
            cropped_img = a[y:y+h, x:x+w]
            filename = f"cropped_{y}_{y+h}_{x}_{x+w}.jpg"
            disk = f'./crop/{filename}'
            print(f"Đang lưu ảnh vào: {disk}")
            cv2.imwrite(disk, cropped_img)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.namedWindow('Overlay Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Overlay Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
crop_img(test_image)

###################################################################################################
class ContourWithData():
    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def test(imgTestingNumbers):
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    a = ""
    if a == "":

        #imgTestingNumbers = cv2.imread(test_image)          # read in testing numbers image
        #imgTestingNumbers = i   
        if imgTestingNumbers is None:                           # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return                                              # and exit function (which exits program)
        # end if

        imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                            # filter image from grayscale to black and white
        imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                        255,                                  # make pixels that pass the threshold full white
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                        cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                        11,                                   # size of a pixel neighborhood used to calculate threshold value
                                        2)                                    # constant subtracted from the mean or weighted mean

        imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
        '''
        _, npaContours, npaHierarchy = cv2.findContours(imgThresh.copy(),             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                    cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                    cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points
        '''
        npaContours, npaHierarchy = cv2.findContours(imgThresh.copy(),             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                    cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                    cv2.CHAIN_APPROX_SIMPLE)
        for npaContour in npaContours:                             # for each contour
            contourWithData = ContourWithData()                                             # instantiate a contour with data object
            contourWithData.npaContour = npaContour                                         # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
            contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
            allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
        # end for

        for contourWithData in allContoursWithData:                 # for all contours
            if contourWithData.checkIfContourIsValid():             # check if valid
                validContoursWithData.append(contourWithData)       # if so, append to valid contour list
            # end if
        # end for

        validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

        strFinalString = ""        # declare final string, this will have the final number sequence by the end of the program
        for contourWithData in validContoursWithData:            # for each contour
            cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                        (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                        (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                        (0, 255, 0),              # green
                        2)                        # thickness

            imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                            contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

            if imgROI.shape[0] != 0 and imgROI.shape[1] != 0 :
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

                npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

                npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

                retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

                strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

                strFinalString = strFinalString + strCurrentChar            # append current char to full string

        return strFinalString, imgTestingNumbers

folder_path = './crop/'
final =[]
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    imgTestingNumbers = cv2.imread(file_path)
    strFinalString, imgTestingNumbers = test(imgTestingNumbers)
        # Đảm bảo là tệp tin là ảnh
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Ghi đè ảnh mới lên tệp tin hiện tại
        cv2.imwrite(file_path, imgTestingNumbers)
    final.append((strFinalString, file_path))

filename = ""
text = ""
check = 0
result_array = []
for i in final:
    if len(i[0]) == 8:
        if any(char.isalpha() for char in i[0]):
            continue  
        print(i[1])
        text = i[0].replace("O", "0")
        count_digits = 0
        for char in text:
            if char.isdigit():
                count_digits += 1
        if count_digits >= check:
            filename = i[1]
            text = i[0]
            result_array.append(text)

print("Result Array:", result_array)


if filename != "":
        name_file = filename.split(".")
        temp = name_file[1].split("_")
        # Đọc ảnh lớn
        large_image = cv2.imread(test_image)
    
        # Tọa độ và kích thước của ảnh nhỏ (điều chỉnh theo tọa độ thực của bạn)
        y, y_end, x, x_end = int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4])

        # Đường dẫn đến ảnh nhỏ
        small_image_path = filename

        # Đọc ảnh nhỏ
        small_image = cv2.imread(small_image_path, cv2.IMREAD_UNCHANGED)

        # Kích thước ảnh nhỏ
        small_height, small_width = small_image.shape[:2]
        # Chồng ảnh nhỏ lên ảnh lớn
        # Chồng ảnh nhỏ lên ảnh lớn
        # Tính chiều cao và chiều rộng của vùng chồng lên
        height = y_end - y
        width = x_end - x

        # Kiểm tra và điều chỉnh kích thước của ảnh nhỏ
        small_image = cv2.resize(small_image, (width, height))

        # Chồng ảnh nhỏ lên ảnh lớn
        large_image[y:y+height, x:x+width, :] = small_image
        # Tọa độ và font
        font_position = (50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 0, 0)  # Màu trắng
        font_thickness = 5

        # Ghi chữ lên ảnh
        cv2.putText(large_image, text, font_position, font, font_scale, font_color, font_thickness)
        # Hiển thị ảnh sau khi chồng lên
        cv2.namedWindow('Overlay Image', cv2.WINDOW_NORMAL)

        cv2.imshow('Overlay Image', large_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    text = "Khong tim thay ma so sinh vien"
    # Ghi chữ lên ảnh
    large_image = cv2.imread(test_image)
    # Tọa độ và font
    font_position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # Màu trắng
    font_thickness = 2
    cv2.putText(large_image, text, font_position, font, font_scale, font_color, font_thickness)
    # Hiển thị ảnh sau khi chồng lên
    cv2.imshow('Overlay Image', large_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    





