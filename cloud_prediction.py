import cv2
import os
import numpy as np


def create_imgList(imgList, path):
    listImg = os.listdir(path)
    for i in listImg:
        img = cv2.imread(f'{path}/{i}')
        imgList.append(img)
    return imgList


def main():
    imgList = []

    imgList = create_imgList(imgList, "cloudless")
    imgList = create_imgList(imgList, "cloudmid")
    imgList = create_imgList(imgList, "cloudmuch")
    imgList = create_imgList(imgList, "cloudrain")

    original = cv2.resize(cv2.imread("cloudrain11.jpg"), (600, 400))
    cv2.imshow('input', original)
    

    max_compare = 0
    indexImg = 0

    hist_original = cv2.calcHist([original], [0], None, [256], [0, 255])
    for i in range(len(imgList)):
        reference = imgList[i]
        hist_reference = cv2.calcHist([reference], [0], None, [256], [0, 255])

        cv2.normalize(hist_original, hist_original, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hist_reference, hist_reference, 0, 255, cv2.NORM_MINMAX)

        compare_value = cv2.compareHist(
            hist_original, hist_reference, cv2.HISTCMP_CORREL)
        if compare_value > max_compare:
            max_compare = compare_value
            indexImg = i

    rangefilecloudless = len(os.listdir("cloudless"))+1
    rangefilecloudmid = (len(os.listdir("cloudless"))) + \
        (len(os.listdir("cloudmid"))+1)
    rangefilecloudmuch = (len(os.listdir("cloudless"))) + \
        (len(os.listdir("cloudmid")))+(len(os.listdir("cloudmuch"))+1)
    rangefilecloudrain = (len(os.listdir("cloudless")))+(len(os.listdir("cloudmid"))) + \
        (len(os.listdir("cloudmuch")))+(len(os.listdir("cloudrain"))+1)

    if (indexImg+1) in range(1, rangefilecloudless):  # (1,8)
        print("There is a less clouds, the sky is clearing up.")
    elif (indexImg+1) in range(rangefilecloudless, rangefilecloudmid):  # (8,13)
        print("There is a moderately clouds, the sky is clearing up.")
    elif (indexImg+1) in range(rangefilecloudmid, rangefilecloudmuch):  # (13,19)
        print("There is a lot of clouds, thunderstorms may occurs.")
    elif (indexImg+1) in range(rangefilecloudmuch, rangefilecloudrain):  # (19,24)
        print("There is a lot of rain clouds, thunderstorms occurs.")

    print('Match result :',max_compare)
    # print(indexImg+1)
    cv2.imshow("match_result", cv2.resize(imgList[indexImg], (600, 400)))
    # cv2.imshow("Image", original)
    img = original
    print('Clouds percentage : ', percent(img), '%')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def percent(img):
    white = [80, 80, 80]
    # You define an interval that covers the values
    # in the tuple and are below and above them by 20
    diff = 175
    # Be aware that opencv loads image in BGR format,
    # that's why the color values have been adjusted here:
    boundaries = [([white[2], white[1], white[0]],
            [white[2]+diff, white[1]+diff, white[0]+diff])]

    # Scale your BIG image into a small one:
    scalePercent = 1

    # Calculate the new dimensions
    width = int(img.shape[1] * scalePercent)
    height = int(img.shape[0] * scalePercent)
    newSize = (width, height)

    # Resize the image:
    img = cv2.resize(img, newSize, None, None, None, cv2.INTER_AREA)

    # # check out the image resized:
    # cv2.imshow("img resized", img)
    # cv2.waitKey(0)


    # for each range in your boundary list:
    for (lower, upper) in boundaries:

        # You get the lower and upper part of the interval:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # cv2.inRange is used to binarize (i.e., render in white/black) an image
        # All the pixels that fall inside your interval [lower, uipper] will be white
        # All the pixels that do not fall inside this interval will
        # be rendered in black, for all three channels:
        mask = cv2.inRange(img, lower, upper)

        # Check out the binary mask:
        # cv2.imshow("binary mask", mask)
        # cv2.waitKey(0)

        # Now, you AND the mask and the input image
        # All the pixels that are white in the mask will
        # survive the AND operation, all the black pixels
        # will remain black
        output = cv2.bitwise_and(img, img, mask=mask)

        # Check out the ANDed mask:
        # cv2.imshow("Original", img)
        # cv2.imshow("ANDed mask", output)
        # cv2.waitKey(0)

        # You can use the mask to count the number of white pixels.
        # Remember that the white pixels in the mask are those that
        # fall in your defined range, that is, every white pixel corresponds
        # to a white pixel. Divide by the image size and you got the
        # percentage of white pixels in the original image:
        ratio_white = cv2.countNonZero(mask)/(img.size/3)

        # This is the color percent calculation, considering the resize I did earlier.
        colorPercent = (ratio_white * 100) / scalePercent

        # Print the color percent, use 2 figures past the decimal point
        percents = np.round(colorPercent, 2)

        # numpy's hstack is used to stack two images horizontally,
        # so you see the various images generated in one figure:
        # cv2.imshow("images", np.hstack([img, output]))
        # cv2.waitKey(0)
        return percents

main()