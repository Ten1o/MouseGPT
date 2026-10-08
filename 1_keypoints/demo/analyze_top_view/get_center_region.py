import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def circle_center_radius(point_list):
    p1, p2, p3 = point_list
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3


    denom = 2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2)


    h = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / denom
    k = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / denom


    radius = ((x1 - h)**2 + (y1 - k)**2)**0.5

    return (h, k), radius

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        print(f'Clicked at x = {event.xdata}, y = {event.ydata}')
        point_list.append([event.xdata, event.ydata])
        ax.plot([event.xdata], [event.ydata], 'ro')

        if len(point_list) >= 3:
            center, radius = circle_center_radius(point_list)
            print(center, radius)

            circle = plt.Circle(center, radius, color='b', fill=False)
            ax.add_artist(circle)
            circle = plt.Circle(center, radius/2, color='g', fill=False)
            ax.add_artist(circle)

        plt.show()


if __name__ == '__main__':

    video_path = os.environ['MOUSEGPT_VIDEO']
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()

    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.set_title('click on circle')

    point_list = []

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


    #(761.136508231083, 357.81895986184674) 391.1801746120894
