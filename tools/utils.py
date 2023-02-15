import cv2

joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [12, 14], [13, 15], [14, 16]]


def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length:
        if W>H:
            factor = max_length / W
            W_resize = max_length
            H_resize = int(H * factor)
        else:
            factor = max_length / H
            H_resize = max_length
            W_resize = int(W * factor)
        frame = cv2.resize(frame, (W_resize, H_resize), interpolation=cv2.INTER_AREA)
        return frame, W_resize, H_resize, factor

    else:
        return frame, W, H, 1.


def draw_2Dimg(img, kpts):
    # kpts : (N, 17, 3)  3-->(x, y, score)
    im = img.copy()
    for kpt in kpts:
        for item in kpt:
            score = item[-1]
            if score > 0.1:
                x, y = int(item[0]), int(item[1])
                cv2.circle(im, (x, y), 1, (255, 5, 0), 5)
        for pair in joint_pairs:
            j, j_parent = pair
            pt1 = (int(kpt[j][0]), int(kpt[j][1]))
            pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
            cv2.line(im, pt1, pt2, (0,255,0), 2)

    return im