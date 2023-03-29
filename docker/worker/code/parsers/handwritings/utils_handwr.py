import numpy as np


def crop_to_list_of_squares(img: np.ndarray, size=(28, 28)) -> list:
    """cut long field to list of short ones and put in white rectange"""
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    # interpolation = cv.INTER_AREA if ms > (size[0] + size[1]) // 2 else cv.INTER_CUBIC
    l = []
    LENGTH_OF_PIECE = 4
    if h < w:
        for i in range(round(w / h) // LENGTH_OF_PIECE):  # //2 - double length of piece
            piece = img[:, h * i * LENGTH_OF_PIECE: h * (i + 1) * LENGTH_OF_PIECE]  # *2 = double piece
            # import matplotlib.pyplot as plot
            # plot.imshow(piece)
            # plot.show()

            hh, ww = piece.shape[:2]
            ms = hh if hh > ww else ww
            x_pos = (ms - ww) // 2
            y_pos = (ms - hh) // 2

            if len(piece.shape) == 2:
                mask = np.ones((ms, ms), dtype=piece.dtype) * 255  # white
                mask[y_pos:y_pos + hh, x_pos:x_pos + ww] = piece[:hh, :ww]
            else:
                mask = np.ones((ms, ms, c), dtype=piece.dtype) * 255  # white
                mask[y_pos:y_pos + hh, x_pos:x_pos + ww, :] = piece[:hh, :ww, :]
            piece = mask

            l.append(piece)
        # if w / h - int(w / h) > 0.2:
        #     piece = img[:, w - h:w]
        #     l.append(piece)

    return l


def test_line_to_list_of_rois():
    import cv2 as cv
    p = '/home/u2/tmp.png'
    img = cv.imread(p)
    import matplotlib.pyplot as plot
    plot.imshow(img)
    plot.show()
    # rois = crop_to_list_of_squares(img)
    # print(rois)


if __name__ == '__main__':
    test_line_to_list_of_rois()