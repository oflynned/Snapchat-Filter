import math

import cv2
import dlib
import numpy as np

from video import create_capture

predictor_path = "resources/landmark_predictor.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
LOWER_NOSE = list(range(31, 36))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


def get_cam_frame(cam):
    ret, img = cam.read()
    img = cv2.resize(img, (800, 450))
    return img


def get_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return -1

    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def face_swap(img1, landmarks1, img2, landmarks2):
    m = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

    mask = get_face_mask(img2, landmarks2)
    warped_mask = warp_im(mask, m, img1.shape)
    combined_mask = np.max([get_face_mask(img1, landmarks1), warped_mask], axis=0)

    warped_img2 = warp_im(img2, m, img1.shape)
    warped_corrected_img2 = correct_colours(img1, warped_img2, landmarks1)

    return img1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask


def get_rotated_points(point, anchor, deg_angle):
    angle = math.radians(deg_angle)
    px, py = point
    ox, oy = anchor

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [int(qx), int(qy)]


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def glasses_filter(cam, glasses):
    face = get_cam_frame(cam)
    landmarks = get_landmarks(face)

    pts1 = np.float32([[0, 0], [599, 0], [0, 208], [599, 208]])

    if type(landmarks) is not int:
        """
        GLASSES ANCHOR POINTS:

        17 & 26 edges of left eye and right eye (left and right extrema)
        0 & 16 edges of face across eyes (other left and right extra, interpolate between 0 & 17, 16 & 26 for half way points)
        19 & 24 top of left and right brows (top extreme)
        27 is centre of the eyes on the nose (centre of glasses)
        28 is the bottom threshold of glasses (perhaps interpolate between 27 & 28 if too low) (bottom extreme)
        """

        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        glasses_width = face_width * 1.0

        # top and bottom of left eye
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height * 1.2

        # generate bounding box from the anchor points
        anchor_point = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, anchor_point, face_angle)

        tr = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, anchor_point, face_angle)

        bl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, anchor_point, face_angle)

        br = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, anchor_point, face_angle)

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)

        rotated = cv2.warpPerspective(glasses, m, (face.shape[1], face.shape[0]))
        result_2 = blend_transparent(face, rotated)

        for p in pts:
            pos = (p[0], p[1])
            cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
            cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))

        cv2.imshow("Glasses Filter", result_2)


def moustache_filter(cam, moustache):
    face = get_cam_frame(cam)
    landmarks = get_landmarks(face)

    pts1 = np.float32([[0, 0], [599, 0], [0, 208], [599, 208]])

    """
    MOUSTACHE ANCHOR POINTS

    centre anchor point is midway between 34 (top of philtrum) and 54 (bottom of philtrum)
    width can be determined by the eyes as the mouth can move
    height also determined by the eyes as before
    generate as before and just modify multiplier coefficients & translate to anchor point?


    ^^^ mouth and jaw can move, use eyes as anchor point initially then translate to philtrum position
    """

    if type(landmarks) is not int:
        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        moustache_width = face_width * 0.8

        # top and bottom of left eye
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height * 0.8

        # generate bounding box from the anchor points
        brow_anchor = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(brow_anchor[0] - (moustache_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, brow_anchor, face_angle)

        tr = [int(brow_anchor[0] + (moustache_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, brow_anchor, face_angle)

        bl = [int(brow_anchor[0] - (moustache_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, brow_anchor, face_angle)

        br = [int(brow_anchor[0] + (moustache_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, brow_anchor, face_angle)

        # locate new location for moustache on philtrum
        top_philtrum_point = [landmarks[33, 0], landmarks[33, 1]]
        bottom_philtrum_point = [landmarks[51, 0], landmarks[51, 1]]
        philtrum_anchor = [(top_philtrum_point[0] + bottom_philtrum_point[0]) / 2,
                           (top_philtrum_point[1] + bottom_philtrum_point[1]) / 2]

        # determine distance from old origin to new origin and translate
        anchor_distance = [int(philtrum_anchor[0] - brow_anchor[0]), int(philtrum_anchor[1] - brow_anchor[1])]
        rot_tl[0] += anchor_distance[0]
        rot_tl[1] += anchor_distance[1]
        rot_tr[0] += anchor_distance[0]
        rot_tr[1] += anchor_distance[1]
        rot_bl[0] += anchor_distance[0]
        rot_bl[1] += anchor_distance[1]
        rot_br[0] += anchor_distance[0]
        rot_br[1] += anchor_distance[1]

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)

        rotated = cv2.warpPerspective(moustache, m, (face.shape[1], face.shape[0]))
        result_2 = blend_transparent(face, rotated)

        # annotate_landmarks(result_2, landmarks)

        for p in pts:
            pos = (p[0], p[1])
            cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
            cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))

        cv2.imshow("Moustache Filter", result_2)


def face_swap_filter(cam, swap_img, swap_img_landmarks):
    me_img = get_cam_frame(cam)
    me_img = cv2.resize(me_img, (me_img.shape[1] * SCALE_FACTOR, me_img.shape[0] * SCALE_FACTOR))
    me_landmarks = get_landmarks(me_img)

    # me_img, me_landmarks = read_im_and_landmarks("resources/bryan_cranston.png")

    if type(me_landmarks) is not int:
        m = transformation_from_points(me_landmarks[ALIGN_POINTS], swap_img_landmarks[ALIGN_POINTS])

        mask = get_face_mask(swap_img, swap_img_landmarks)
        warped_mask = warp_im(mask, m, me_img.shape)
        combined_mask = np.max([get_face_mask(me_img, me_landmarks), warped_mask], axis=0)

        warped_swap = warp_im(swap_img, m, me_img.shape)
        warped_corrected_swap = correct_colours(me_img, warped_swap, me_landmarks)

        output_im = me_img * (1.0 - combined_mask) + warped_corrected_swap * combined_mask
        cv2.imwrite("swap_output.png", output_im)
        out = cv2.imread("swap_output.png", 1)
        cv2.imshow("Swap Output", out)


def main():
    cam = create_capture(0)

    glasses = cv2.imread('resources/glasses.png', -1)
    moustache = cv2.imread('resources/moustache.png', -1)

    swap_img = cv2.imread('resources/bryan_cranston.png', -1)
    swap_img_landmarks = get_landmarks(swap_img)

    while True:
        # glasses_filter(cam, glasses)
        # moustache_filter(cam, moustache)
        face_swap_filter(cam, swap_img, swap_img_landmarks)

        if 0xFF & cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
