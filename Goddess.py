
import cv2
import numpy as np
import turtle

# Load reference image in grayscale
image_path = "mataji.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not find '{image_path}'. Make sure it is in the same folder.")
    raise SystemExit

# Set canvas resolution
target_width = 750
aspect_ratio = image.shape[0] / image.shape[1]
target_height = int(target_width * aspect_ratio)

resized_image = cv2.resize(image, (target_width, target_height))

# Thresholding to preserve details
_, binary = cv2.threshold(
    resized_image, 180, 255, cv2.THRESH_BINARY_INV
)

# Detect contours and nested details
contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
)

if hierarchy is None:
    print("Error: No shapes detected.")
    raise SystemExit

hierarchy = hierarchy[0]

# Turtle window setup
screen = turtle.Screen()
screen.setup(
    width=target_width + 60,
    height=target_height + 60
)
screen.title("Mataji and Lion - Live Drawing")
screen.bgcolor("white")

# MODERATE DRAWING SPEED
screen.tracer(2, 10)

# Pen setup
pen = turtle.Turtle()
pen.speed(5)        # Moderate speed: not too slow, not too fast
pen.pensize(1)
pen.shape("classic")
pen.showturtle()

# Coordinate conversion
half_w = target_width / 2
half_h = target_height / 2

def to_turtle_coords(x, y):
    return (x - half_w, half_h - y)

# Find contour nesting depth
def get_contour_depth(idx):
    depth = 0
    parent = hierarchy[idx][3]

    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]

    return depth

# Draw each contour
def render_shape(cnt, fill_col, border_col):
    first_pt = cnt[0][0]
    tx, ty = to_turtle_coords(first_pt[0], first_pt[1])

    pen.penup()
    pen.goto(tx, ty)
    pen.pendown()

    pen.color(border_col, fill_col)
    pen.begin_fill()

    # Skip fewer points to preserve image details
    step = 2 if len(cnt) > 80 else 1

    for i in range(1, len(cnt), step):
        pt = cnt[i][0]
        tx, ty = to_turtle_coords(pt[0], pt[1])
        pen.goto(tx, ty)

    # Close the contour
    pen.goto(to_turtle_coords(first_pt[0], first_pt[1]))
    pen.end_fill()

# Group contours by depth
max_depth = max(
    get_contour_depth(i) for i in range(len(contours))
)

depth_layers = {d: [] for d in range(max_depth + 1)}

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) < 4:
        continue

    depth = get_contour_depth(i)
    depth_layers[depth].append(cnt)

print("Drawing in progress... Watch the cursor!")

# Draw outer shapes and inner details
for depth in range(max_depth + 1):
    fill_color = "black" if depth % 2 == 0 else "white"
    border_color = fill_color

    for cnt in depth_layers[depth]:
        render_shape(cnt, fill_color, border_color)

# Finish
pen.hideturtle()
screen.update()

print("Drawing complete!")
screen.mainloop()
