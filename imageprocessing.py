import cv2 as cv
import numpy as np
from tkinter import Tk, Button, Label, filedialog, Scale, HORIZONTAL, Frame, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Initialize global variables
original_image = None
processed_image = None
history = [] 
redo_stack = []

# Load image function
def load_image():
    global original_image, processed_image, history, redo_stack
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    if filepath:
        original_image = cv.imread(filepath)
        processed_image = original_image.copy()
        history.clear()
        redo_stack.clear()
        save_history()  # Save the initial state
        display_image(processed_image)

# Display image on GUI
def display_image(image):
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    tk_image = ImageTk.PhotoImage(pil_image)
    image_label.config(image=tk_image)
    image_label.photo = tk_image

# Save current state to history
def save_history():
    if processed_image is not None:
        history.append(processed_image.copy())
        if len(history) > 10:  # Limit history to 10 states
            history.pop(0)

# Undo function
def undo_image():
    global processed_image, redo_stack
    if history:
        redo_stack.append(history.pop())
        if history:
            processed_image = history[-1].copy()
            display_image(processed_image)

# Redo function
def redo_image():
    global processed_image
    if redo_stack:
        save_history()
        processed_image = redo_stack.pop()
        display_image(processed_image)

# Rotate image
def rotate_image():
    global processed_image
    save_history()
    processed_image = cv.rotate(processed_image, cv.ROTATE_90_CLOCKWISE)
    display_image(processed_image)

# Apply Gaussian blur
def gaussian_blur():
    def apply_blur():
        global processed_image
        save_history()
        blur_value = blur_slider.get()
        processed_image = cv.GaussianBlur(processed_image, (blur_value, blur_value), 0)
        display_image(processed_image)
        blur_window.destroy()

    if processed_image is None:
        return

    blur_window = Toplevel(root)
    blur_window.title("Adjust Blur")
    blur_window.geometry("300x150")
    blur_window.configure(bg="#f0f0f0")

    Label(blur_window, text="Adjust Blur Level", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

    blur_slider = Scale(blur_window, from_=1, to=51, resolution=2, orient=HORIZONTAL, length=200)
    blur_slider.set(7)  # Default value
    blur_slider.pack()

    Button(blur_window, text="Apply Blur", command=apply_blur, bg="#0984e3", fg="white", width=15).pack(pady=10)

# Apply Sepia filter
def sepia_filter():
    global processed_image
    save_history()
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    processed_image = cv.transform(processed_image, kernel)
    processed_image = np.clip(processed_image, 0, 255) # Clip values to prevent overflow
    display_image(processed_image)

# Apply Grayscale filter
def grayscale_filter():
    global processed_image
    save_history()
    processed_image = cv.cvtColor(processed_image, cv.COLOR_BGR2GRAY)
    processed_image = cv.cvtColor(processed_image, cv.COLOR_GRAY2BGR) # Compatibility Fix
    display_image(processed_image)

# Apply Canny edge detection
def canny_edge_detection():
    global processed_image
    save_history()
    gray_image = cv.cvtColor(processed_image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_image, threshold1=50, threshold2=150)
    processed_image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # Convert back to 3 channels for display
    display_image(processed_image)

# Resize image
def resize_image():
    def apply_resize():
        global processed_image
        save_history()
        width = width_slider.get()
        height = height_slider.get()
        processed_image = cv.resize(original_image, (width, height), interpolation=cv.INTER_AREA)# cv.INTER_AREA -> reducing the size of the image 
        display_image(processed_image)
        resize_window.destroy()

    if original_image is None:
        return

    resize_window = Toplevel(root)
    resize_window.title("Resize Image")
    resize_window.geometry("350x250")
    resize_window.configure(bg="#f0f0f0")

    Label(resize_window, text="Resize Image", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

    Label(resize_window, text="Width", bg="#f0f0f0").pack()
    width_slider = Scale(resize_window, from_=50, to=original_image.shape[1] * 2, orient=HORIZONTAL, length=200)
    width_slider.set(original_image.shape[1])
    width_slider.pack()

    Label(resize_window, text="Height", bg="#f0f0f0").pack()
    height_slider = Scale(resize_window, from_=50, to=original_image.shape[0] * 2, orient=HORIZONTAL, length=200)
    height_slider.set(original_image.shape[0])
    height_slider.pack()

    Button(resize_window, text="Apply Resize", command=apply_resize, bg="#0984e3", fg="white", width=15).pack(pady=10)

# Show histogram
def show_histogram():
    if processed_image is None:
        return

    gray_image = cv.cvtColor(processed_image, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

# Show color histogram
def show_color_histogram():
    if processed_image is None:
        return

    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for i, color in enumerate(colors):
        hist = cv.calcHist([processed_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

# Save image
def save_image():
    global processed_image
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if filepath:
        cv.imwrite(filepath, processed_image)
        print(f"Image saved at {filepath}")

# Adjust Brightness and Contrast
def adjust_brightness_contrast():
    def apply_adjustments():
        global processed_image
        save_history()
        alpha = contrast_slider.get() / 100  # Scale contrast to a factor
        beta = brightness_slider.get()      # Brightness offset
        processed_image = cv.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
        display_image(processed_image)
        preprocess_window.destroy()

    if processed_image is None:
        return

    preprocess_window = Toplevel(root)
    preprocess_window.title("Adjust Brightness & Contrast")
    preprocess_window.geometry("300x250")
    preprocess_window.configure(bg="#f0f0f0")

    Label(preprocess_window, text="Adjust Brightness & Contrast", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

    Label(preprocess_window, text="Contrast", bg="#f0f0f0").pack()
    contrast_slider = Scale(preprocess_window, from_=50, to=150, orient=HORIZONTAL, length=200)
    contrast_slider.set(100)
    contrast_slider.pack()

    Label(preprocess_window, text="Brightness", bg="#f0f0f0").pack()
    brightness_slider = Scale(preprocess_window, from_=-100, to=100, orient=HORIZONTAL, length=200)
    brightness_slider.set(0)
    brightness_slider.pack()

    Button(preprocess_window, text="Apply Adjustments", command=apply_adjustments, bg="#0984e3", fg="white", width=15).pack(pady=10)

def normalize_image():
    global processed_image
    save_history()
    processed_image = cv.normalize(processed_image, None, 0, 255, cv.NORM_MINMAX)
    display_image(processed_image)

def sharpen_image():
    global processed_image
    save_history()
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    processed_image = cv.filter2D(processed_image, -1, kernel)
    display_image(processed_image)

# GUI setup
root = Tk()
root.title("Image Processing")
root.geometry("1200x800")
root.configure(bg="#f0f0f0")

frame_top = Frame(root, bg="#dfe6e9", pady=10)
frame_top.pack(fill="x")

frame_left = Frame(root, bg="#f0f0f0", padx=10, pady=10)
frame_left.pack(side="left", fill="y")

frame_right = Frame(root, bg="#f0f0f0", padx=10, pady=10)
frame_right.pack(side="right", fill="both", expand=True)

# Top buttons
Button(frame_top, text="Load Image", command=load_image, bg="#0984e3", fg="white", width=15).pack(side="left", padx=10)
Button(frame_top, text="Save Image", command=save_image, bg="#6c5ce7", fg="white", width=15).pack(side="left", padx=10)
Button(frame_top, text="Undo", command=undo_image, bg="#00b894", fg="white", width=15).pack(side="left", padx=10)
Button(frame_top, text="Redo", command=redo_image, bg="#fdcb6e", fg="white", width=15).pack(side="left", padx=10)

# Left panel buttons
Label(frame_left, text="Image Adjustments", bg="#f0f0f0", font=("Arial", 14, "bold")).pack(anchor="w", pady=5)

Button(frame_left, text="Rotate", command=rotate_image, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Gaussian Blur", command=gaussian_blur, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Sepia Filter", command=sepia_filter, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Grayscale Filter", command=grayscale_filter, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Edge Detection", command=canny_edge_detection, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Resize", command=resize_image, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Histogram", command=show_histogram, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Color Histogram", command=show_color_histogram, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Brightness/Contrast", command=adjust_brightness_contrast, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Normalize Image", command=normalize_image, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)
Button(frame_left, text="Sharpen Image", command=sharpen_image, bg="#2d3436", fg="white", width=15).pack(anchor="w", pady=5)

# Image display
image_label = Label(frame_right, bg="#dfe6e9", relief="sunken", bd=2)
image_label.pack(fill="both", expand=True)

root.mainloop()