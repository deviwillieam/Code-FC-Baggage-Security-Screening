import os
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.decomposition import PCA
import image_processing  # Import functions from image_processing.py
from image_processing import classify_image

# Get list of images from the dataset using function from image_processing.py
images = image_processing.get_images()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Suspicious Object Detector")
        self.geometry("900x630")

        # Set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Load images for light and dark mode
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
        self.logo_image = ctk.CTkImage(Image.open(os.path.join(image_path, "BillionPrimaLogoReal.png")), size=(100, 26))
        self.large_test_image = ctk.CTkImage(Image.open(os.path.join(image_path, "titlelogo.png")), size=(500, 150))
        self.image_icon_image = ctk.CTkImage(Image.open(os.path.join(image_path, "image_icon_light.png")), size=(20, 20))
        self.home_image = ctk.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")),
                                       dark_image=Image.open(os.path.join(image_path, "home_light.png")), size=(20, 20))
        self.chat_image = ctk.CTkImage(light_image=Image.open(os.path.join(image_path, "searchlogodark.png")),
                                       dark_image=Image.open(os.path.join(image_path, "searchlogo.png")), size=(20, 20))
        self.add_user_image = ctk.CTkImage(light_image=Image.open(os.path.join(image_path, "add_user_dark.png")),
                                           dark_image=Image.open(os.path.join(image_path, "add_user_light.png")), size=(20, 20))

        # Create navigation frame
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="", image=self.logo_image,
                                                   compound="left", font=ctk.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
                                         fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                         image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                      command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # Create home frame
        self.home_frame = ctk.CTkScrollableFrame(self, label_text="", corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_frame_large_image_label = ctk.CTkLabel(self.home_frame, text="", image=self.large_test_image)
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)

        self.home_frame_image = ctk.CTkFrame(self.home_frame, corner_radius=0)
        self.home_frame_image.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.home_frame_image.grid_columnconfigure(0, weight=1)

        self.home_frame_button_1 = ctk.CTkButton(self.home_frame_image, text="Select and Compare Image", image=self.image_icon_image, command=self.show_images)
        self.home_frame_button_1.grid(row=0, column=0, padx=20, pady=10)

        # Labels for displaying images
        self.query_image_resized = ctk.CTkLabel(self.home_frame_image, text="")
        self.query_image_resized.grid(row=1, column=0, padx=20, pady=10)

        self.text_image_label = ctk.CTkLabel(self.home_frame_image, text="", anchor="w")
        self.text_image_label.grid(row=2, column=0, padx=20, pady=(10, 0))

        # Progress bar
        self.progressbar_1 = ctk.CTkProgressBar(self.home_frame)
        self.progressbar_1.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.progressbar_1.configure(mode="indeterminate")
        self.progressbar_1.start()

        # Create tab view
        self.tabview = ctk.CTkTabview(self.home_frame, height=150)
        self.tabview.grid(row=4, column=0, padx=(20, 20), pady=(5, 0), sticky="nsew")
        self.tabview.add("Detection")
        self.tabview.add("Differences")
        self.tabview.add("Thresh")
        self.tabview.tab("Detection").grid_columnconfigure(0, weight=1)  # Configure grid of individual tabs
        self.tabview.tab("Differences").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Thresh").grid_columnconfigure(0, weight=1)

        self.query_image_label = ctk.CTkLabel(self.tabview.tab("Detection"), text="")
        self.query_image_label.grid(row=0, column=0, padx=10, pady=10)
        self.diff_image_label = ctk.CTkLabel(self.tabview.tab("Differences"), text="")
        self.diff_image_label.grid(row=0, column=0, padx=10, pady=10)
        self.thresh_image_label = ctk.CTkLabel(self.tabview.tab("Thresh"), text="")
        self.thresh_image_label.grid(row=0, column=0, padx=10, pady=10)

        # Label for displaying detection message
        self.detection_message_label = ctk.CTkLabel(self.tabview.tab("Detection"), text="", anchor="w")
        self.detection_message_label.grid(row=1, column=0, padx=10, pady=10)

        # Select default frame
        self.select_frame_by_name("home")
        self.change_appearance_mode_event("System")

    def select_frame_by_name(self, name):
        # Update the color of the home button based on the selected frame
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")

        # Show or hide the home frame based on the selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

    def open_input_dialog_event(self):
        dialog = ctk.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def select_file(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(title="Select an image file", filetypes=(("image files", "*.png *.jpg *.jpeg"), ("all files", "*.*")))
        return file_path

    def show_images(self):
        # Get the path of the selected image file
        query_image_path = self.select_file()
        if query_image_path:
            query_image_path = os.path.abspath(query_image_path)

            # Classify the selected image
            predicted_class_name, confidence = classify_image(query_image_path)
            print(f"Class: {predicted_class_name}, Confidence: {confidence:.2f}%")

            # Display the classification result
            result_text = f"Class: {predicted_class_name}, Confidence: {confidence:.2f}%"
            self.text_image_label.configure(text=result_text)

            # Load and preprocess the selected image
            _, query_image_array = image_processing.load_image(query_image_path)
            query_features = image_processing.feat_extractor.predict(query_image_array)
            query_features = query_features.reshape(1, -1)

            # Find the closest image in the dataset
            idx_closest = image_processing.get_closest_images(query_features)
            results_image_idx = idx_closest[0]
            results_image_path = images[results_image_idx]

            # Read the query and result images
            result_image = cv2.imread(results_image_path)
            query_image = cv2.imread(query_image_path)
            query_image_resized = cv2.resize(query_image, (result_image.shape[1], result_image.shape[0]))
            query_image_resized1 = cv2.resize(query_image, (result_image.shape[1], result_image.shape[0]))

            # Convert images to grayscale
            grayA = cv2.cvtColor(query_image_resized, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

            # Compute SSIM between the two images
            (score, diff) = compare_ssim(grayA, grayB, full=True, data_range=255)
            diff = (diff * 255).astype("uint8")
            print("SSIM: {}".format(score))

            # Threshold the difference image
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Find contours in the threshold image
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            suspicious_objects_detected = False

            # Draw rectangles around suspicious objects
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(query_image_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                suspicious_objects_detected = True

            # Display images and update detection message
            self.display_image(query_image_resized1, query_image_resized, results_image_path, diff, thresh, suspicious_objects_detected)

    def display_image(self, query_image_resized1, query_image_resized, results_image_path, diff, thresh, suspicious_objects_detected):
        # Convert images to RGB format
        query_image_resized = cv2.cvtColor(query_image_resized, cv2.COLOR_BGR2RGB)
        result_image = cv2.cvtColor(cv2.imread(results_image_path), cv2.COLOR_BGR2RGB)
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        # List of images and labels for display
        images = [query_image_resized1, query_image_resized, diff, thresh]
        labels = [self.query_image_resized, self.query_image_label, self.diff_image_label, self.thresh_image_label]

        # Display images on the corresponding labels
        for i in range(len(images)):
            img = Image.fromarray(images[i])
            img = ImageTk.PhotoImage(img)
            labels[i].configure(image=img)
            labels[i].image = img

        # Update detection message
        if suspicious_objects_detected:
            self.detection_message_label.configure(text="Suspicious object detected!", text_color="red")
        else:
            self.detection_message_label.configure(text="No suspicious object detected.", text_color="green")

if __name__ == "__main__":
    app = App()
    app.mainloop()
