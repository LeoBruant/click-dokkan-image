import cv2
import glob
import numpy as np
import pyautogui
import time
from dataclasses import dataclass
from mss import mss
from pathlib import Path
from PIL import Image
from termcolor import colored
from typing import Optional, Tuple, Union

class Config:
    """Configuration constants."""
    IMAGE_SIMILARITY_THRESHOLD = 0.55
    TARGETS_DIRECTORY = 'targets'
    TIMEOUT = 0.375

@dataclass
class ImageData:
    """Data class for storing image data."""
    path: Path
    data: np.ndarray

def click_image(image_data: ImageData) -> bool:
    """Clicks the image if it is found on the screen."""
    location = find_image_on_screen(image_data)

    if location is not None:
        center_x = location[0] + image_data.data.shape[1] / 2
        center_y = location[1] + image_data.data.shape[0] / 2

        screen_width, screen_height = pyautogui.size()
        center_x = min(max(center_x, 0), screen_width - 1)
        center_y = min(max(center_y, 0), screen_height - 1)

        time.sleep(Config.TIMEOUT)
        pyautogui.click(center_x, center_y)
        time.sleep(Config.TIMEOUT * 2)
        pyautogui.click(center_x, center_y)
        return True

    return False

def find_image_on_screen(image_data: ImageData) -> Optional[Tuple[int, int]]:
    """Finds the image on the screen and returns its location."""
    with mss() as sct:
        screenshot = sct.grab(sct.monitors[0])
    screenshot_img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    screenshot_np = np.array(screenshot_img)

    match_result = cv2.matchTemplate(screenshot_np, image_data.data, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(match_result)

    print(
        colored(f'Best similarity for "{image_data.path.name}"', 'yellow')
        + " | " +
        colored(f'{round(max_val * 100, 2)} %', 'blue')
    )

    if max_val >= Config.IMAGE_SIMILARITY_THRESHOLD:
        return max_loc

    return None

def load_and_convert_image(image_source: Union[str, Image.Image]) -> ImageData:
    """Loads and converts the image source into ImageData."""
    if isinstance(image_source, str):
        try:
            image = cv2.imread(image_source, cv2.IMREAD_COLOR)
            path = Path(image_source)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {image_source} not found.")
    else:
        image = np.array(image_source)
        path = None

    return ImageData(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def auto_battle() -> None:
    """Automatically battles by clicking on images."""
    image_files = sorted(glob.glob(f"{Config.TARGETS_DIRECTORY}/*.png"))
    images = [load_and_convert_image(img) for img in image_files if img is not None]

    while True:
        for image_data in images:
            while not click_image(image_data):
                time.sleep(Config.TIMEOUT)
            print("\n" + colored(f'Clicked on "{image_data.path.name}" !', 'green') + "\n")

def main():
    """Main function to execute the script."""
    auto_battle()

if __name__ == "__main__":
    main()
