import string
from paddleocr import PaddleOCR
import re

# Initialize the PaddleOCR reader
# use_gpu can be set to True if GPU is available
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, "w") as f:
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                "frame_nmr",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            )
        )

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if (
                    "car" in results[frame_nmr][car_id].keys()
                    and "license_plate" in results[frame_nmr][car_id].keys()
                    and "text" in results[frame_nmr][car_id]["license_plate"].keys()
                ):
                    f.write(
                        "{},{},{},{},{},{},{}\n".format(
                            frame_nmr,
                            car_id,
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["car"]["bbox"][0],
                                results[frame_nmr][car_id]["car"]["bbox"][1],
                                results[frame_nmr][car_id]["car"]["bbox"][2],
                                results[frame_nmr][car_id]["car"]["bbox"][3],
                            ),
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["license_plate"]["bbox"][0],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][1],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][2],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][3],
                            ),
                            results[frame_nmr][car_id]["license_plate"]["bbox_score"],
                            results[frame_nmr][car_id]["license_plate"]["text"],
                            results[frame_nmr][car_id]["license_plate"]["text_score"],
                        )
                    )
        f.close()


def extract_plate_format(text):
    """
    Extract license plate in the format KF-7617 from various possible detections.
    This function will try to find the main part of the license plate,
    ignoring province codes like "NW".

    Args:
        text (str): Detected license plate text.

    Returns:
        str: Extracted license plate text in the expected format or None if not found.
    """
    # Remove all spaces and special characters
    text = text.upper().replace(" ", "").replace("-", "")

    # Try to match the pattern: 2 letters followed by 4 digits (KF7617)
    import re

    match = re.search(r"([A-Z]{2})(\d{4})", text)
    if match:
        letters = match.group(1)
        digits = match.group(2)
        return f"{letters}-{digits}"

    return None


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format (KF-7617).
    The province code (NW) is ignored.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Remove spaces and handle hyphen variations
    text = text.upper().replace(" ", "")

    # Pattern can be KF7617 or KF-7617
    if "-" in text:
        parts = text.split("-")
        if len(parts) == 2:
            # Check if first part might contain province code (e.g. "NWKF")
            if len(parts[0]) > 2:
                # Try to extract the last two characters as the main letters
                parts[0] = parts[0][-2:]
            letters, numbers = parts[0], parts[1]
        else:
            return False
    else:
        # Try to extract the format without hyphen
        # If text contains province code (e.g. "NWKF7617")
        match = re.search(r"([A-Z]+)(\d{4})$", text)
        if match:
            letters = match.group(1)[-2:]  # Take last two letters
            numbers = match.group(2)
        else:
            return False

    # Validate the format: 2 letters + 4 digits
    if (
        len(letters) == 2
        and len(numbers) == 4
        and all(
            l in string.ascii_uppercase or l in dict_int_to_char.keys() for l in letters
        )
        and all(n in "0123456789" or n in dict_char_to_int.keys() for n in numbers)
    ):
        return True

    return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    Extracts and formats plates like KF-7617, ignoring province codes.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    # Extract the main part of the license plate
    formatted_text = extract_plate_format(text)
    if not formatted_text:
        # If extraction failed, try to format the raw text
        text = text.upper().replace(" ", "")

        # If text contains a hyphen, split it
        if "-" in text:
            parts = text.split("-")
            if len(parts) == 2:
                if len(parts[0]) > 2:
                    # Take the last two characters as main letters
                    letters = parts[0][-2:]
                else:
                    letters = parts[0]
                numbers = parts[1]
            else:
                return text  # Return original if format is unclear
        else:
            # Try to extract without hyphen
            match = re.search(r"([A-Z]+)(\d{4})$", text)
            if match:
                letters = match.group(1)[-2:]  # Take last two letters
                numbers = match.group(2)
            else:
                return text  # Return original if format is unclear

        # Format letter part (first two characters)
        formatted_letters = ""
        for char in letters:
            if char in dict_char_to_int.keys():
                formatted_letters += dict_int_to_char[dict_char_to_int[char]]
            else:
                formatted_letters += char

        # Format number part (last four characters)
        formatted_numbers = ""
        for char in numbers:
            if char in dict_int_to_char.keys():
                formatted_numbers += dict_char_to_int[char]
            elif char in dict_char_to_int.keys():
                formatted_numbers += dict_char_to_int[char]
            else:
                formatted_numbers += char

        formatted_text = f"{formatted_letters}-{formatted_numbers}"

    return formatted_text


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    Optimized for KF-7617 format plates, ignoring province codes.

    Args:
        license_plate_crop (numpy.ndarray): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    # PaddleOCR returns a list of tuples: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence))]
    result = ocr.ocr(license_plate_crop, cls=True)

    # Check if result is empty or None
    if not result or not result[0]:
        return None, None

    best_text = None
    best_score = 0

    for detection in result[0]:
        _, (text, score) = detection

        # Process the text
        text = text.upper()

        # Extract the main part of the license plate if possible
        extracted_plate = extract_plate_format(text)
        if extracted_plate:
            return extracted_plate, score

        # Check if the raw text might match our format
        if license_complies_format(text):
            formatted_text = format_license(text)
            if score > best_score:
                best_text = formatted_text
                best_score = score

    if best_text:
        return best_text, best_score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
