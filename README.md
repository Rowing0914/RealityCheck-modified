# 📦 RealityCheck Usage Guide
Original: https://sites.google.com/view/arrealitycheck/home

---

## 🖥️ Required Installed Software

* Python **3.7.0+**
* OpenCV **3.0+**
* NumPy **1.15.4+**
* Matplotlib **3.0.0+**

---

## 🚀 Command Line

### Format:

```bash
python RealityCheck.py <Video Directory Name> <Config File Name (no extension)> <True|False (use Annotations)> <True|False (Rotate images 90°)>
```

* The last two arguments are optional and **default to `False`**.
* The last argument (`Rotate`) is **mandatory** for the test video if it was filmed in portrait mode.

### Example:

```bash
python RealityCheck.py TestVideo TestConfig True True
```

* Output will be saved to:

  ```
  <Video Directory Name>Output.txt
  ```

---

## 🎥 Video Directory

* All images in the specified directory will be processed in **alphabetical order** (`sorted()`).
* **Videos must be split into individual frames** before running RealityCheck.
* The directory should contain **only image files** (important for Mac users).
* Use the rotation option if your video was recorded in portrait mode.

---

## 📝 Annotation File

### Format:

```
OFFSET: <x-offset>, <y-offset>
GRIDSIZE: <grid-square size>
<frame number>: <x squares>, <y squares>
...
```

* All values should be in **floating point**, using **meters**.
* RealityCheck expects the annotation file to have the **same name** as the video directory, with a `.txt` extension.

> 📁 A test annotation file is provided for reference.

---

## ⚙️ Config File

### Format:

```
CAMERA: <fx>,<skew>,<cx>,<0.0>,<fy>,<cy>,<0.0>,<0.0>,<1.0>  (float32)
REALDICT: <ArUco Dictionary Type>  (int)
REALBOARDARRANGEMENT: <x markers>, <y markers>  (int)
REALMARKERSIZE: <marker size in meters>  (float)
REALBOARDSPACING: <spacing in meters>  (float)
VIRTUALDICT: <ArUco Dictionary Type>  (int)
VIRTUALBOARDARRANGEMENT: <x markers>, <y markers>  (int)
VIRTUALMARKERSIZE: <marker size in meters>  (float)
VIRTUALBOARDSPACING: <spacing in meters>  (float)
```

### Dictionary Values:

| Dictionary Name       | Value |
| --------------------- | ----- |
| DICT\_4X4\_50         | 0     |
| DICT\_4X4\_100        | 1     |
| DICT\_4X4\_250        | 2     |
| DICT\_4X4\_1000       | 3     |
| DICT\_5X5\_50         | 4     |
| DICT\_5X5\_100        | 5     |
| DICT\_5X5\_250        | 6     |
| DICT\_5X5\_1000       | 7     |
| DICT\_6X6\_50         | 8     |
| DICT\_6X6\_100        | 9     |
| DICT\_6X6\_250        | 10    |
| DICT\_6X6\_1000       | 11    |
| DICT\_7X7\_50         | 12    |
| DICT\_7X7\_100        | 13    |
| DICT\_7X7\_250        | 14    |
| DICT\_7X7\_1000       | 15    |
| DICT\_ARUCO\_ORIGINAL | 16    |
| DICT\_APRILTAG\_16h5  | 17    |
| DICT\_APRILTAG\_25h9  | 18    |
| DICT\_APRILTAG\_36h10 | 19    |
| DICT\_APRILTAG\_36h11 | 20    |

> 📁 A test configuration file is provided for reference.

> 🧭 Currently, RealityCheck only supports **ArUco markers**, but the code can be extended for other types (e.g., AprilTag).

---

## 📁 File Structure Notes

* All required files — config, annotation, video frames — must be located in the **same working directory**.
