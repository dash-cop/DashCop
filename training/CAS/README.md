## CAS - Cross Association and Segmentation

This repository contains the implementation of **CAS (Cross Association and Segmentation)**, as described in the paper *"DashCop: Automated E-ticket Generation for Two-Wheeler Traffic Violations Using Dashcam Videos"*. 

The primary goal of this repository is to train a joint segmentation and association model for two data classes that exhibit a correspondence relationship, such as:

- **Rider and Bike**
- **Person and Skateboard**
- **Person and Chair**

The model achieves the following:

1. **Segmentation:** Identifies and segments all objects in the scene corresponding to the two specified classes.
2. **Association:** For every object of class A, predicts the segmentation mask of its associated object in class B, and vice versa.

---

## Data Setup

### Annotation File Format
The annotation file format is similar to the standard YOLO format, with an additional attribute to specify the association ID (*assoc_id*) for each object. This ID establishes the correspondence between objects of the two classes.

#### YOLO Format Recap:
In the YOLO format, each row in an annotation file represents an object and is structured as:

```
<cls> <xn1> <yn1> <xn2> <yn2> ...
```

Here:
- `<cls>`: The class ID of the object.
- `<xn1>, <yn1>`: The normalized x and y coordinates of the first point of the object’s segmentation contour.
- `<xn2>, <yn2>`: The normalized x and y coordinates of subsequent contour points.

#### CAS Annotation Format:
For CAS, the annotation format extends the YOLO structure by including an additional field, `<assoc_id>`:

```
<cls> <assoc_id> <xn1> <yn1> <xn2> <yn2> ...
```

- `<assoc_id>`: An integer ID that links an object of class A to its corresponding object in class B.

---

### File Structure
Ensure that the dataset is organized into a folder structure where each image has a corresponding annotation file. For instance:

```
a.jpg, a.txt
b.jpg, b.txt
```

The `.txt` file should contain annotations in the correct CAS annotation format described above.

#### Dataset Split
Prepare at least two folders for splitting the dataset:
- One for **training**
- One for **validation**

For example:
```
/train
    image1.jpg
    image1.txt
    image2.jpg
    image2.txt

/val
    image3.jpg
    image3.txt
    image4.jpg
    image4.txt
```

Example annotation files are given in the dataset folder.
---

## Training

### Configuration
Once the annotation files are prepared, update the `config.yaml` file:
- Specify the paths for the training, validation, and (optionally) test folders.
- Define the class names in the `names` section.

### Running the Training Script
Run the training process using the command:

```
python3 train.py
```

The `train.py` script supports arguments similar to the Ultralytics YOLOv8 codebase. For detailed usage, refer to the inline comments and documentation provided in the script.

During training, an additional assoc loss will be visible 

---