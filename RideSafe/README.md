# RideSafe-400

## Dataset Description

### Dataset Summary
RideSafe-400 is a dataset of annotated dashcam videos designed specifically for detecting traffic violations involving motorized two-wheelers, such as helmet non-compliance and triple riding. The dataset was created to address the lack of publicly available resources tailored to these safety violations. It supports tasks like violation detection, traffic safety analysis, and automated E-ticket generation.

### Supported Tasks and Leaderboards
RideSafe-400 is designed to support:
- Helmet compliance detection
- Passenger count classification (single, double, triple riding)
- Two-wheeler localization in dashcam videos

Suggested metrics for evaluation include accuracy, precision, recall, and mean average precision (mAP) for object detection tasks.

### Languages
The dataset primarily features data from regions where signage and other visible text are in English and regional languages (e.g., Telugu, Urdu, Kannada, Konkani).

## Dataset Structure

### Data Instances
Each data instance consists of a video clip accompanied by annotations in XML format (CVAT for Video 1.1). Example:
```
<annotations>
  <track id="0" label="rider">
    <box frame="275" xtl="1793.60" ytl="215.50" xbr="1881.30" ybr="298.10">
      <attribute name="association_id">196</attribute>
    <\box>
    <polyline frame="275" points="1098.00,224.00;1097.00,225.00;1096.00,225.00;1092.00...">
      <attribute name="association_id">196</attribute>
    <\box>
    ...
  <\track>
  <track id="1" label="motorcycle">
    <box frame="275" xtl="1046.10" ytl="226.10" xbr="1075.90" ybr="268.10">
      <attribute name="motor_track_id">196</attribute>
    <\box>
    <polyline frame="275" points="1069.00,225.00;1068.00,226.00;1066.00,226.00;1065.00...">
      <attribute name="motor_track_id">196</attribute>
    <\box>
    ...
  <\track>
  ...
<\annotations>
```

### Data Fields
- **label**: The annotation class (rider, motorcycle, helmet, no-helmet, license plate)
- **frame**: Frame number for the annotation (integer)
- **xtl**, **ytl**, **xbr**, **ybr**: Coordinates of the rider and motorcycle
- **association_id** is an attribute of the `rider` bbox
- **motor_track_id** is an attribute of the `motorcycle` bbox
- If **association_id** is equal to **motor_track_id**, it means that rider belongs to the corresponding motorcycle

### Data Splits
- **Training set**: 300 videos (70%)
- **Validation set**: 50 videos (20%)
- **Test set**: 50 videos (10%)

The dataset maintains a balanced distribution of helmet compliance and triple riding scenarios across the splits.

## Dataset Creation

### Curation Rationale
RideSafe-400 was created to fill the gap in datasets tailored for two-wheeler traffic violations. Existing datasets lack specific annotations for helmet use and passenger count, making it difficult to train models for these applications.

### Source Data
The videos were collected from real-world dashcams in urban, suburban, and highway environments. Data was sourced from publicly shared dashcam footage (with permissions) and custom recordings.

### Annotations
Annotations were generated using a combination of manual labeling and semi-automated tools. The annotation process involved three annotators per video for consistency, followed by a quality-check stage.

## Considerations for Using the Data

### Social Impact of Dataset
The dataset aims to enhance road safety by enabling technologies that detect and deter unsafe driving practices. Potential impacts include improved compliance with traffic laws, reduced accident rates, and safer road environments.

### Discussion of Biases
The dataset may exhibit regional biases, as the videos primarily feature traffic scenarios from Asia-Pacific regions. Helmet designs, road conditions, and vehicle types in other regions may differ, which could affect model generalization.

## Additional Information

### Dataset Access
Reach out to `deepti.rawat@research.iiit.ac.in` for queries.

### Licensing Information
The dataset is released under a Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) license.

### Acknowledgements
Thanks to IHub-Data, IIIT-H for supporting this work.
