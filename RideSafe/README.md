# RideSafe-500

## Dataset Description

### Dataset Summary
RideSafe-500 is a dataset of annotated dashcam videos designed specifically for detecting traffic violations involving motorized two-wheelers, such as helmet non-compliance and triple riding. The dataset was created to address the lack of publicly available resources tailored to these safety violations. It supports tasks like violation detection, traffic safety analysis, and automated E-ticket generation.

### Supported Tasks and Leaderboards
RideSafe-500 is designed to support:
- Helmet compliance detection
- Passenger count classification (single, double, triple riding)
- Two-wheeler localization in dashcam videos

Suggested metrics for evaluation include accuracy, precision, recall, and mean average precision (mAP) for object detection tasks.

### Languages
The dataset primarily features data from regions where signage and other visible text are in English and regional languages (e.g., Telugu, Urdu, Kannada, Konkani).

## Dataset Structure

### Data Instances
Each data instance consists of a video clip (average length: 20 seconds) accompanied by annotations in JSON format. Example:
```
{
  "video_id": "rs500_001",
  "annotations": [
    {
      "frame": 15,
      "bounding_box": [120, 85, 300, 250],
      "helmet_compliance": "No",
      "passenger_count": 3
    }
  ]
}
```

### Data Fields
- **video_id**: Unique identifier for each video (string)
- **frame**: Frame number for the annotation (integer)
- **bounding_box**: Coordinates of the two-wheeler in the frame (list of integers)
- **helmet_compliance**: Compliance status (Yes or No)
- **passenger_count**: Number of passengers (integer)

### Data Splits
- **Training set**: 350 videos (70%)
- **Validation set**: 100 videos (20%)
- **Test set**: 50 videos (10%)

The dataset maintains a balanced distribution of helmet compliance and triple riding scenarios across the splits.

## Dataset Creation

### Curation Rationale
RideSafe-500 was created to fill the gap in datasets tailored for two-wheeler traffic violations. Existing datasets lack specific annotations for helmet use and passenger count, making it difficult to train models for these applications.

### Source Data
The videos were collected from real-world dashcams in urban, suburban, and highway environments. Data was sourced from publicly shared dashcam footage (with permissions) and custom recordings.

### Annotations
Annotations were generated using a combination of manual labeling and semi-automated tools. The annotation process involved three annotators per video for consistency, followed by a quality-check stage.

### Personal and Sensitive Information
The dataset does not include personally identifiable information (PII) or sensitive data. Vehicle license plates and faces of riders were blurred during preprocessing to maintain privacy.

## Considerations for Using the Data

### Social Impact of Dataset
The dataset aims to enhance road safety by enabling technologies that detect and deter unsafe driving practices. Potential impacts include improved compliance with traffic laws, reduced accident rates, and safer road environments.

### Discussion of Biases
The dataset may exhibit regional biases, as the videos primarily feature traffic scenarios from Asia-Pacific regions. Helmet designs, road conditions, and vehicle types in other regions may differ, which could affect model generalization.

### Other Known Limitations
- Some annotation artifacts may exist due to low-light conditions or video compression.
- Certain edge cases, such as partially visible riders, are not comprehensively covered.

## Additional Information

### Dataset Curators
RideSafe-500 was curated by a team of researchers from XYZ University, led by [Your Name].

### Licensing Information
The dataset is released under a Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) license.

### Citation Information
To cite RideSafe-500:
```
@dataset{ridesafe500,  
  author = {Your Name and Team},  
  title = {RideSafe-500: A Dataset for Two-Wheeler Traffic Violations},  
  year = {2024},  
  publisher = {XYZ University},  
  url = {https://github.com/ridesafe500}  
}
```

### Acknowledgements
Thanks to IHub-Data, IIIT-H for supporting this work.
  
