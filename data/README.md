===== Place Pulse 2.0 README =====
======== October 10, 2016 ========
========== MIT Media Lab =========

The votes.csv file contains the pairwise vote data used for the study "Deep Learning the City : Quantifying Urban Perception At a Global Scale". The votes.csv has the following fields:

left_id: Serialized ID for the left image.
---
right_id: Serialized ID for the right image.
---
winner : One of {left,right,equal}, indicating which image was voted for in this comparison. 'equal' denotes that both were rated equally.
---
left_lat : Latitude for the left image.
left_long : Longitude for the left image.
---
right_lat : Latitude for the right image.
right_long : Longitude for the left image.
---
category : Category the vote belongs to, one of {safety, beautiful, lively, wealthy, boring, depressing}.

================== 

Notes:

To download the images, you can use Google's StreetView API. A sample request is :

https://maps.googleapis.com/maps/api/streetview?size=400x300&location=42.3629288,-71.0930129

For our experiments, the size of 400x300 was used. We did not specify the heading, pitch and field-of-view (fov) values in our requests. Additionally, the equality votes are also included in the release, but were not considered for our model, which might give some small differences in the number of votes present per attribute.

Citation:

If you find this dataset useful for your research, we request you to cite our recent paper:

Dubey, A., Naik, N., Parikh, D., Raskar, R., & Hidalgo, C. A. (2016, October). Deep learning the city: Quantifying urban perception at a global scale. In European Conference on Computer Vision (pp. 196-212). Springer International Publishing.

Bibtex:

@inproceedings{dubey2016deep,
  title={Deep learning the city: Quantifying urban perception at a global scale},
  author={Dubey, Abhimanyu and Naik, Nikhil and Parikh, Devi and Raskar, Ramesh and Hidalgo, C{\'e}sar A},
  booktitle={European Conference on Computer Vision},
  pages={196--212},
  year={2016},
  organization={Springer}
}