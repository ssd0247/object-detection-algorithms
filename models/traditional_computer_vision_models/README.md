# TRADITIONAL COMPUTER-VISION SUPPORTED OBJECT DETECTION MODELS

---

* All the models in this directory are composed of two separately trained units.
    
    1. `Segregated image-stub classification`
        - Classify the image stub it receives from upstream preprocessing steps.
        - Gets images that are potential regions extracted by sliding the *context window* across each image.
        - *Each image* above refers to the multiple avatars of the single, original image derived by *scaling*, to allow the detection of objects of all dimensions present in a single image using the same sliding window.
        - The potential region have now become regions of interest, unsurprisingly named as ***Region-Of-Interest (ROI)***.

    2. `Bounding Box Prediction`
        - The ROIs one gets from those classifications that surpass the **Confidence-Threshold***(a hyperparameter)* are juxtaposed onto the original image, to contrast and compute the *Area-Of-Overlap*.
        - If the area of overlap exceeds the **Overlap-Threshold***(another hyperparameter)* than the regions are squashed together into one, thus decreasing the number of redundant bounding boxes and keeping only the most relevant ones!
        - Note that, only those bounding boxes are merged together, that have the highest confidence for the same class.
        - Probability of the class can be taken as the average of all the probabilities from individual boxes. Though other methods to compute the *best probability* can also be applied.
        - This gives us the objects segregated from others by their respective bounding boxes and annotated with class labels, along with the class probabilities.

# ALGORITHMS UTILIZED

---

> Before passing the image to the Classifier
>>
>> [Sliding Window Technique](http://www.cs.utoronto.ca/~fidler/slides/CSC420/lecture17.pdf)
>>
>> [Image Pyramid](https://en.wikipedia.org/wiki/Pyramid_(image_processing)#:~:text=Pyramid%2C%20or%20pyramid%20representation%2C%20is,to%20repeated%20smoothing%20and%20subsampling.)



> After classification returned high confidence score for a particular label, at multiple scales and sliding steps
>>
>> [Non Maximum Suppression](https://paperswithcode.com/method/non-maximum-suppression#:~:text=Non%20Maximum%20Suppression%20is%20a,below%20a%20given%20probability%20bound.)
