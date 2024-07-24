# edcl-and-research

This repository hosts my work as a research assistant for Professor Kent Moore at the University of Toronto. Out of this work results two unfinished products: a package (edcl) for abstracting and plotting Earth data, and scripts leveraging this package for research into the extremity of tip jets off Cape Farewell, Greenland.

Note, the source code of the package likely does not contain all required citations. It is not publish-ready.


## The Package
The edcl, or Earth Data Collection package abstracts Earth Data with an object-oriented fashion to simplify statistics on and visualization of Earth data. The name comes from the primary class, `DataCollection`. Which stores metadata and data itself for a variable, time period and region.

## The Research
### _Abstract:_
Greenland's topography prevents winds from travelling over its surface. Instead, they speed up along its coast (Moore & Renfrew, 2005). The tip jet off Cape Farewell on August 19, 2022, at 12H is of particular interest, appearing more extreme than seen before. Using satellite data and numerical weather prediction models, we characterize this event. Both point and area-based statistics are used to find that the event consists of high percentile winds in an uncommon contiguous region of roughly 395,402 square kilometers. 181,248 hourly samples of winds in this region were analysed by computing the fraction of points whose speed met or exceeded the 98th percentile at that point. Full coverage was found only on the event of interest, while 60 samples had coverage at or above 90%, and 855 had coverage at or above 50%.

