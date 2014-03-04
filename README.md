`guacamole` is a collection of tools we use at Khan Academy to train our models
from new data on a regular basis. These tools are meant to be compatible
with a variety of data formats from anyone who has learning data - especially
but not only data from online instruction.

**The Tools:**

Two pipelines are included here.  One trains Multi-dimensional Item Response
Theory (MIRT) models, including both item correctness and response time if you
have that data.  The other, called Knowledge Params, trains a classifier with a mixture of random and hand designed features to predict whether the the next
item will be answered correctly based upon the response history. The MIRT model is probably better suited to testing data (at Khan Academy, we use it for our assessments) and the Knowledge model is probably better suited to online tutoring data (we use it to evaluate the probability of answering the next question in a sequence correctly)

**Getting Started for Learning@Scale attendees**

To use this library, you'll need to have numpy, scipy, and matplotlib installed on your machine. If this is not already the case, I recommend using the [Scipy Superpack](http://fonnesbeck.github.io/ScipySuperpack/) for Mac, or following the [SciPy Stack installation instructions](http://www.scipy.org/install.html) for Linux or Windows.


To generate some fake data, run the command
`./start_mirt_pipeline.py --generate`

To start looking at visualizations for that data, first you'll have to train a model. Try running
`./start_mirt_pipeline.py --train`
to train.

To visualize the results of your training, look at

`./start_mirt_pipeline.py --visualize`

**Guide to Using the Item Response Theory Tool**

You can use the guacamole item response theory tool to generate and examine
testing data.


To generate feedback
`./start_mirt_pipeline.py --generate`

To train a model
`./start_mirt_pipeline.py --train`

To visualize a mode
`./start_mirt_pipeline.py --visualize`

To take an adaptive test from a trained model
`./start_mirt_pipeline.py --test`

To score several taken tests test from a trained model
`./start_mirt_pipeline.py --score`


**The Algorithms**

`guacamole` aspires to be a general purpose library with a spectrum of commonly used algorithms for analyzing educational data (especially at scale). For now, we support a few common algorithms.

*Multidimensional Item Response Theory*

Item response theory is a classic technique in psychometrics to calibrate tests and test items with student abilities, resulting in difficulty ratings for test items and ability ratings for students.

**Visualizations**

A few visualizations are available for the data.

First, you can see an ROC curve given your parameters:
`--roc_viz`

![ROC curve](imgs/roc.png)

You can also see graphs of each exercise by difficulty and discrimination
`./start_mirt_pipeline.py --sigmoid_viz`

![sigmoids](imgs/sigmoids.png)

To see how well each student did, call
'./start_mirt_pipeline.py --score'

**The names**

The names are from the US census bureau.

**Khan Academy Data**

This library is designed to be used on Khan Academy data. We have sample, non-student, data in that format now.
If you are interested in using our real data at scale in your research, you should visit [http://khanacademy.org/r/research](http://khanacademy.org/r/research), and then email us at [research@khanacademy.org](mailto:research@khanacademy.org).


If these tools are useful to you, let us know! If you'd like to contribute,
you can submit a pull request or
[apply to work at Khan Academy](https://www.khanacademy.org/careers) - we're hiring data
scientists and software engineers for both full time positions and internships.

Authors: Eliana Feasley, Jace Kohlmeier, Matt Faus, Jascha Sohl-Dickstein (2014)

This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/4.0/ )
