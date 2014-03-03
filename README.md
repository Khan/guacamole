`guacamole` is a collection of tools we use at Khan Academy to train our models
from new data on a regular basis. These tools are meant to be compatible
with a variety of data formats from anyone who has learning data - especially
but not only online learning data.

**The Tools:**

We have two pipelines set up here, one for training Multi-dimensional Item Response
Theory models (with time, if you have that data) and one for looking at responses
and training a model to predict if the next item will be answered correctly, Knowledge Params.
The Item Response Theory model is probably better suited to testing data (at Khan Academy,
we use it for our assessments) and the Knowledge model is probably better suited to
online tutoring data (we use to evaluate the probability of answering the next question
in a sequence correctly)

**Guide to Using These Tools**

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

# TODO(eliana)

You can also see graphs of each exercise by difficulty and discrimination
`--sigmoid_viz`

# TODO(eliana)

Finally, You can see a visualization of how well each student did
`--student_viz`

# TODO(eliana)

And how difficult each problem is
`--problem_viz`

# TODO(eliana)

**Khan Academy Data**

This library is designed to be used on Khan Academy data. We have sample data in that format now, and if you're interested research based on our real data at scale, you can get in touch at research@khanacademy.org


If these tools are useful to you, let us know! If you'd like to contribute,
you can submit a pull request or apply to work at Khan Academy - we're hiring
software engineers and data scientists.
