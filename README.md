This is a collection of tools we use at Khan Academy to train our models
from new data on a regular basis. These tools are meant to be compatible
with a variety of data formats from anyone who has learning data - especially
but not only online learning data.

The Tools:

We have two pipelines set up here, one for training Multi-dimensional Item Response
Theory models (with time, if you have that data) and one for looking at responses
and training a model to predict if the next item will be answered correctly, Knowledge Params.
The Item Response Theory model is probably better suited to testing data (at Khan Academy,
we use it for our assessments) and the Knowledge model is probably better suited to
online tutoring data (we use to evaluate the probability of answering the next question
in a sequence correctly)

Guide to Using These Tools

Quickstart guide: I've included here some simple data in csv format that you can
use to train either model. If you call either of the start_x_pipeline scripts,
it will train on those data. Feel free to substitute your own data

If these tools are useful to you, let us know! If you'd like to contribute,
you can submit a pull request or just apply to work at Khan Academy - we're hiring
software engineers and data scientists.
