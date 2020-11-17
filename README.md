# Stormlight

## Project Members
* Harrison Mamin

## Project Description

I fine-tuned a language model on the first 3 books of The Stormlight Archive (as well as Edgedancer and Warbreaker), then incorporated it into an app that lets users generate Stormlight-esque passages in real time.

![Text Generation Example](data/movies/generation-kaladin-charged.gif)
![Text Generation Example](data/movies/generation-odium-v2.gif)
![Text Generation Example](data/movies/generation-odium.gif)

Curious if the Sanderlanche is visible in graph form? The app also lets users explore how sentiment scores vary throughout each book:

![Sentiment Score Example](data/movies/sentiment-scores.gif)

Users can also find how frequently different terms occur. This lets us investigate important questions like "which book has the most blushing?":

![Word Count Example](data/movies/counts-blush.gif)

More examples can be found in the `data/movies` directory.

## Running the App

The model is too large for any of the free deployment options I found and this is just a hobby project, so if you want to use the app you have two options:

Option 1. Run the app in Google Colab

I put together a notebook that makes the app temporarily available through Google Colab (this is the easiest way to play with the text generation model for non-technical users). Click the link below and follow the instructions (you may need to log in to a Google account first):

[Open in Colab](https://colab.research.google.com/github/hdmamin/stormlight/blob/master/notebooks/colab-app.ipynb)

Option 2. Run the app locally. 

First, clone this repo by entering the following command in your terminal:

```
cd && git clone https://github.com/hdmamin/stormlight.git && cd stormlight
```

A script is provided to help get the app up and running. It provides several useful commands (you may need to enter `chmod u+x app.sh` first to grant permissions to run the script). Start with this:

```
./app.sh build
```

This builds the app's docker image and downloads model weights (it will take several minutes but you only need to do this once). Next, use the following command to run the app. Open a web browser (I've only tried it in Chrome, but others will probably work too) and go to http://localhost:8080. The model is enormous so it will take several seconds to load all the weights.

```
./app.sh run
```

At this point, you're ready to play around with the text generation and sentiment score tabs. The word counts tab requires the full text of each book so I can't make that data publicly available. When you're done, you can stop the app with the command:

```
./app.sh stop
```

You can leave it running for as long as you want, but your computer may run slightly slower while the model's loaded in memory.

## Repo Structure
```
stormlight/
├── data         # Raw and processed data. Actual files are excluded from github.
├── notes        # Miscellaneous notes stored as raw text files.
├── notebooks    # Jupyter notebooks for experimentation and exploratory analysis.
├── reports      # Markdown reports (performance reports, blog posts, etc.)
├── bin          # Executable scripts to be run from the project root directory.
├── lib          # Python package. Code can be imported in analysis notebooks, py scripts, etc.
└── services     # Serve model predictions through a Flask/FastAPI app.
```
