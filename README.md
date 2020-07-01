# Body Measure

Get Measurements of body using just two pics

## Installation

Use python 3.6 or higher
download models by running getmodels.sh



## Usage

if using Anaconda
conda create --name bodyMeasure
conda activate bodyMeasure

```bash
pip install requirements.txt
```

start server 

```bash
python app.py
```
```bash
route: /api/send
```

send two images and height in body of the request.

{image1: "image1.jpg",
image2: "image2.jpg",
height: {height}}

returns a multipart responsewith  json with all the measurements and images body part markings


## License
[MIT](https://choosealicense.com/licenses/mit/)
