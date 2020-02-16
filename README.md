# tesseract-lambda-layer


## Fast setup

clone the repo
cd .../repo

bash build_py37_pkgs.sh
bash build_tesseract4.0.0

Upload manually layers to Aws lambda layers
Check env to python 3.7

Create lambda function
environment variable / memory / timeout / code

Test lambda

Create Api
new method POST
enable CORS

Test Api if OK deploy

Test Api in Postman

Done !

#### Check Medium Blog for all th steps in details

# Credits

[Ocr Layer] https://github.com/bweigel/aws-lambda-tesseract-layer
[Python libraries to layers] https://github.com/tiivik/LambdaZipper 
[python3.7 lambda] https://github.com/lambci/docker-lambda
