## Fast setup of OCR lambda function using Tesseract 5 and a custom OCR (here we use PaddleOCR ONNX version)

### Setup:

- clone repo

- create ECR repo in your AWS / copy its URI and add it to `zip_fct.sh` #line 27/28

- connect if not done `aws ecr get-login-password --region yourREGION | docker login --username AWS --password-stdin yourURI`

- run ```cd lambda-tesseract-api/; bash zip_fct.sh```

Done ! Your ECR image is ready to be uploaded from your lambda function (you can use the `example.json` to test it).

**Notes** :
- Docker must be installed, tested in Ubuntu 20.04.
- Here we do only the Recognition part, You can edit OCR fcts in `lambda_function.py` for your needs.

Check [Medium link](https://medium.com/analytics-vidhya/build-tesseract-serverless-api-using-aws-lambda-and-docker-in-minutes-dd97a79b589b?source=friends_link&sk=5c1c6948bc1a6c2a7e918e0874bf80c9) to setup lambda and Api in AWS console. Not updated (the lambda setup is easier now, you only need to upload the Image from ECR).

## References
- [Tesseract Layer](https://github.com/bweigel/aws-lambda-tesseract-layer)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR onnx/inference setup](https://github.com/amtam0/PADDLE-ONNX)