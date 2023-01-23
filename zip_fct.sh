set -e
rm *.zip || true
rm -rf layer/ || true
rm -rf tessconfigs/ || true
rm -rf tesseractconfigs/ || true
rm -rf configs/ || true
# Download tessconfigs folder
git clone https://github.com/tesseract-ocr/tessconfigs.git tesseractconfigs
mv tesseractconfigs/configs .
mv tesseractconfigs/tessconfigs .
rm -rf tesseractconfigs || true
# Build Tesseract LAYER
docker build -t tess_layer -f Dockerfile-tess5.1.0 .
CONTAINER=$(docker run -d tess_layer false)
docker cp $CONTAINER:/opt/build-dist layer
docker rm $CONTAINER
cd layer/
zip -r ../tesseract-layer5.1.0best.zip .
# Clean
cd ..
rm -rf layer/ || true 
rm -rf tessconfigs/ || true
rm -rf tesseractconfigs/ || true
rm -rf configs/ || true
#Build docker image and push to ECR
docker build -t  lambda-ocr .
docker tag lambda-ocr:latest yourURI:v0.1
docker push yourURI:v0.1