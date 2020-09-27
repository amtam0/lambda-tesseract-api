set -e
rm -rf tesseract-layer.zip exit 0
# Download tessconfigs folder
git clone https://github.com/tesseract-ocr/tessconfigs.git tesseractconfigs
mv tesseractconfigs/configs .
mv tesseractconfigs/tessconfigs .
rm -rf tesseractconfigs
# Build Docker image containing Tesseract
docker build -t tess_layer -f Dockerfile-tess4 .
# Copy Tesseract locally
CONTAINER=$(docker run -d tess_layer false)
docker cp $CONTAINER:/opt/build-dist layer
docker rm $CONTAINER
# # Zip Tesseract
cd layer/
zip -r ../tesseract-layer.zip .
# Clean
cd ..
rm -rf layer/
rm -rf tessconfigs/
rm -rf configs/
