# Download tessconfigs folder
git clone https://github.com/tesseract-ocr/tessconfigs.git
#TYPE can be best, fast or <empty>
TYPE=
# Remove old layer if exists
rm tesseract-layer.zip
rm tesseract-layer-fast.zip
rm tesseract-layer-best.zip
# Build Docker image containing Tesseract
set -e
docker build -t tess_layer -f Dockerfile-tess4 .
# Copy Tesseract locally
CONTAINER=$(docker run -d tess_layer false)
docker cp $CONTAINER:/opt/build-dist layer
docker rm $CONTAINER
# Zip Tesseract
cd layer/
zip -r ../tesseract-layer-$TYPE.zip .
# Clean
cd ..
rm -rf layer/
rm -rf tessconfigs/
